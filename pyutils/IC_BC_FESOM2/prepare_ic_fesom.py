#!/usr/bin/env python3
"""
FESOM-C Initial Conditions Preparation Tool

This script prepares initial conditions (IC) for the FESOM-C ocean model
from FESOM2 output data. It reads FESOM2 3D temperature and salinity data
on an unstructured mesh, interpolates it to a regular grid, performs
optional smoothing and stability checks, and outputs a NetCDF file that
FESOM-C can read.

Author: Generated with Claude Code
Date: 2026-01-11
"""

import os
import sys
import argparse
import numpy as np
import xarray as xr
from netCDF4 import Dataset
from scipy.spatial import cKDTree
from scipy.signal import savgol_filter
from scipy.interpolate import griddata
import pyproj
import gsw
import yaml
from typing import Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')


def load_config(config_file: str = 'config.yaml') -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_file: Path to YAML configuration file

    Returns:
        Dictionary containing configuration parameters
    """
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file not found: {config_file}")

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    return config


def read_fesomc_mesh(nodes_file: str, config: Dict[str, Any]) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Read FESOM-C mesh from ASCII file.

    The file format is:
    - First line: number of nodes
    - Following lines: index longitude latitude boundary_flag

    Args:
        nodes_file: Path to the nodes file (nod2d.out)
        config: Configuration dictionary

    Returns:
        Tuple containing:
            - nod2d (int): Total number of nodes
            - coord_nod2d (np.ndarray): Coordinates of nodes, shape (nod2d, 2) [lon, lat]
            - nodes_obindex (np.ndarray): Open boundary indices
    """
    if not os.path.exists(nodes_file):
        raise FileNotFoundError(f"Nodes file not found: {nodes_file}")

    print(f"Reading FESOM-C mesh from {nodes_file}...")

    # Constants for coordinate conversion
    r_earth = 6400000.0  # Earth radius in meters
    rad = np.pi / 180.0  # Degrees to radians conversion

    with open(nodes_file, 'r') as f:
        # Read total number of nodes
        nod2d = int(f.readline().strip())

        # Initialize arrays
        coord_nod2d = np.empty((nod2d, 2), dtype=float)  # [lon, lat]
        nodes_obindex = np.empty(nod2d, dtype=int)

        # Read node data
        for i in range(nod2d):
            line = f.readline().strip()
            parts = line.split()

            # Note: Index in file is 1-based (Fortran), but we use 0-based (Python)
            # parts[0] is the node index (not used here, we rely on line order)
            longitude = float(parts[1])
            latitude = float(parts[2])
            obindex = int(parts[3])

            coord_nod2d[i, :] = [longitude, latitude]
            nodes_obindex[i] = obindex

    # Apply coordinate transformation if needed
    if config["mesh"]["is_cartesian"]:
        # Convert from Cartesian to spherical
        coord_nod2d /= r_earth
    else:
        # Convert degrees to radians if working in radians
        # (Keeping in degrees for now as it's more common)
        pass

    print(f"  Read {nod2d} nodes")
    print(f"  Longitude range: [{coord_nod2d[:, 0].min():.4f}, {coord_nod2d[:, 0].max():.4f}]")
    print(f"  Latitude range: [{coord_nod2d[:, 1].min():.4f}, {coord_nod2d[:, 1].max():.4f}]")

    return nod2d, coord_nod2d, nodes_obindex


def read_depth_file(depth_file: str, nod2d: int) -> np.ndarray:
    """
    Read bathymetry depths from ASCII file.

    Args:
        depth_file: Path to depth file (depth.out)
        nod2d: Expected number of nodes

    Returns:
        node_depth: Array of depths at each node, shape (nod2d,)
    """
    if not os.path.exists(depth_file):
        raise FileNotFoundError(f"Depth file not found: {depth_file}")

    print(f"Reading node depths from {depth_file}...")

    node_depth = np.empty(nod2d, dtype=float)

    with open(depth_file, 'r') as f:
        for i in range(nod2d):
            line = f.readline().strip()
            node_depth[i] = float(line)

    print(f"  Depth range: [{node_depth.min():.1f}, {node_depth.max():.1f}] m")

    return node_depth


def create_regular_grid(coord_nod2d: np.ndarray, config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a regular lon-lat grid that covers the FESOM-C mesh.

    Args:
        coord_nod2d: Node coordinates, shape (nod2d, 2) [lon, lat]
        config: Configuration dictionary

    Returns:
        Tuple containing:
            - lonreg: 1D array of longitude values
            - latreg: 1D array of latitude values
            - lonreg2d: 2D meshgrid of longitudes
            - latreg2d: 2D meshgrid of latitudes
    """
    print("Creating regular grid...")

    # Get mesh bounds with margin
    margin = config["regular_grid"]["margin_degrees"]
    lon_min = coord_nod2d[:, 0].min() - margin
    lon_max = coord_nod2d[:, 0].max() + margin
    lat_min = coord_nod2d[:, 1].min() - margin
    lat_max = coord_nod2d[:, 1].max() + margin

    # Define the bounding box
    box = [lon_min, lon_max, lat_min, lat_max]
    left, right, down, up = box

    # Calculate grid resolution
    resolution = config["regular_grid"]["resolution"]

    if config["regular_grid"]["use_resolution_in_meters"]:
        # Calculate number of grid points from resolution in meters
        wgs84_geod = pyproj.Geod(ellps='WGS84')

        # Distance in x-direction (longitude)
        dist_x = wgs84_geod.inv(left, (up + down) * 0.5, right, (up + down) * 0.5)[2]
        lon_number = round(dist_x / resolution[0])

        # Distance in y-direction (latitude)
        dist_y = wgs84_geod.inv((left + right) * 0.5, down, (left + right) * 0.5, up)[2]
        lat_number = round(dist_y / resolution[1])
    else:
        # Use resolution as number of points
        lon_number, lat_number = resolution

    # Create 1D coordinate arrays
    lonreg = np.linspace(left, right, lon_number)
    latreg = np.linspace(down, up, lat_number)

    # Create 2D meshgrid
    lonreg2d, latreg2d = np.meshgrid(lonreg, latreg)

    print(f"  Grid dimensions: {lon_number} x {lat_number}")
    print(f"  Longitude: [{lonreg[0]:.4f}, {lonreg[-1]:.4f}], step: {(lonreg[1]-lonreg[0]):.4f}")
    print(f"  Latitude: [{latreg[0]:.4f}, {latreg[-1]:.4f}], step: {(latreg[1]-latreg[0]):.4f}")

    return lonreg, latreg, lonreg2d, latreg2d


def read_fesom2_data(config: Dict[str, Any], coord_nod2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Read FESOM2 output data (temperature, salinity, coordinates, depths).

    Only loads data from nodes within the region of interest to save memory.

    Args:
        config: Configuration dictionary
        coord_nod2d: FESOM-C mesh coordinates for determining region bounds

    Returns:
        Tuple containing:
            - temp: Temperature array, shape (nz, nnodes_subset)
            - salt: Salinity array, shape (nz, nnodes_subset)
            - nodes_lon: Node longitudes, shape (nnodes_subset,)
            - nodes_lat: Node latitudes, shape (nnodes_subset,)
            - depth: Depth levels, shape (nz,)
    """
    print("Reading FESOM2 data...")

    data_dir = config["input"]["fesom2_data_dir"]
    time_idx = config["input"]["fesom2_time_index"]

    # Determine region of interest from FESOM-C mesh with extra margin
    margin = config["regular_grid"]["margin_degrees"] + 2.0  # Extra margin for safety
    lon_min = coord_nod2d[:, 0].min() - margin
    lon_max = coord_nod2d[:, 0].max() + margin
    lat_min = coord_nod2d[:, 1].min() - margin
    lat_max = coord_nod2d[:, 1].max() + margin

    print(f"  Region of interest: lon=[{lon_min:.2f}, {lon_max:.2f}], lat=[{lat_min:.2f}, {lat_max:.2f}]")

    # Read mesh coordinates first to filter nodes
    mesh_file = os.path.join(data_dir, config["input"]["fesom2_mesh_file"])
    print(f"  Reading mesh coordinates from {mesh_file}...")
    ncfile = xr.open_dataset(mesh_file)
    nodes_lon_full = ncfile.lon.values
    nodes_lat_full = ncfile.lat.values
    ncfile.close()
    print(f"  Total FESOM2 nodes: {len(nodes_lon_full)}")

    # Filter nodes within region of interest
    print("  Filtering nodes within region...")
    mask_region = (
        (nodes_lon_full >= lon_min) & (nodes_lon_full <= lon_max) &
        (nodes_lat_full >= lat_min) & (nodes_lat_full <= lat_max)
    )
    node_indices = np.where(mask_region)[0]
    nodes_lon = nodes_lon_full[mask_region]
    nodes_lat = nodes_lat_full[mask_region]
    print(f"  Filtered to {len(nodes_lon)} nodes in region ({100*len(nodes_lon)/len(nodes_lon_full):.1f}% of total)")

    if len(nodes_lon) == 0:
        raise ValueError("No FESOM2 nodes found in the region of interest! Check coordinate systems.")

    # Read temperature for filtered nodes only
    temp_file = os.path.join(data_dir, config["input"]["fesom2_temp_file"])
    print(f"  Reading temperature from {temp_file}...")
    data = xr.open_dataset(temp_file)
    temp = data.temp[time_idx, :, node_indices].values  # Shape: (nz, nnodes_subset)
    depth = data.nz1.values  # Depth levels
    data.close()
    print(f"  Temperature shape: {temp.shape}, depth levels: {len(depth)}")

    # Read salinity for filtered nodes only
    salt_file = os.path.join(data_dir, config["input"]["fesom2_salt_file"])
    print(f"  Reading salinity from {salt_file}...")
    data = xr.open_dataset(salt_file)
    salt = data.salt[time_idx, :, node_indices].values  # Shape: (nz, nnodes_subset)
    data.close()
    print(f"  Salinity shape: {salt.shape}")

    # Filter invalid data
    temp_min = config["data_filtering"]["temp_min"]
    temp_max = config["data_filtering"]["temp_max"]
    salt_min = config["data_filtering"]["salt_min"]
    salt_max = config["data_filtering"]["salt_max"]

    temp[temp < temp_min] = np.nan
    temp[temp > temp_max] = np.nan
    salt[salt < salt_min] = np.nan
    salt[salt > salt_max] = np.nan

    # Fill missing values at depth with last valid shallow value
    print("  Filling missing values with last valid shallow value...")
    for i in range(temp.shape[1]):  # Loop over nodes
        # Temperature
        valid_idx = ~np.isnan(temp[:, i])
        if np.any(valid_idx):
            last_valid_idx = np.where(valid_idx)[0][-1]
            temp[last_valid_idx+1:, i] = temp[last_valid_idx, i]

        # Salinity
        valid_idx = ~np.isnan(salt[:, i])
        if np.any(valid_idx):
            last_valid_idx = np.where(valid_idx)[0][-1]
            salt[last_valid_idx+1:, i] = salt[last_valid_idx, i]

    print(f"  Data range - Temperature: [{np.nanmin(temp):.2f}, {np.nanmax(temp):.2f}] °C")
    print(f"  Data range - Salinity: [{np.nanmin(salt):.2f}, {np.nanmax(salt):.2f}] PSU")

    return temp, salt, nodes_lon, nodes_lat, depth


def interpolate_to_regular_grid(
    temp: np.ndarray,
    salt: np.ndarray,
    nodes_lon: np.ndarray,
    nodes_lat: np.ndarray,
    depth: np.ndarray,
    lonreg2d: np.ndarray,
    latreg2d: np.ndarray,
    coord_nod2d: np.ndarray,
    config: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interpolate FESOM2 data to regular grid using inverse distance weighting.

    Interpolation is done depth by depth (2D for each depth level).

    Args:
        temp: Temperature on FESOM2 mesh, shape (nz, nnodes)
        salt: Salinity on FESOM2 mesh, shape (nz, nnodes)
        nodes_lon: FESOM2 node longitudes
        nodes_lat: FESOM2 node latitudes
        depth: Depth levels
        lonreg2d: Regular grid longitudes, 2D
        latreg2d: Regular grid latitudes, 2D
        coord_nod2d: FESOM-C mesh coordinates for coverage check
        config: Configuration dictionary

    Returns:
        Tuple containing:
            - temp_reg: Temperature on regular grid, shape (nz, nlat, nlon)
            - salt_reg: Salinity on regular grid, shape (nz, nlat, nlon)
    """
    print("Interpolating to regular grid...")

    nz = temp.shape[0]
    nlat, nlon = lonreg2d.shape

    # Initialize output arrays
    temp_reg = np.zeros((nz, nlat, nlon))
    salt_reg = np.zeros((nz, nlat, nlon))

    # Get interpolation parameters
    k = config["interpolation"]["k_neighbors"]
    use_idw = config["interpolation"]["use_idw"]
    idw_power = config["interpolation"]["idw_power"]

    # Set up stereographic projection centered on the region
    lat_mean = nodes_lat.mean()
    lon_mean = nodes_lon.mean()
    pste = pyproj.Proj(proj="stere", ellps='WGS84', lat_0=lat_mean, lon_0=lon_mean)

    # Convert FESOM2 node coordinates to Cartesian
    nodes_x, nodes_y = pste(nodes_lon, nodes_lat)

    # Build KDTree from FESOM2 nodes
    print(f"  Building KDTree from {len(nodes_lon)} FESOM2 nodes...")
    tree = cKDTree(np.column_stack([nodes_x, nodes_y]))

    # Convert regular grid coordinates to Cartesian
    lonreg_flat = lonreg2d.flatten()
    latreg_flat = latreg2d.flatten()
    xreg_flat, yreg_flat = pste(lonreg_flat, latreg_flat)
    query_points = np.column_stack([xreg_flat, yreg_flat])

    # Query KDTree for k nearest neighbors
    print(f"  Querying KDTree for {k} nearest neighbors...")
    distances, indices = tree.query(query_points, k=k, workers=-1)

    # Interpolate depth by depth
    for iz in range(nz):
        if config["diagnostics"]["verbose"] and iz % 5 == 0:
            print(f"  Processing depth level {iz+1}/{nz} ({depth[iz]:.1f} m)...")

        # Get temperature and salinity at this depth
        temp_depth = temp[iz, :]
        salt_depth = salt[iz, :]

        if use_idw and k > 1:
            # Inverse distance weighting
            # Handle zero distances (query point coincides with data point)
            distances_safe = np.where(distances == 0, 1e-10, distances)
            weights = 1.0 / (distances_safe ** idw_power)

            # Calculate weighted average
            temp_interp = np.sum(weights * temp_depth[indices], axis=1) / np.sum(weights, axis=1)
            salt_interp = np.sum(weights * salt_depth[indices], axis=1) / np.sum(weights, axis=1)
        else:
            # Nearest neighbor
            temp_interp = temp_depth[indices[:, 0]]
            salt_interp = salt_depth[indices[:, 0]]

        # Reshape to 2D grid
        temp_reg[iz, :, :] = temp_interp.reshape(nlat, nlon)
        salt_reg[iz, :, :] = salt_interp.reshape(nlat, nlon)

    print("  Interpolation complete.")

    # Check coverage of FESOM-C mesh
    print("  Checking coverage of FESOM-C mesh nodes...")
    fesomc_x, fesomc_y = pste(coord_nod2d[:, 0], coord_nod2d[:, 1])
    fesomc_points = np.column_stack([fesomc_x, fesomc_y])

    # Find nearest regular grid point for each FESOM-C node
    tree_reg = cKDTree(query_points)
    dist_to_reg, _ = tree_reg.query(fesomc_points, k=1, workers=-1)

    # Check if all FESOM-C nodes are within a reasonable distance
    grid_spacing = np.mean([
        np.abs(lonreg2d[0, 1] - lonreg2d[0, 0]),
        np.abs(latreg2d[1, 0] - latreg2d[0, 0])
    ])
    max_dist_deg = 2 * grid_spacing  # Allow 2 grid cells

    # Convert to meters for comparison
    max_dist_m = max_dist_deg * 111000  # Rough conversion: 1 degree ≈ 111 km

    uncovered = np.sum(dist_to_reg > max_dist_m)
    if uncovered > 0:
        print(f"  WARNING: {uncovered} FESOM-C nodes may not be fully covered by regular grid")
        print(f"  Consider increasing margin_degrees in config")
    else:
        print(f"  All FESOM-C nodes are covered by regular grid.")

    # Extrapolate to fill any remaining NaNs
    print("  Extrapolating to fill NaNs...")
    for iz in range(nz):
        temp_reg[iz, :, :] = fill_nans_with_neighbors(temp_reg[iz, :, :])
        salt_reg[iz, :, :] = fill_nans_with_neighbors(salt_reg[iz, :, :])

    return temp_reg, salt_reg


def fill_nans_with_neighbors(data: np.ndarray, max_iter: int = 100) -> np.ndarray:
    """
    Fill NaN values in 2D array using nearest neighbor extrapolation.

    Args:
        data: 2D array with potential NaN values
        max_iter: Maximum number of iterations

    Returns:
        Array with NaNs filled
    """
    data = data.copy()

    for iteration in range(max_iter):
        # Check if there are any NaNs left
        mask = np.isnan(data)
        if not np.any(mask):
            break

        # Get coordinates of non-NaN and NaN values
        coords_valid = np.argwhere(~mask)
        coords_nan = np.argwhere(mask)

        if len(coords_valid) == 0:
            # No valid data to extrapolate from
            break

        # Use griddata with nearest neighbor
        filled_values = griddata(coords_valid, data[~mask], coords_nan, method='nearest')
        data[mask] = filled_values

        # Break after one iteration since nearest neighbor should fill all at once
        break

    return data


def apply_smoothing(temp_reg: np.ndarray, salt_reg: np.ndarray, config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Savitzky-Golay smoothing filter to 3D data.

    Args:
        temp_reg: Temperature on regular grid, shape (nz, nlat, nlon)
        salt_reg: Salinity on regular grid, shape (nz, nlat, nlon)
        config: Configuration dictionary

    Returns:
        Tuple containing:
            - temp_smooth: Smoothed temperature
            - salt_smooth: Smoothed salinity
    """
    if not config["smoothing"]["enabled"]:
        print("Smoothing disabled, skipping...")
        return temp_reg, salt_reg

    print("Applying Savitzky-Golay smoothing...")

    wl = config["smoothing"]["window_length"]
    po = config["smoothing"]["polyorder"]

    # Apply smoothing along each axis
    temp_smooth = temp_reg.copy()
    salt_smooth = salt_reg.copy()

    # Smooth along depth axis
    temp_smooth = savgol_filter(temp_smooth, window_length=wl, polyorder=po, mode='nearest', axis=0)
    salt_smooth = savgol_filter(salt_smooth, window_length=wl, polyorder=po, mode='nearest', axis=0)

    # Smooth along latitude axis
    temp_smooth = savgol_filter(temp_smooth, window_length=wl, polyorder=po, mode='nearest', axis=1)
    salt_smooth = savgol_filter(salt_smooth, window_length=wl, polyorder=po, mode='nearest', axis=1)

    # Smooth along longitude axis
    temp_smooth = savgol_filter(temp_smooth, window_length=wl, polyorder=po, mode='nearest', axis=2)
    salt_smooth = savgol_filter(salt_smooth, window_length=wl, polyorder=po, mode='nearest', axis=2)

    print("  Smoothing complete.")

    return temp_smooth, salt_smooth


def check_density_stability(
    temp_reg: np.ndarray,
    salt_reg: np.ndarray,
    depth: np.ndarray,
    lonreg2d: np.ndarray,
    latreg2d: np.ndarray,
    config: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Check and fix density instabilities (lighter water below denser water).

    Uses GSW (Gibbs SeaWater) package to calculate density.

    Args:
        temp_reg: Temperature on regular grid, shape (nz, nlat, nlon)
        salt_reg: Salinity on regular grid, shape (nz, nlat, nlon)
        depth: Depth levels, shape (nz,)
        lonreg2d: Regular grid longitudes, 2D
        latreg2d: Regular grid latitudes, 2D
        config: Configuration dictionary

    Returns:
        Tuple containing:
            - temp_stable: Temperature with stable stratification
            - salt_stable: Salinity with stable stratification
    """
    if not config["stability"]["check_density"]:
        print("Density stability check disabled, skipping...")
        return temp_reg, salt_reg

    print("Checking density stability...")

    temp_stable = temp_reg.copy()
    salt_stable = salt_reg.copy()

    nz, nlat, nlon = temp_reg.shape

    # Prepare arrays for GSW functions
    # GSW functions need: (pressure/depth, lon, lat)
    # We'll process column by column to handle 1D depth profiles

    max_iter = config["stability"]["max_iterations"]
    total_swaps = 0

    for j in range(nlat):
        for i in range(nlon):
            # Get T and S profile at this location
            temp_profile = temp_stable[:, j, i]
            salt_profile = salt_stable[:, j, i]
            lon_point = lonreg2d[j, i]
            lat_point = latreg2d[j, i]

            # Iterate until stable
            for iteration in range(max_iter):
                # Calculate absolute salinity
                SA = gsw.SA_from_SP(salt_profile, depth, lon_point, lat_point)

                # Calculate conservative temperature
                CT = gsw.CT_from_pt(SA, temp_profile)

                # Calculate density
                rho = gsw.rho(SA, CT, depth)

                # Check for instabilities (density should increase with depth)
                unstable = False
                for k in range(nz - 1):
                    if rho[k] > rho[k + 1]:
                        # Found instability: swap values
                        temp_profile[k + 1] = temp_profile[k]
                        salt_profile[k + 1] = salt_profile[k]
                        unstable = True
                        total_swaps += 1

                # If no instabilities found, we're done with this column
                if not unstable:
                    break

            # Store stabilized profile
            temp_stable[:, j, i] = temp_profile
            salt_stable[:, j, i] = salt_profile

    if total_swaps > 0:
        print(f"  Fixed {total_swaps} density instabilities")
    else:
        print("  No density instabilities found")

    return temp_stable, salt_stable


def save_to_netcdf(
    temp_reg: np.ndarray,
    salt_reg: np.ndarray,
    depth: np.ndarray,
    lonreg: np.ndarray,
    latreg: np.ndarray,
    config: Dict[str, Any]
) -> None:
    """
    Save temperature and salinity to NetCDF file for FESOM-C.

    Args:
        temp_reg: Temperature on regular grid, shape (nz, nlat, nlon)
        salt_reg: Salinity on regular grid, shape (nz, nlat, nlon)
        depth: Depth levels, shape (nz,)
        lonreg: 1D longitude array
        latreg: 1D latitude array
        config: Configuration dictionary
    """
    print("Saving to NetCDF file...")

    output_file = config["output"]["output_file"]

    # Add boundary depth levels if requested
    if config["output"]["add_boundary_depths"]:
        shallow_depth = config["output"]["shallow_depth"]
        deep_depth = config["output"]["deep_depth"]

        # Create new depth array
        nz = len(depth)
        depth_out = np.zeros(nz + 2)
        depth_out[0] = shallow_depth
        depth_out[1:-1] = depth
        depth_out[-1] = deep_depth

        # Create new T/S arrays
        nz, nlat, nlon = temp_reg.shape
        temp_out = np.zeros((nz + 2, nlat, nlon))
        salt_out = np.zeros((nz + 2, nlat, nlon))

        # Copy data
        temp_out[1:-1, :, :] = temp_reg
        salt_out[1:-1, :, :] = salt_reg

        # Repeat surface values at shallow depth
        temp_out[0, :, :] = temp_reg[0, :, :]
        salt_out[0, :, :] = salt_reg[0, :, :]

        # Repeat bottom values at deep depth
        temp_out[-1, :, :] = temp_reg[-1, :, :]
        salt_out[-1, :, :] = salt_reg[-1, :, :]
    else:
        depth_out = depth
        temp_out = temp_reg
        salt_out = salt_reg

    # Create NetCDF file
    ncfile = Dataset(output_file, 'w', format='NETCDF4')

    # Create dimensions
    ncfile.createDimension('depth', len(depth_out))
    ncfile.createDimension('lat', len(latreg))
    ncfile.createDimension('lon', len(lonreg))

    # Create coordinate variables
    var_depth = ncfile.createVariable('depth', 'f8', ('depth',))
    var_lat = ncfile.createVariable('lat', 'f8', ('lat',))
    var_lon = ncfile.createVariable('lon', 'f8', ('lon',))

    # Create data variables
    var_temp = ncfile.createVariable('temp', 'f8', ('depth', 'lat', 'lon',))
    var_salt = ncfile.createVariable('salt', 'f8', ('depth', 'lat', 'lon',))

    # Set coordinate attributes
    var_lon.axis = 'X'
    var_lon.units = 'degrees_east'
    var_lon.standard_name = 'longitude'
    var_lon.long_name = 'longitude'

    var_lat.axis = 'Y'
    var_lat.units = 'degrees_north'
    var_lat.standard_name = 'latitude'
    var_lat.long_name = 'latitude'

    var_depth.axis = 'Z'
    var_depth.units = 'm'
    var_depth.standard_name = 'depth'
    var_depth.long_name = 'depth'
    var_depth.positive = 'down'

    # Set data variable attributes
    var_temp.units = 'degree_Celsius'
    var_temp.standard_name = 'sea_water_potential_temperature'
    var_temp.long_name = 'Potential Temperature'

    var_salt.units = 'PSU'
    var_salt.standard_name = 'sea_water_salinity'
    var_salt.long_name = 'Salinity'

    # Write data
    var_lon[:] = lonreg
    var_lat[:] = latreg
    var_depth[:] = depth_out
    var_temp[:, :, :] = temp_out
    var_salt[:, :, :] = salt_out

    # Add global attributes
    ncfile.title = 'Initial Conditions for FESOM-C'
    ncfile.source = 'Interpolated from FESOM2 output'
    ncfile.history = f'Created with prepare_ic_fesom.py'

    # Close file
    ncfile.close()

    print(f"  Saved to {output_file}")
    print(f"  Dimensions: depth={len(depth_out)}, lat={len(latreg)}, lon={len(lonreg)}")


def print_statistics(temp_reg: np.ndarray, salt_reg: np.ndarray, depth: np.ndarray, config: Dict[str, Any]) -> None:
    """
    Print statistics of the interpolated data.

    Args:
        temp_reg: Temperature on regular grid
        salt_reg: Salinity on regular grid
        depth: Depth levels
        config: Configuration dictionary
    """
    if not config["diagnostics"]["print_statistics"]:
        return

    print("\n" + "="*60)
    print("DATA STATISTICS")
    print("="*60)

    print("\nOverall statistics:")
    print(f"  Temperature: min={np.min(temp_reg):.4f}, max={np.max(temp_reg):.4f}, mean={np.mean(temp_reg):.4f} °C")
    print(f"  Salinity:    min={np.min(salt_reg):.4f}, max={np.max(salt_reg):.4f}, mean={np.mean(salt_reg):.4f} PSU")

    print("\nStatistics by depth level:")
    print(f"{'Depth (m)':<12} {'T_min':<10} {'T_max':<10} {'T_mean':<10} {'S_min':<10} {'S_max':<10} {'S_mean':<10}")
    print("-" * 72)

    nz = temp_reg.shape[0]
    # Print first 10, then every 5th, then last 5
    indices_to_print = list(range(min(10, nz)))
    if nz > 10:
        indices_to_print += list(range(10, nz - 5, 5))
        indices_to_print += list(range(max(10, nz - 5), nz))

    for iz in indices_to_print:
        d = depth[iz]
        t_min = np.min(temp_reg[iz, :, :])
        t_max = np.max(temp_reg[iz, :, :])
        t_mean = np.mean(temp_reg[iz, :, :])
        s_min = np.min(salt_reg[iz, :, :])
        s_max = np.max(salt_reg[iz, :, :])
        s_mean = np.mean(salt_reg[iz, :, :])

        print(f"{d:<12.1f} {t_min:<10.4f} {t_max:<10.4f} {t_mean:<10.4f} {s_min:<10.4f} {s_max:<10.4f} {s_mean:<10.4f}")

    print("="*60 + "\n")


def plot_surface_data(
    temp_reg: np.ndarray,
    salt_reg: np.ndarray,
    lonreg2d: np.ndarray,
    latreg2d: np.ndarray,
    coord_nod2d: np.ndarray,
    config: Dict[str, Any],
    profile_points: np.ndarray = None
) -> None:
    """
    Create plots of surface temperature and salinity with FESOM-C mesh nodes overlay.

    Args:
        temp_reg: Temperature on regular grid
        salt_reg: Salinity on regular grid
        lonreg2d: 2D longitude grid
        latreg2d: 2D latitude grid
        coord_nod2d: FESOM-C mesh node coordinates (lon, lat)
        config: Configuration dictionary
        profile_points: Optional array of profile point indices to mark in red
    """
    if not config["diagnostics"]["create_plots"]:
        return

    print("Creating surface plots...")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  WARNING: matplotlib not available, skipping plots")
        return

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot surface temperature
    temp_surface = temp_reg[0, :, :]
    cs1 = ax1.contourf(lonreg2d, latreg2d, temp_surface, levels=20, cmap='RdYlBu_r')
    ax1.set_xlabel('Longitude [°E]')
    ax1.set_ylabel('Latitude [°N]')
    ax1.set_title('Surface Temperature [°C]')
    fig.colorbar(cs1, ax=ax1)

    # Overlay FESOM-C mesh nodes (black dots with transparency)
    ax1.scatter(coord_nod2d[:, 0], coord_nod2d[:, 1],
                c='black', s=0.5, alpha=0.5, label='FESOM-C nodes')

    # Plot surface salinity
    salt_surface = salt_reg[0, :, :]
    cs2 = ax2.contourf(lonreg2d, latreg2d, salt_surface, levels=20, cmap='viridis')
    ax2.set_xlabel('Longitude [°E]')
    ax2.set_ylabel('Latitude [°N]')
    ax2.set_title('Surface Salinity [PSU]')
    fig.colorbar(cs2, ax=ax2)

    # Overlay FESOM-C mesh nodes (black dots with transparency)
    ax2.scatter(coord_nod2d[:, 0], coord_nod2d[:, 1],
                c='black', s=0.5, alpha=0.5, label='FESOM-C nodes')

    # Mark profile points in red if provided
    if profile_points is not None and len(profile_points) > 0:
        profile_lons = coord_nod2d[profile_points, 0]
        profile_lats = coord_nod2d[profile_points, 1]

        ax1.scatter(profile_lons, profile_lats,
                   c='red', s=50, marker='o', edgecolors='white', linewidths=1,
                   label='Profile points', zorder=10)
        ax2.scatter(profile_lons, profile_lats,
                   c='red', s=50, marker='o', edgecolors='white', linewidths=1,
                   label='Profile points', zorder=10)

        # Add point numbers as labels
        for idx, (lon, lat) in enumerate(zip(profile_lons, profile_lats)):
            ax1.text(lon, lat, f' {idx+1}', fontsize=10, fontweight='bold',
                    color='red', va='center', ha='left')
            ax2.text(lon, lat, f' {idx+1}', fontsize=10, fontweight='bold',
                    color='red', va='center', ha='left')

    plt.tight_layout()

    # Save figure
    plot_file = config["diagnostics"]["plot_file"]
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"  Saved plot to {plot_file}")

    plt.close()


def read_elem_file(elem_file: str) -> np.ndarray:
    """
    Read FESOM-C element connectivity from ASCII file.

    Args:
        elem_file: Path to elements file (elem2d.out)

    Returns:
        elem2d: Element connectivity array, shape (nelem, 4), 1-indexed
    """
    if not os.path.exists(elem_file):
        raise FileNotFoundError(f"Elements file not found: {elem_file}")

    with open(elem_file, 'r') as f:
        # Read total number of elements
        nelem = int(f.readline().strip())

        # Initialize array
        elem2d = np.empty((nelem, 4), dtype=int)

        # Read element data
        for i in range(nelem):
            line = f.readline().strip()
            parts = line.split()
            # Store connectivity (already 1-indexed in file, keep as is)
            elem2d[i, :] = [int(p) for p in parts[:4]]

    return elem2d


def plot_profile_data(
    temp_reg: np.ndarray,
    salt_reg: np.ndarray,
    depth: np.ndarray,
    lonreg: np.ndarray,
    latreg: np.ndarray,
    coord_nod2d: np.ndarray,
    profile_points: np.ndarray,
    node_depth: np.ndarray,
    config: Dict[str, Any]
) -> None:
    """
    Plot temperature and salinity profiles at selected points.

    Creates a 3-row, 1-column figure with T/S profiles at each point.
    Each subplot has dual x-axes (top for T, bottom for S).
    Y-axis limits set based on actual bathymetry depth at each point.

    Args:
        temp_reg: Temperature on regular grid, shape (nz, nlat, nlon)
        salt_reg: Salinity on regular grid, shape (nz, nlat, nlon)
        depth: Depth levels
        lonreg: 1D longitude array
        latreg: 1D latitude array
        coord_nod2d: FESOM-C mesh node coordinates
        profile_points: Indices of points to plot profiles
        node_depth: Bathymetry depth at each node
        config: Configuration dictionary
    """
    if not config["diagnostics"]["create_profile_plots"]:
        return

    print("Creating T/S profile plots...")

    try:
        import matplotlib.pyplot as plt
        from scipy.interpolate import RegularGridInterpolator
    except ImportError:
        print("  WARNING: matplotlib or scipy not available, skipping profile plots")
        return

    # Create interpolators for T and S at each depth
    n_points = len(profile_points)

    fig, axes = plt.subplots(n_points, 1, figsize=(8, 3 * n_points))
    if n_points == 1:
        axes = [axes]  # Make it iterable

    for idx, point_idx in enumerate(profile_points):
        lon_point = coord_nod2d[point_idx, 0]
        lat_point = coord_nod2d[point_idx, 1]

        # Extract T/S profiles by interpolating from regular grid
        temp_profile = np.zeros(len(depth))
        salt_profile = np.zeros(len(depth))

        for iz in range(len(depth)):
            # Create interpolator for this depth level
            temp_interp = RegularGridInterpolator(
                (latreg, lonreg), temp_reg[iz, :, :],
                bounds_error=False, fill_value=None
            )
            salt_interp = RegularGridInterpolator(
                (latreg, lonreg), salt_reg[iz, :, :],
                bounds_error=False, fill_value=None
            )

            # Interpolate at point location
            temp_profile[iz] = temp_interp([lat_point, lon_point])[0]
            salt_profile[iz] = salt_interp([lat_point, lon_point])[0]

        # Plot on this subplot
        ax = axes[idx]

        # Create twin axis for salinity
        ax2 = ax.twiny()

        # Plot temperature (bottom x-axis, blue)
        ax.plot(temp_profile, depth, 'b-', linewidth=2, label='Temperature')
        ax.scatter(temp_profile, depth, c='blue', s=20, zorder=5)

        # Plot salinity (top x-axis, red)
        ax2.plot(salt_profile, depth, 'r-', linewidth=2, label='Salinity')
        ax2.scatter(salt_profile, depth, c='red', s=20, zorder=5)

        # Invert y-axis (depth increases downward)
        ax.invert_yaxis()

        # Set Y-axis limits based on bathymetry depth at this node (if available)
        if node_depth is not None:
            max_depth = node_depth[point_idx] + 10  # Add 10m margin
            ax.set_ylim(max_depth, 0)  # Inverted: bottom to top

        # Set labels
        ax.set_xlabel('Temperature [°C]', color='blue', fontsize=11)
        ax2.set_xlabel('Salinity [PSU]', color='red', fontsize=11)
        ax.set_ylabel('Depth [m]', fontsize=11)

        # Color the tick labels
        ax.tick_params(axis='x', labelcolor='blue')
        ax2.tick_params(axis='x', labelcolor='red')

        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--')

        # Title with point location (include depth if available)
        if node_depth is not None:
            ax.set_title(f'Point {idx+1}: Lon={lon_point:.2f}°E, Lat={lat_point:.2f}°N (Depth: {node_depth[point_idx]:.0f}m)',
                        fontweight='bold', fontsize=12)
        else:
            ax.set_title(f'Point {idx+1}: Lon={lon_point:.2f}°E, Lat={lat_point:.2f}°N',
                        fontweight='bold', fontsize=12)

    plt.tight_layout()

    # Save figure
    plot_file = config["diagnostics"]["profile_plot_file"]
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"  Saved profile plot to {plot_file}")

    plt.close()


def plot_native_mesh_data(
    temp_reg: np.ndarray,
    salt_reg: np.ndarray,
    lonreg: np.ndarray,
    latreg: np.ndarray,
    coord_nod2d: np.ndarray,
    elem2d: np.ndarray,
    config: Dict[str, Any]
) -> None:
    """
    Plot temperature and salinity on native FESOM-C mesh using PolyCollection.

    Args:
        temp_reg: Temperature on regular grid
        salt_reg: Salinity on regular grid
        lonreg: 1D longitude array
        latreg: 1D latitude array
        coord_nod2d: FESOM-C mesh node coordinates (lon, lat)
        elem2d: Element connectivity (1-indexed)
        config: Configuration dictionary
    """
    if not config["diagnostics"]["create_native_mesh_plot"]:
        return

    print("Creating native mesh plots...")

    try:
        import matplotlib.pyplot as plt
        from matplotlib.collections import PolyCollection
        from scipy.interpolate import RegularGridInterpolator
    except ImportError:
        print("  WARNING: matplotlib or scipy not available, skipping native mesh plots")
        return

    # Interpolate regular grid data to FESOM-C mesh nodes
    print("  Interpolating data to mesh nodes...")
    temp_surface = temp_reg[0, :, :]
    salt_surface = salt_reg[0, :, :]

    temp_interp = RegularGridInterpolator(
        (latreg, lonreg), temp_surface,
        bounds_error=False, fill_value=np.nan
    )
    salt_interp = RegularGridInterpolator(
        (latreg, lonreg), salt_surface,
        bounds_error=False, fill_value=np.nan
    )

    # Get temperature and salinity at each node
    temp_nodes = temp_interp(coord_nod2d[:, [1, 0]])  # Note: lat, lon order
    salt_nodes = salt_interp(coord_nod2d[:, [1, 0]])

    # Create polygons from elements
    print("  Building mesh polygons...")
    polygons_temp = []
    values_temp = []
    polygons_salt = []
    values_salt = []

    for elem in elem2d:
        # Get node indices (convert from 1-indexed to 0-indexed)
        nodes = elem - 1

        # Check if it's a triangle (4th node is same as 3rd) or quadrilateral
        if nodes[2] == nodes[3]:
            # Triangle
            node_coords = coord_nod2d[nodes[:3], :]
            elem_temp = np.mean(temp_nodes[nodes[:3]])
            elem_salt = np.mean(salt_nodes[nodes[:3]])
        else:
            # Quadrilateral
            node_coords = coord_nod2d[nodes, :]
            elem_temp = np.mean(temp_nodes[nodes])
            elem_salt = np.mean(salt_nodes[nodes])

        # Only add if all values are valid
        if not np.isnan(elem_temp):
            polygons_temp.append(node_coords)
            values_temp.append(elem_temp)
        if not np.isnan(elem_salt):
            polygons_salt.append(node_coords)
            values_salt.append(elem_salt)

    print(f"  Created {len(polygons_temp)} polygons")

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Temperature plot
    temp_collection = PolyCollection(
        polygons_temp,
        array=np.array(values_temp),
        cmap='RdYlBu_r',
        edgecolors='none',
        linewidths=0
    )
    ax1.add_collection(temp_collection)
    ax1.autoscale_view()
    ax1.set_xlabel('Longitude [°E]')
    ax1.set_ylabel('Latitude [°N]')
    ax1.set_title('Surface Temperature on FESOM-C Mesh [°C]')
    ax1.set_aspect('equal', adjustable='box')
    fig.colorbar(temp_collection, ax=ax1)

    # Salinity plot
    salt_collection = PolyCollection(
        polygons_salt,
        array=np.array(values_salt),
        cmap='viridis',
        edgecolors='none',
        linewidths=0
    )
    ax2.add_collection(salt_collection)
    ax2.autoscale_view()
    ax2.set_xlabel('Longitude [°E]')
    ax2.set_ylabel('Latitude [°N]')
    ax2.set_title('Surface Salinity on FESOM-C Mesh [PSU]')
    ax2.set_aspect('equal', adjustable='box')
    fig.colorbar(salt_collection, ax=ax2)

    plt.tight_layout()

    # Save figure
    plot_file = config["diagnostics"]["native_mesh_plot_file"]
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"  Saved native mesh plot to {plot_file}")

    plt.close()


def main():
    """
    Main execution function.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Prepare Initial Conditions for FESOM-C from FESOM2 output',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file (default: config.yaml)')
    parser.add_argument('--k-neighbors', type=int, default=None,
                        help='Override number of nearest neighbors for interpolation')
    parser.add_argument('--smoothing', action='store_true',
                        help='Enable smoothing (overrides config)')
    parser.add_argument('--no-smoothing', action='store_true',
                        help='Disable smoothing (overrides config)')
    parser.add_argument('--plot', action='store_true',
                        help='Create plots (overrides config)')
    parser.add_argument('--no-plot', action='store_true',
                        help='Disable plots (overrides config)')

    args = parser.parse_args()

    # Load configuration
    print("="*60)
    print("FESOM-C Initial Conditions Preparation Tool")
    print("="*60)
    print(f"\nLoading configuration from {args.config}...")
    config = load_config(args.config)

    # Apply command-line overrides
    if args.k_neighbors is not None:
        config["interpolation"]["k_neighbors"] = args.k_neighbors
    if args.smoothing:
        config["smoothing"]["enabled"] = True
    if args.no_smoothing:
        config["smoothing"]["enabled"] = False
    if args.plot:
        config["diagnostics"]["create_plots"] = True
    if args.no_plot:
        config["diagnostics"]["create_plots"] = False

    # Step 1: Read FESOM-C mesh
    nodes_file = os.path.join(
        config["input"]["fesomc_mesh_dir"],
        config["input"]["fesomc_nodes_file"]
    )
    nod2d, coord_nod2d, nodes_obindex = read_fesomc_mesh(nodes_file, config)

    # Step 1b: Read bathymetry depths (optional, for profile plots)
    node_depth = None
    if config["diagnostics"].get("create_profile_plots", False):
        depth_file = os.path.join(
            config["input"]["fesomc_mesh_dir"],
            "depth.out"
        )
        try:
            node_depth = read_depth_file(depth_file, nod2d)
        except FileNotFoundError as e:
            print(f"  WARNING: {e}")
            print("  Profile plots will use full depth range.")

    # Step 2: Create regular grid
    lonreg, latreg, lonreg2d, latreg2d = create_regular_grid(coord_nod2d, config)

    # Step 3: Read FESOM2 data (only nodes in region of interest)
    temp, salt, nodes_lon, nodes_lat, depth = read_fesom2_data(config, coord_nod2d)

    # Step 4: Interpolate to regular grid
    temp_reg, salt_reg = interpolate_to_regular_grid(
        temp, salt, nodes_lon, nodes_lat, depth,
        lonreg2d, latreg2d, coord_nod2d, config
    )

    # Step 5: Optional smoothing
    temp_reg, salt_reg = apply_smoothing(temp_reg, salt_reg, config)

    # Step 6: Check density stability
    temp_reg, salt_reg = check_density_stability(
        temp_reg, salt_reg, depth, lonreg2d, latreg2d, config
    )

    # Step 7: Save to NetCDF
    save_to_netcdf(temp_reg, salt_reg, depth, lonreg, latreg, config)

    # Step 8: Print statistics
    print_statistics(temp_reg, salt_reg, depth, config)

    # Step 9: Optional advanced plotting - select random profile points
    profile_points = None
    if config["diagnostics"].get("create_profile_plots", False):
        num_points = config["diagnostics"].get("num_profile_points", 3)
        print(f"\nSelecting {num_points} random points for profile plots...")
        # Select random points from FESOM-C mesh (excluding boundary nodes if possible)
        internal_nodes = np.where(nodes_obindex == 0)[0]
        if len(internal_nodes) >= num_points:
            profile_points = np.random.choice(internal_nodes, size=num_points, replace=False)
        else:
            profile_points = np.random.choice(nod2d, size=num_points, replace=False)
        print(f"  Selected points: {profile_points}")
        for idx, pt in enumerate(profile_points):
            if node_depth is not None:
                print(f"    Point {idx+1}: lon={coord_nod2d[pt, 0]:.2f}°E, lat={coord_nod2d[pt, 1]:.2f}°N, depth={node_depth[pt]:.0f}m")
            else:
                print(f"    Point {idx+1}: lon={coord_nod2d[pt, 0]:.2f}°E, lat={coord_nod2d[pt, 1]:.2f}°N")

    # Step 10: Surface plots (with optional profile point markers)
    plot_surface_data(temp_reg, salt_reg, lonreg2d, latreg2d, coord_nod2d, config, profile_points)

    # Step 11: Profile plots
    if profile_points is not None:
        plot_profile_data(temp_reg, salt_reg, depth, lonreg, latreg, coord_nod2d, profile_points, node_depth, config)

    # Step 12: Native mesh plot (requires elements file)
    if config["diagnostics"].get("create_native_mesh_plot", False):
        elem_file = os.path.join(
            config["input"]["fesomc_mesh_dir"],
            config["input"]["fesomc_elements_file"]
        )
        try:
            elem2d = read_elem_file(elem_file)
            plot_native_mesh_data(temp_reg, salt_reg, lonreg, latreg, coord_nod2d, elem2d, config)
        except FileNotFoundError as e:
            print(f"  WARNING: {e}")
            print("  Skipping native mesh plot.")

    print("\n" + "="*60)
    print("PROCESSING COMPLETE!")
    print("="*60)
    print(f"\nOutput file: {config['output']['output_file']}")
    print("Ready to use with FESOM-C model.")


if __name__ == '__main__':
    main()
