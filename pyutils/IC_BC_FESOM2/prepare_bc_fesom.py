#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FESOM-C Boundary Conditions Preparation from FESOM2 Output

This script prepares boundary condition (BC) time series for FESOM-C ocean model
from FESOM2 output data. It extracts data at open boundary nodes over multiple
time steps (years/months) and creates CF-compliant NetCDF output.

Key differences from IC preparation:
- Extracts only boundary nodes (index==2) instead of full mesh
- Handles time series (multiple years/months) instead of single snapshot
- Direct interpolation to nodes (no regular grid intermediate step)
- Output dimensions: (time, depth, node) instead of (depth, lat, lon)
- Creates Hovmuller diagrams and spatial BC plots

Author: Generated with Claude Code
Date: 2026-01-12
"""

import numpy as np
import xarray as xr
from scipy.spatial import cKDTree
from scipy.interpolate import RegularGridInterpolator
import pyproj
import gsw
import yaml
import argparse
from typing import Dict, Tuple, List, Any
from pathlib import Path
import sys
import datetime

# Optional matplotlib import
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.collections import PolyCollection
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available, plotting will be disabled")


def load_config(config_file: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_file: Path to YAML configuration file

    Returns:
        Dictionary with configuration parameters
    """
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config


def read_fesomc_boundary_nodes(nodes_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read FESOM-C mesh and extract only boundary nodes (index==2).

    Args:
        nodes_file: Path to nod2d.out file

    Returns:
        ob_indices: Original node indices (0-indexed for Python)
        ob_coords: Coordinates of boundary nodes (shape: n_boundary, 2) [lon, lat]
        ob_indices_1based: 1-indexed node numbers for NetCDF output
    """
    print("\nStep 1: Reading FESOM-C mesh and extracting boundary nodes...")

    with open(nodes_file, 'r') as f:
        nod2d = int(f.readline().strip().split()[0])

        coord_nod2d = np.zeros((nod2d, 2))
        nodes_obindex = np.zeros(nod2d, dtype=int)

        for i, line in enumerate(f):
            columns = line.strip().split()
            coord_nod2d[i, 0] = float(columns[1])  # longitude
            coord_nod2d[i, 1] = float(columns[2])  # latitude
            nodes_obindex[i] = int(columns[3])      # boundary flag

    # Extract boundary nodes (index == 2)
    ob_mask = nodes_obindex == 2
    ob_indices = np.where(ob_mask)[0]
    ob_coords = coord_nod2d[ob_mask]
    ob_indices_1based = ob_indices + 1  # 1-indexed for output

    print(f"  Total nodes: {nod2d:,}")
    print(f"  Boundary nodes (index==2): {len(ob_indices):,}")
    print(f"  Boundary longitude range: [{ob_coords[:, 0].min():.2f}, {ob_coords[:, 0].max():.2f}]°E")
    print(f"  Boundary latitude range: [{ob_coords[:, 1].min():.2f}, {ob_coords[:, 1].max():.2f}]°N")

    if len(ob_indices) == 0:
        raise ValueError("No boundary nodes found (index==2). Check mesh file.")

    return ob_indices, ob_coords, ob_indices_1based


def read_boundary_depths(depth_file: str, ob_indices: np.ndarray) -> np.ndarray:
    """
    Read bathymetry depths for boundary nodes from depth.out file.

    Args:
        depth_file: Path to depth.out file
        ob_indices: Indices of boundary nodes (0-indexed)

    Returns:
        ob_depths: Depths at boundary nodes (meters)
    """
    try:
        with open(depth_file, 'r') as f:
            # Skip first line (total number)
            f.readline()

            # Read all depths
            all_depths = []
            for line in f:
                all_depths.append(float(line.strip()))

            all_depths = np.array(all_depths)

            # Extract depths for boundary nodes
            ob_depths = all_depths[ob_indices]

            print(f"  Boundary depth range: [{ob_depths.min():.1f}, {ob_depths.max():.1f}] m")

            return ob_depths

    except FileNotFoundError:
        print(f"  WARNING: Depth file not found: {depth_file}")
        print("  Hovmuller plots will use full depth range")
        return None
    except Exception as e:
        print(f"  WARNING: Error reading depth file: {e}")
        print("  Hovmuller plots will use full depth range")
        return None


def get_boundary_spatial_bounds(ob_coords: np.ndarray, margin_degrees: float) -> Tuple[float, float, float, float]:
    """
    Determine rectangular bounds around boundary nodes for spatial filtering.

    Args:
        ob_coords: Boundary node coordinates (n_boundary, 2)
        margin_degrees: Extra margin to add (degrees)

    Returns:
        lon_min, lon_max, lat_min, lat_max
    """
    print("\nStep 2: Determining spatial bounds for FESOM2 data filtering...")

    lon_min = ob_coords[:, 0].min() - margin_degrees
    lon_max = ob_coords[:, 0].max() + margin_degrees
    lat_min = ob_coords[:, 1].min() - margin_degrees
    lat_max = ob_coords[:, 1].max() + margin_degrees

    print(f"  Boundary bounds with {margin_degrees:.1f}° margin: "
          f"[{lon_min:.2f}, {lon_max:.2f}]°E × [{lat_min:.2f}, {lat_max:.2f}]°N")

    return lon_min, lon_max, lat_min, lat_max


def read_fesom2_timeseries(
    config: Dict[str, Any],
    spatial_bounds: Tuple[float, float, float, float]
) -> Dict[str, Any]:
    """
    Read FESOM2 data for specified years with spatial filtering.
    Reads ALL time steps from each year's file.

    This is the memory-critical function. It applies spatial filtering to reduce
    the FESOM2 mesh from millions of nodes to only those near the boundary.

    Args:
        config: Configuration dictionary
        spatial_bounds: (lon_min, lon_max, lat_min, lat_max)

    Returns:
        Dictionary containing:
            'temp': (n_times, n_depths, n_filtered_nodes)
            'salt': (n_times, n_depths, n_filtered_nodes)
            'time': List of datetime objects
            'depth': (n_depths,)
            'coords': (n_filtered_nodes, 2) [lon, lat]
            'n_original': Original number of FESOM2 nodes
    """
    print("\nStep 3: Reading FESOM2 time series...")

    input_config = config['input']
    bc_config = config['boundary_conditions']
    data_dir = Path(input_config['fesom2_data_dir'])

    # Use year-based file patterns for BC
    mesh_file = data_dir / input_config['fesom2_mesh_file']
    temp_pattern = input_config['fesom2_temp_pattern']
    salt_pattern = input_config['fesom2_salt_pattern']
    years = bc_config['years']

    lon_min, lon_max, lat_min, lat_max = spatial_bounds

    # Step 3a: Read mesh coordinates and apply spatial filter
    print("  Reading FESOM2 mesh coordinates...")
    mesh_ds = xr.open_dataset(mesh_file)

    nodes_lon = mesh_ds['lon'].values
    nodes_lat = mesh_ds['lat'].values
    n_original = len(nodes_lon)

    # Spatial filtering
    lon_mask = (nodes_lon >= lon_min) & (nodes_lon <= lon_max)
    lat_mask = (nodes_lat >= lat_min) & (nodes_lat <= lat_max)
    spatial_mask = lon_mask & lat_mask

    filtered_indices = np.where(spatial_mask)[0]
    filtered_coords = np.column_stack([nodes_lon[spatial_mask], nodes_lat[spatial_mask]])

    mesh_ds.close()

    print(f"  Original FESOM2 nodes: {n_original:,}")
    print(f"  Nodes after spatial filtering: {len(filtered_indices):,} ({100*len(filtered_indices)/n_original:.1f}%)")
    print(f"  Reduction: {100*(1 - len(filtered_indices)/n_original):.1f}%")

    # Step 3b: Read ALL time steps from each year's file
    print(f"  Processing {len(years)} year(s)")

    temp_list = []
    salt_list = []
    time_list = []
    depth_array = None
    total_time_steps = 0

    for year in years:
        print(f"  Year {year}:")
        temp_file = data_dir / temp_pattern.format(year=year)
        salt_file = data_dir / salt_pattern.format(year=year)

        if not temp_file.exists():
            raise FileNotFoundError(f"Temperature file not found: {temp_file}")
        if not salt_file.exists():
            raise FileNotFoundError(f"Salinity file not found: {salt_file}")

        temp_ds = xr.open_dataset(temp_file)
        salt_ds = xr.open_dataset(salt_file)

        # Get depth levels (same for all years)
        if depth_array is None:
            depth_array = temp_ds['nz1'].values
            print(f"    Depth levels: {len(depth_array)}")

        # Get number of time steps in this file
        n_times_in_file = temp_ds.sizes['time']
        print(f"    Time steps in file: {n_times_in_file}")

        # Read ALL time steps from this year's file
        for time_idx in range(n_times_in_file):
            # Read data for this time step
            temp_full = temp_ds['temp'][time_idx, :, :].values
            salt_full = salt_ds['salt'][time_idx, :, :].values

            # Apply spatial filtering
            temp_filtered = temp_full[:, filtered_indices]
            salt_filtered = salt_full[:, filtered_indices]

            temp_list.append(temp_filtered)
            salt_list.append(salt_filtered)

            # Create datetime - assume monthly data starting from January
            month = (time_idx % 12) + 1  # 1-12
            time_list.append(datetime.datetime(year, month, 15))
            total_time_steps += 1

        temp_ds.close()
        salt_ds.close()

        print(f"    Processed {n_times_in_file} time step(s)")

    print(f"  Total time steps: {total_time_steps}")

    if len(temp_list) == 0:
        raise ValueError("No time steps were processed. Check your years configuration and file existence.")

    # Stack into arrays
    temp_array = np.stack(temp_list, axis=0)  # (n_times, n_depths, n_filtered)
    salt_array = np.stack(salt_list, axis=0)

    print(f"  Final data shape: {temp_array.shape} (time, depth, nodes)")

    return {
        'temp': temp_array,
        'salt': salt_array,
        'time': time_list,
        'depth': depth_array,
        'coords': filtered_coords,
        'n_original': n_original,
        'filtered_indices': filtered_indices
    }


def build_kdtree_for_bc(fesom2_coords: np.ndarray, center_lon: float, center_lat: float) -> Tuple[cKDTree, pyproj.Proj]:
    """
    Build KDTree from FESOM2 coordinates using stereographic projection.

    Args:
        fesom2_coords: Filtered FESOM2 coordinates (n_filtered, 2)
        center_lon: Longitude for projection center
        center_lat: Latitude for projection center

    Returns:
        kdtree: cKDTree for nearest neighbor queries
        proj: Projection object for coordinate transformation
    """
    print("\nStep 4: Building KDTree for interpolation...")

    # Create stereographic projection centered on data
    proj = pyproj.Proj(proj='stere', lat_0=center_lat, lon_0=center_lon, ellps='WGS84')

    # Project to Cartesian coordinates
    x, y = proj(fesom2_coords[:, 0], fesom2_coords[:, 1])
    cart_coords = np.column_stack([x, y])

    # Build KDTree
    kdtree = cKDTree(cart_coords)

    print(f"  KDTree built with {len(fesom2_coords):,} nodes")

    return kdtree, proj


def interpolate_to_boundary_nodes(
    fesom2_data: Dict[str, Any],
    ob_coords: np.ndarray,
    kdtree: cKDTree,
    proj: pyproj.Proj,
    config: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interpolate FESOM2 time series to FESOM-C boundary nodes using IDW.

    Args:
        fesom2_data: Dictionary with FESOM2 data
        ob_coords: Boundary node coordinates (n_boundary, 2)
        kdtree: KDTree for nearest neighbor queries
        proj: Projection for coordinate transformation
        config: Configuration dictionary

    Returns:
        temp_bc: (n_times, n_depths, n_boundary)
        salt_bc: (n_times, n_depths, n_boundary)
    """
    print("\nStep 5: Interpolating to boundary nodes...")

    bc_config = config['boundary_conditions']
    k_neighbors = bc_config['interpolation']['k_neighbors']
    use_idw = bc_config['interpolation']['use_idw']
    idw_power = bc_config['interpolation']['idw_power']

    temp_fesom2 = fesom2_data['temp']
    salt_fesom2 = fesom2_data['salt']
    n_times, n_depths, n_filtered = temp_fesom2.shape
    n_boundary = len(ob_coords)

    print(f"  Time steps: {n_times}")
    print(f"  Depth levels: {n_depths}")
    print(f"  Boundary nodes: {n_boundary}")
    print(f"  Total interpolations: {n_times * n_depths * n_boundary:,}")
    print(f"  Using k={k_neighbors} neighbors, IDW={use_idw}, power={idw_power}")

    # Project boundary coordinates
    ob_x, ob_y = proj(ob_coords[:, 0], ob_coords[:, 1])
    ob_cart = np.column_stack([ob_x, ob_y])

    # Query KDTree for all boundary nodes at once
    distances, indices = kdtree.query(ob_cart, k=k_neighbors)

    # Initialize output arrays
    temp_bc = np.zeros((n_times, n_depths, n_boundary))
    salt_bc = np.zeros((n_times, n_depths, n_boundary))

    # Interpolate for each time step and depth level
    import time as time_module
    start_time = time_module.time()

    for t in range(n_times):
        if t % max(1, n_times // 10) == 0 or t == n_times - 1:
            elapsed = time_module.time() - start_time
            if t > 0:
                eta = elapsed / t * (n_times - t)
                print(f"  Progress: {100*t/n_times:.0f}% ({t}/{n_times}), "
                      f"Elapsed: {elapsed/60:.1f} min, ETA: {eta/60:.1f} min")
            else:
                print(f"  Starting interpolation...")

        for d in range(n_depths):
            temp_slice = temp_fesom2[t, d, :]
            salt_slice = salt_fesom2[t, d, :]

            # Get values at nearest neighbors for all boundary nodes
            temp_neighbors = temp_slice[indices]  # (n_boundary, k_neighbors)
            salt_neighbors = salt_slice[indices]

            if use_idw and k_neighbors > 1:
                # Inverse distance weighting
                # Avoid division by zero
                distances_safe = np.where(distances < 1e-10, 1e-10, distances)
                weights = 1.0 / (distances_safe ** idw_power)
                weights_sum = np.sum(weights, axis=1, keepdims=True)
                weights_norm = weights / weights_sum

                temp_bc[t, d, :] = np.sum(temp_neighbors * weights_norm, axis=1)
                salt_bc[t, d, :] = np.sum(salt_neighbors * weights_norm, axis=1)
            else:
                # Nearest neighbor (k=1) or simple average
                temp_bc[t, d, :] = np.mean(temp_neighbors, axis=1)
                salt_bc[t, d, :] = np.mean(salt_neighbors, axis=1)

    elapsed = time_module.time() - start_time
    print(f"  Interpolation complete: {elapsed/60:.1f} min")

    return temp_bc, salt_bc


def fill_nans_bc(temp_bc: np.ndarray, salt_bc: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fill any remaining NaNs using nearest neighbor extrapolation.

    Args:
        temp_bc: Temperature BC array (n_times, n_depths, n_boundary)
        salt_bc: Salinity BC array

    Returns:
        temp_bc_filled: Temperature with NaNs filled
        salt_bc_filled: Salinity with NaNs filled
    """
    print("\nStep 6: Filling NaNs with nearest neighbor...")

    n_nans_temp = np.isnan(temp_bc).sum()
    n_nans_salt = np.isnan(salt_bc).sum()
    total_points = temp_bc.size

    print(f"  NaNs found in temperature: {n_nans_temp:,} ({100*n_nans_temp/total_points:.2f}%)")
    print(f"  NaNs found in salinity: {n_nans_salt:,} ({100*n_nans_salt/total_points:.2f}%)")

    if n_nans_temp == 0 and n_nans_salt == 0:
        print("  No NaNs to fill")
        return temp_bc, salt_bc

    if n_nans_temp > 0.05 * total_points or n_nans_salt > 0.05 * total_points:
        print("  WARNING: More than 5% NaNs found. Check data quality!")

    # Fill NaNs by propagating from previous depth level
    temp_filled = temp_bc.copy()
    salt_filled = salt_bc.copy()

    for t in range(temp_bc.shape[0]):
        for d in range(temp_bc.shape[1]):
            if d > 0:
                # Fill with previous depth level
                temp_mask = np.isnan(temp_filled[t, d, :])
                salt_mask = np.isnan(salt_filled[t, d, :])
                temp_filled[t, d, temp_mask] = temp_filled[t, d-1, temp_mask]
                salt_filled[t, d, salt_mask] = salt_filled[t, d-1, salt_mask]

    # If still NaNs in first level, fill with nearest valid value
    for t in range(temp_bc.shape[0]):
        if np.isnan(temp_filled[t, 0, :]).any():
            valid_idx = np.where(~np.isnan(temp_filled[t, 0, :]))[0]
            if len(valid_idx) > 0:
                for i in range(temp_filled.shape[2]):
                    if np.isnan(temp_filled[t, 0, i]):
                        nearest = valid_idx[np.argmin(np.abs(valid_idx - i))]
                        temp_filled[t, :, i] = temp_filled[t, :, nearest]
                        salt_filled[t, :, i] = salt_filled[t, :, nearest]

    n_nans_after_temp = np.isnan(temp_filled).sum()
    n_nans_after_salt = np.isnan(salt_filled).sum()

    if n_nans_after_temp > 0 or n_nans_after_salt > 0:
        print(f"  WARNING: {n_nans_after_temp + n_nans_after_salt} NaNs remain after filling")
    else:
        print(f"  All NaNs filled successfully")

    return temp_filled, salt_filled


def check_density_stability_bc(
    temp_bc: np.ndarray,
    salt_bc: np.ndarray,
    depth: np.ndarray,
    ob_coords: np.ndarray,
    max_iterations: int = 100
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Check and fix density stability for all time steps and boundary nodes.

    Args:
        temp_bc: Temperature (n_times, n_depths, n_boundary)
        salt_bc: Salinity
        depth: Depth levels (n_depths,)
        ob_coords: Boundary coordinates for pressure calculation
        max_iterations: Maximum iterations per time step

    Returns:
        temp_stable: Corrected temperature
        salt_stable: Corrected salinity
        total_fixes: Total number of instabilities fixed
    """
    print("\nStep 7: Checking density stability...")

    n_times, n_depths, n_boundary = temp_bc.shape
    temp_stable = temp_bc.copy()
    salt_stable = salt_bc.copy()
    total_fixes = 0

    print(f"  Processing {n_times} time steps...")

    for t in range(n_times):
        if t % max(1, n_times // 10) == 0 or t == n_times - 1:
            print(f"    Time step {t+1}/{n_times} ({100*(t+1)/n_times:.0f}%)")

        iteration = 0
        fixed_this_timestep = False

        while iteration < max_iterations:
            unstable_count = 0

            # Calculate density for all boundary nodes
            for node in range(n_boundary):
                # Get T/S profile
                temp_profile = temp_stable[t, :, node]
                salt_profile = salt_stable[t, :, node]

                # Calculate pressure (dbar) from depth (m) - approximate
                pressure = depth  # Simplified: 1 m ≈ 1 dbar

                # Use average coordinates for all depths
                lon = ob_coords[node, 0]
                lat = ob_coords[node, 1]

                # Calculate absolute salinity and conservative temperature
                SA = gsw.SA_from_SP(salt_profile, pressure, lon, lat)
                CT = gsw.CT_from_pt(SA, temp_profile)

                # Calculate in-situ density
                rho = gsw.rho(SA, CT, pressure)

                # Check stability (density should increase with depth)
                for iz in range(n_depths - 1):
                    if rho[iz] > rho[iz + 1]:
                        # Unstable! Swap with layer below
                        temp_stable[t, iz + 1, node] = temp_stable[t, iz, node]
                        salt_stable[t, iz + 1, node] = salt_stable[t, iz, node]
                        unstable_count += 1
                        fixed_this_timestep = True

            total_fixes += unstable_count

            if unstable_count == 0:
                break

            iteration += 1

        if iteration >= max_iterations:
            print(f"    WARNING: Time step {t}: Max iterations reached, may still be unstable")

    total_points = n_times * n_depths * n_boundary
    print(f"  Instabilities found and fixed: {total_fixes:,}")
    print(f"  Percentage of data affected: {100*total_fixes/total_points:.2f}%")

    return temp_stable, salt_stable, total_fixes


def save_bc_to_netcdf(
    output_file: str,
    temp_bc: np.ndarray,
    salt_bc: np.ndarray,
    time_list: List[datetime.datetime],
    depth_array: np.ndarray,
    ob_indices_1based: np.ndarray,
    ob_coords: np.ndarray,
    config: Dict[str, Any]
):
    """
    Save boundary conditions to CF-compliant NetCDF file.

    Args:
        output_file: Output file path
        temp_bc: Temperature (n_times, n_depths, n_boundary)
        salt_bc: Salinity
        time_list: List of datetime objects
        depth_array: Depth levels
        ob_indices_1based: 1-indexed boundary node numbers
        ob_coords: Boundary coordinates (n_boundary, 2)
        config: Configuration dictionary
    """
    print("\nStep 8: Saving to NetCDF...")

    n_times, n_depths, n_boundary = temp_bc.shape

    # Convert time to seconds since reference
    time_ref = config['boundary_conditions']['time_reference']
    ref_datetime = datetime.datetime.strptime(time_ref, "%Y-%m-%d %H:%M:%S")
    time_seconds = np.array([(t - ref_datetime).total_seconds() for t in time_list])

    # Create dataset
    ds = xr.Dataset(
        data_vars={
            'temp': (['time', 'depth', 'node'], temp_bc, {
                'long_name': 'temperature',
                'units': 'degree_Celsius',
                'standard_name': 'sea_water_temperature'
            }),
            'salt': (['time', 'depth', 'node'], salt_bc, {
                'long_name': 'salinity',
                'units': 'PSU',
                'standard_name': 'sea_water_salinity'
            }),
            'lon': (['node'], ob_coords[:, 0], {
                'long_name': 'longitude',
                'units': 'degrees_east',
                'standard_name': 'longitude'
            }),
            'lat': (['node'], ob_coords[:, 1], {
                'long_name': 'latitude',
                'units': 'degrees_north',
                'standard_name': 'latitude'
            }),
            'node': (['node'], ob_indices_1based, {
                'long_name': 'boundary node index',
                'units': '1',
                'description': '1-indexed boundary node numbers from nod2d.out'
            })
        },
        coords={
            'time': ('time', time_seconds, {
                'long_name': 'time',
                'units': f'seconds since {time_ref}',
                'standard_name': 'time',
                'axis': 'T'
            }),
            'depth': ('depth', depth_array, {
                'long_name': 'depth',
                'units': 'm',
                'standard_name': 'depth',
                'positive': 'down',
                'axis': 'Z'
            })
        },
        attrs={
            'title': 'FESOM-C Boundary Conditions',
            'source': 'FESOM2 output interpolated to FESOM-C boundary nodes',
            'institution': 'Generated with prepare_bc_fesom.py',
            'history': f'Created on {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
            'Conventions': 'CF-1.8',
            'description': 'Boundary conditions for FESOM-C ocean model',
            'n_boundary_nodes': n_boundary,
            'n_time_steps': n_times,
            'n_depth_levels': n_depths
        }
    )

    # Save to file
    ds.to_netcdf(output_file, encoding={
        'temp': {'zlib': True, 'complevel': 4},
        'salt': {'zlib': True, 'complevel': 4}
    })

    # Get file size
    file_size = Path(output_file).stat().st_size / (1024 * 1024)  # MB

    print(f"  Output file: {output_file}")
    print(f"  Dimensions: time={n_times}, depth={n_depths}, node={n_boundary}")
    print(f"  File size: {file_size:.1f} MB")
    print(f"  Successfully saved")


def print_bc_statistics(
    temp_bc: np.ndarray,
    salt_bc: np.ndarray,
    n_times: int,
    n_depths: int,
    n_boundary: int
):
    """
    Print summary statistics for boundary conditions.

    Args:
        temp_bc: Temperature array
        salt_bc: Salinity array
        n_times: Number of time steps
        n_depths: Number of depth levels
        n_boundary: Number of boundary nodes
    """
    print("\nStep 9: Statistics")
    print(f"  Temperature range: [{np.nanmin(temp_bc):.2f}, {np.nanmax(temp_bc):.2f}]°C")
    print(f"  Temperature mean: {np.nanmean(temp_bc):.2f}°C")
    print(f"  Salinity range: [{np.nanmin(salt_bc):.2f}, {np.nanmax(salt_bc):.2f}] PSU")
    print(f"  Salinity mean: {np.nanmean(salt_bc):.2f} PSU")
    print(f"  Total data points: {n_times} × {n_depths} × {n_boundary} = {n_times * n_depths * n_boundary:,}")


def plot_hovmoller_diagrams(
    temp_bc: np.ndarray,
    salt_bc: np.ndarray,
    time_list: List[datetime.datetime],
    depth_array: np.ndarray,
    ob_coords: np.ndarray,
    ob_indices_1based: np.ndarray,
    ob_depths: np.ndarray,
    config: Dict[str, Any]
):
    """
    Create Hovmuller diagrams (time vs depth) for randomly selected boundary points.

    Args:
        temp_bc: Temperature (n_times, n_depths, n_boundary)
        salt_bc: Salinity
        time_list: List of datetime objects
        depth_array: Depth levels
        ob_coords: Boundary coordinates
        ob_indices_1based: 1-indexed node numbers
        ob_depths: Bathymetry depths at boundary nodes (or None)
        config: Configuration dictionary
    """
    if not MATPLOTLIB_AVAILABLE:
        print("  Skipping: matplotlib not available")
        return

    print("\nStep 10: Creating Hovmuller diagrams...")

    bc_config = config['boundary_conditions']
    n_points = bc_config.get('num_hovmoller_points', 3)
    output_file = bc_config.get('hovmoller_plot_file', 'bc_hovmoller.png')

    n_boundary = len(ob_coords)

    # Select random boundary nodes
    if n_boundary < n_points:
        n_points = n_boundary
        selected_indices = np.arange(n_boundary)
    else:
        selected_indices = np.random.choice(n_boundary, size=n_points, replace=False)

    print(f"  Selected points for Hovmuller plots:")
    for i, idx in enumerate(selected_indices):
        depth_str = f", depth={ob_depths[idx]:.0f}m" if ob_depths is not None else ""
        print(f"    Point {i+1}: node={ob_indices_1based[idx]}, "
              f"lon={ob_coords[idx, 0]:.2f}°E, lat={ob_coords[idx, 1]:.2f}°N{depth_str}")

    # Create figure: n_points rows × 2 columns
    fig, axes = plt.subplots(n_points, 2, figsize=(14, 4*n_points))
    if n_points == 1:
        axes = axes.reshape(1, -1)

    # Convert time to matplotlib dates
    time_dates = mdates.date2num(time_list)

    for i, node_idx in enumerate(selected_indices):
        # Extract data for this node
        temp_profile = temp_bc[:, :, node_idx].T  # (n_depths, n_times)
        salt_profile = salt_bc[:, :, node_idx].T

        # Determine depth limit for this node
        if ob_depths is not None:
            max_depth = ob_depths[node_idx] + 10  # Add 10m margin
            depth_str = f' (Depth: {ob_depths[node_idx]:.0f}m)'
        else:
            max_depth = depth_array[-1]  # Use full depth range
            depth_str = ''

        # Temperature panel
        ax_t = axes[i, 0]
        im_t = ax_t.pcolormesh(time_dates, depth_array, temp_profile,
                               shading='auto', cmap='RdYlBu_r')
        ax_t.set_ylim(max_depth, 0)  # Limit depth and invert
        ax_t.set_ylabel('Depth [m]')
        ax_t.set_title(f'Point {i+1}: Temperature\n'
                       f'Node={ob_indices_1based[node_idx]}, '
                       f'Lon={ob_coords[node_idx, 0]:.2f}°E, '
                       f'Lat={ob_coords[node_idx, 1]:.2f}°N{depth_str}')
        ax_t.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax_t.xaxis.set_major_locator(mdates.MonthLocator(interval=max(1, len(time_list)//10)))
        plt.setp(ax_t.xaxis.get_majorticklabels(), rotation=45, ha='right')
        cbar_t = plt.colorbar(im_t, ax=ax_t)
        cbar_t.set_label('Temperature [°C]')
        ax_t.grid(True, alpha=0.3)

        # Salinity panel
        ax_s = axes[i, 1]
        im_s = ax_s.pcolormesh(time_dates, depth_array, salt_profile,
                               shading='auto', cmap='viridis')
        ax_s.set_ylim(max_depth, 0)  # Limit depth and invert
        ax_s.set_ylabel('Depth [m]')
        ax_s.set_title(f'Point {i+1}: Salinity\n'
                       f'Node={ob_indices_1based[node_idx]}, '
                       f'Lon={ob_coords[node_idx, 0]:.2f}°E, '
                       f'Lat={ob_coords[node_idx, 1]:.2f}°N{depth_str}')
        ax_s.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax_s.xaxis.set_major_locator(mdates.MonthLocator(interval=max(1, len(time_list)//10)))
        plt.setp(ax_s.xaxis.get_majorticklabels(), rotation=45, ha='right')
        cbar_s = plt.colorbar(im_s, ax=ax_s)
        cbar_s.set_label('Salinity [PSU]')
        ax_s.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_file}")


def plot_spatial_bc_distribution(
    temp_bc: np.ndarray,
    salt_bc: np.ndarray,
    time_list: List[datetime.datetime],
    depth_array: np.ndarray,
    ob_coords: np.ndarray,
    ob_indices_1based: np.ndarray,
    config: Dict[str, Any]
):
    """
    Create spatial distribution plots for a random time step and depth level.

    Args:
        temp_bc: Temperature (n_times, n_depths, n_boundary)
        salt_bc: Salinity
        time_list: List of datetime objects
        depth_array: Depth levels
        ob_coords: Boundary coordinates
        ob_indices_1based: 1-indexed node numbers
        config: Configuration dictionary
    """
    if not MATPLOTLIB_AVAILABLE:
        print("  Skipping: matplotlib not available")
        return

    print("\nStep 11: Creating spatial distribution plots...")

    bc_config = config['boundary_conditions']
    output_file = bc_config.get('spatial_plot_file', 'bc_spatial.png')

    n_times, n_depths, n_boundary = temp_bc.shape

    # Select random time and depth
    time_idx = np.random.randint(0, n_times)
    depth_idx = np.random.randint(0, min(5, n_depths))  # Prefer shallow depths

    time_str = time_list[time_idx].strftime('%Y-%m-%d')
    depth_val = depth_array[depth_idx]

    print(f"  Random time step: {time_str} (index {time_idx})")
    print(f"  Random depth level: {depth_val:.1f} m (index {depth_idx})")

    # Extract data slice
    temp_slice = temp_bc[time_idx, depth_idx, :]
    salt_slice = salt_bc[time_idx, depth_idx, :]

    # Create figure: 3 rows × 2 columns
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    node_numbers = np.arange(1, n_boundary + 1)
    lons = ob_coords[:, 0]
    lats = ob_coords[:, 1]

    # Row 1: T/S vs node index
    axes[0, 0].scatter(node_numbers, temp_slice, c=lats, cmap='coolwarm', s=30, alpha=0.7)
    axes[0, 0].set_xlabel('Boundary Node Index')
    axes[0, 0].set_ylabel('Temperature [°C]')
    axes[0, 0].set_title(f'Temperature vs Node Index\n{time_str}, Depth={depth_val:.1f}m')
    axes[0, 0].grid(True, alpha=0.3)

    sc = axes[0, 1].scatter(node_numbers, salt_slice, c=lats, cmap='coolwarm', s=30, alpha=0.7)
    axes[0, 1].set_xlabel('Boundary Node Index')
    axes[0, 1].set_ylabel('Salinity [PSU]')
    axes[0, 1].set_title(f'Salinity vs Node Index\n{time_str}, Depth={depth_val:.1f}m')
    axes[0, 1].grid(True, alpha=0.3)
    cbar = plt.colorbar(sc, ax=axes[0, 1])
    cbar.set_label('Latitude [°N]')

    # Row 2: T/S vs longitude
    axes[1, 0].scatter(lons, temp_slice, c=lats, cmap='coolwarm', s=30, alpha=0.7)
    axes[1, 0].set_xlabel('Longitude [°E]')
    axes[1, 0].set_ylabel('Temperature [°C]')
    axes[1, 0].set_title('Temperature vs Longitude')
    axes[1, 0].grid(True, alpha=0.3)

    sc = axes[1, 1].scatter(lons, salt_slice, c=lats, cmap='coolwarm', s=30, alpha=0.7)
    axes[1, 1].set_xlabel('Longitude [°E]')
    axes[1, 1].set_ylabel('Salinity [PSU]')
    axes[1, 1].set_title('Salinity vs Longitude')
    axes[1, 1].grid(True, alpha=0.3)
    cbar = plt.colorbar(sc, ax=axes[1, 1])
    cbar.set_label('Latitude [°N]')

    # Row 3: T/S vs latitude
    axes[2, 0].scatter(lats, temp_slice, c=lons, cmap='viridis', s=30, alpha=0.7)
    axes[2, 0].set_xlabel('Latitude [°N]')
    axes[2, 0].set_ylabel('Temperature [°C]')
    axes[2, 0].set_title('Temperature vs Latitude')
    axes[2, 0].grid(True, alpha=0.3)

    sc = axes[2, 1].scatter(lats, salt_slice, c=lons, cmap='viridis', s=30, alpha=0.7)
    axes[2, 1].set_xlabel('Latitude [°N]')
    axes[2, 1].set_ylabel('Salinity [PSU]')
    axes[2, 1].set_title('Salinity vs Latitude')
    axes[2, 1].grid(True, alpha=0.3)
    cbar = plt.colorbar(sc, ax=axes[2, 1])
    cbar.set_label('Longitude [°E]')

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_file}")


def main():
    """
    Main function to orchestrate BC preparation workflow.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Prepare FESOM-C Boundary Conditions from FESOM2 output'
    )
    parser.add_argument(
        '--config',
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    args = parser.parse_args()

    # Print header
    print("=" * 80)
    print("FESOM-C Boundary Conditions Preparation")
    print("=" * 80)

    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"ERROR: Configuration file not found: {args.config}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to load configuration: {e}")
        sys.exit(1)

    bc_config = config.get('boundary_conditions', {})

    try:
        # Step 1: Read boundary nodes
        nodes_file = Path(config['input']['fesomc_mesh_dir']) / config['input']['fesomc_nodes_file']
        ob_indices, ob_coords, ob_indices_1based = read_fesomc_boundary_nodes(str(nodes_file))

        # Step 1b: Read boundary depths (optional, for Hovmuller plot depth limits)
        mesh_dir = Path(config['input']['fesomc_mesh_dir'])
        depth_file = mesh_dir / 'depth.out'
        ob_depths = read_boundary_depths(str(depth_file), ob_indices)

        # Step 2: Get spatial bounds
        margin = bc_config.get('margin_degrees', 2.0)
        spatial_bounds = get_boundary_spatial_bounds(ob_coords, margin)

        # Step 3: Read FESOM2 time series
        fesom2_data = read_fesom2_timeseries(config, spatial_bounds)

        # Step 4: Build KDTree
        center_lon = np.mean(ob_coords[:, 0])
        center_lat = np.mean(ob_coords[:, 1])
        kdtree, proj = build_kdtree_for_bc(fesom2_data['coords'], center_lon, center_lat)

        # Step 5: Interpolate to boundary nodes
        temp_bc, salt_bc = interpolate_to_boundary_nodes(
            fesom2_data, ob_coords, kdtree, proj, config
        )

        # Step 6: Fill NaNs
        temp_bc, salt_bc = fill_nans_bc(temp_bc, salt_bc)

        # Step 7: Check density stability
        if bc_config.get('check_density', True):
            max_iter = bc_config.get('max_iterations', 100)
            temp_bc, salt_bc, n_fixes = check_density_stability_bc(
                temp_bc, salt_bc, fesom2_data['depth'], ob_coords, max_iter
            )
        else:
            print("\nStep 7: Density stability check disabled")
            n_fixes = 0

        # Step 8: Save to NetCDF
        output_file = bc_config['output_file']
        save_bc_to_netcdf(
            output_file, temp_bc, salt_bc,
            fesom2_data['time'], fesom2_data['depth'],
            ob_indices_1based, ob_coords, config
        )

        # Step 9: Print statistics
        print_bc_statistics(
            temp_bc, salt_bc,
            len(fesom2_data['time']),
            len(fesom2_data['depth']),
            len(ob_coords)
        )

        # Step 10: Hovmuller diagrams
        if bc_config.get('create_hovmoller_plots', False):
            plot_hovmoller_diagrams(
                temp_bc, salt_bc,
                fesom2_data['time'], fesom2_data['depth'],
                ob_coords, ob_indices_1based, ob_depths, config
            )
        else:
            print("\nStep 10: Hovmuller plots disabled")

        # Step 11: Spatial plots
        if bc_config.get('create_spatial_plots', False):
            plot_spatial_bc_distribution(
                temp_bc, salt_bc,
                fesom2_data['time'], fesom2_data['depth'],
                ob_coords, ob_indices_1based, config
            )
        else:
            print("\nStep 11: Spatial plots disabled")

        # Print summary
        print("\n" + "=" * 80)
        print("Boundary Conditions Preparation Complete!")
        print("=" * 80)
        print(f"Output file: {output_file}")
        print(f"Boundary nodes: {len(ob_coords):,}")
        print(f"Time steps: {len(fesom2_data['time'])}")
        print(f"Depth levels: {len(fesom2_data['depth'])}")
        if n_fixes > 0:
            print(f"Density instabilities fixed: {n_fixes:,}")
        print("=" * 80)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
