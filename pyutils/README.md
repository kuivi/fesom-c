# FESOM-C Python Tools

Python utilities for working with FESOM-C (Finite Volume Ocean Model) output and mesh generation for coastal regions.

## Scripts

### Mesh Processing
- **mesh_fesomc.py** - FESOM-C mesh class for loading and processing unstructured mesh data (supports 'drift' and 'rise' models)
- **load_mesh_c_data.py** - Load mesh and data from FESOM-C output files
- **msh2shp.py** - Convert FESOM-C mesh to shapefile format

### Data Regridding
- **fv2regular.py** - Regrid FESOM-C output to regular grid in NetCDF format
- **interp_lines_z_sigma.py** - Regrid FESOM-C data to transects with sigma-to-z interpolation
- **interp_lines_z_1.py** - Regrid FESOM-C output to transects defined by shapefile lines

### Forcing Data Generation
- **bathymetry_generation_v9_1_smoothed.py** - Generate smoothed bathymetry data for FESOM-C mesh
- **tides_generation.py** - Generate tidal forcing data for FESOM-C using pyTMD
- **fesomc_gettides_mask.py** - Extract tidal amplitudes and phases from FESOM-C output and regrid
- **rivers_generation_v9_1.py** - Generate river runoff forcing data for FESOM-C

### Utilities
- **load_nodal_corrections.py** - Load nodal corrections for tidal calculations using pyTMD
- **rewrite_mpi_to_serial_v2.py** - Convert MPI-generated NetCDF to serial format

## Author

I. Kuznetsov (kuivi)

## Dependencies

Common dependencies include: numpy, netCDF4, scipy, pyproj, geopandas, shapely, pyTMD, xarray
