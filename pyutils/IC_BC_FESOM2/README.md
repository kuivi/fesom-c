# FESOM-C Initial Conditions and Boundary Conditions Preparation Tools

Python tools to prepare Initial Conditions (IC) and Boundary Conditions (BC) for the FESOM-C ocean model from FESOM2 output data.

## Two Tools Available

### 1. Initial Conditions (IC): `prepare_ic_fesom.py`
Creates a 3D snapshot of temperature and salinity on a regular grid for model initialization.

### 2. Boundary Conditions (BC): `prepare_bc_fesom.py`
Creates time series of temperature and salinity at open boundary nodes for model forcing. See [BC_README.md](BC_README.md) for detailed documentation.

## Overview (IC Tool)

This tool reads FESOM2 3D temperature and salinity data on an unstructured mesh, interpolates it to a regular grid, performs optional smoothing and density stability checks, and outputs a NetCDF file that FESOM-C can read.

## Features

- **Flexible mesh reading**: Reads FESOM-C mesh from ASCII files
- **Smart grid generation**: Automatically creates regular grid covering the target mesh
- **Advanced interpolation**: Uses KDTree with inverse distance weighting for accurate interpolation
- **Depth-by-depth processing**: Efficient 2D interpolation for each depth level
- **Extrapolation**: Fills missing values to ensure complete coverage
- **Optional smoothing**: Savitzky-Golay filter for smoothing the data
- **Density stability check**: Ensures physically stable water column using GSW package
- **CF-compliant NetCDF output**: Standard-compliant output for FESOM-C
- **Diagnostics**: Statistics and optional plotting of surface fields
- **Configurable**: All parameters in YAML config file with command-line overrides

## Installation

### 1. Create virtual environment (recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### Required packages:
- numpy
- scipy
- xarray
- netCDF4
- pyproj
- gsw (Gibbs SeaWater)
- pyyaml
- matplotlib (optional, for plotting)

## Usage

### Basic usage with config file

```bash
python prepare_ic_fesom.py
```

This will use the default `config.yaml` file.

### Use custom config file

```bash
python prepare_ic_fesom.py --config my_config.yaml
```

### Override config parameters

```bash
# Use 15 nearest neighbors instead of default
python prepare_ic_fesom.py --k-neighbors 15

# Enable smoothing
python prepare_ic_fesom.py --smoothing

# Disable smoothing
python prepare_ic_fesom.py --no-smoothing

# Enable plotting
python prepare_ic_fesom.py --plot

# Disable plotting
python prepare_ic_fesom.py --no-plot
```

### Get help

```bash
python prepare_ic_fesom.py --help
```

## Configuration

All parameters are configured in `config.yaml`. The file is organized into sections:

### Input files

```yaml
input:
  fesomc_mesh_dir: "./fesomc/mesh/"
  fesomc_nodes_file: "nod2d.out"
  fesom2_data_dir: "./fesom2/"
  fesom2_mesh_file: "fesom.mesh.diag.nc"
  fesom2_temp_file: "temp.fesom.2020.nc"
  fesom2_salt_file: "salt.fesom.2020.nc"
  fesom2_time_index: 0  # Which time step to use
```

### Regular grid

```yaml
regular_grid:
  resolution: [500, 500]  # Number of points [lon, lat]
  margin_degrees: 1.0  # Extra margin around mesh
```

### Interpolation

```yaml
interpolation:
  k_neighbors: 10  # Number of nearest neighbors
  use_idw: true  # Inverse distance weighting
  idw_power: 2  # Power for IDW (w = 1/d^2)
```

### Smoothing (optional)

```yaml
smoothing:
  enabled: false
  window_length: 3  # Savitzky-Golay window
  polyorder: 1  # Polynomial order
```

### Advanced Plotting (optional)

```yaml
diagnostics:
  create_plots: true  # Basic surface plots
  create_profile_plots: false  # T/S profiles at random points
  num_profile_points: 3  # Number of profile locations
  create_native_mesh_plot: false  # Plot on FESOM-C mesh
```

### Stability check

```yaml
stability:
  check_density: true  # Check and fix instabilities
```

### Output

```yaml
output:
  output_file: "ic_fesomc_from_fesom2.nc"
  add_boundary_depths: true  # Add depths at -20m and 10000m
```

See [`config.yaml`](config.yaml) for complete configuration options.

## Workflow

The tool follows these steps:

1. **Read FESOM-C mesh** from ASCII file (`nod2d.out`)
2. **Create regular grid** covering the mesh with specified resolution
3. **Read FESOM2 data** (temperature, salinity, coordinates, depths)
4. **Interpolate to regular grid** using KDTree and inverse distance weighting
   - Depth-by-depth 2D interpolation
   - Extrapolation to fill NaNs
5. **Apply smoothing** (optional) using Savitzky-Golay filter
6. **Check density stability** using GSW package
   - Fix instabilities by swapping T/S values
7. **Save to NetCDF** with CF-compliant attributes
8. **Print statistics** (min, max, mean for T and S)
9. **Create plots** (optional):
   - **Surface plots**: Temperature and salinity with FESOM-C mesh nodes overlay
   - **Profile plots**: T/S profiles at random mesh points (dual x-axes)
   - **Native mesh plots**: Data visualized on FESOM-C triangular/quad mesh elements

## Input Data Requirements

### FESOM-C Mesh

- **nod2d.out**: ASCII file with node coordinates
  - Format: `index longitude latitude boundary_flag`
  - First line: total number of nodes
- **depth.out**: ASCII file with bathymetry depth at each node (optional)
  - Used for profile plots to set appropriate Y-axis limits
  - One value per line (meters)

### FESOM2 Output

- **fesom.mesh.diag.nc**: Mesh coordinates
  - Variables: `lon`, `lat`
- **temp.fesom.YYYY.nc**: Temperature data
  - Dimensions: `(time, nz1, x)`
  - Coordinates: `nz1` (depth levels)
- **salt.fesom.YYYY.nc**: Salinity data
  - Same structure as temperature

## Output

NetCDF file with the following structure:

```
Dimensions:
  depth: nz (+ 2 if boundary depths enabled)
  lat: nlat
  lon: nlon

Variables:
  lon(lon): longitude [degrees_east]
  lat(lat): latitude [degrees_north]
  depth(depth): depth [m]
  temp(depth, lat, lon): temperature [degree_Celsius]
  salt(depth, lat, lon): salinity [PSU]
```

The output file can be directly used by FESOM-C for initial conditions.

## Algorithm Details

### Interpolation

The tool uses inverse distance weighting (IDW) for interpolation:

1. Build KDTree from FESOM2 node coordinates (converted to Cartesian)
2. For each regular grid point, find k nearest neighbors
3. Calculate weights: w = 1 / distance^power
4. Interpolated value = Σ(w × value) / Σ(w)

### Density Stability

Uses GSW (Gibbs SeaWater) package:

1. Calculate Absolute Salinity from Practical Salinity
2. Calculate Conservative Temperature from Potential Temperature
3. Calculate in-situ density
4. Check if density increases with depth
5. If not, swap T/S values with layer above
6. Iterate until stable

### Coverage Check

Verifies that all FESOM-C mesh nodes are covered by the regular grid within ±2 grid cells. Warns if coverage is incomplete.

## Examples

### Example 1: Basic usage

```bash
# Use default config
python prepare_ic_fesom.py
```

### Example 2: Higher resolution with smoothing

Modify `config.yaml`:
```yaml
regular_grid:
  resolution: [1000, 1000]

smoothing:
  enabled: true
  window_length: 3
```

Run:
```bash
python prepare_ic_fesom.py
```

### Example 3: Nearest neighbor interpolation

```bash
python prepare_ic_fesom.py --k-neighbors 1
```

This will use nearest neighbor instead of IDW.

## Troubleshooting

### Coverage warnings

If you see warnings about uncovered FESOM-C nodes:
- Increase `margin_degrees` in config
- Check that FESOM2 data covers the FESOM-C domain

### Memory issues

For very large grids:
- Reduce regular grid resolution
- Process in chunks (manual modification needed)

### NaN values in output

- Check FESOM2 input data quality
- Increase `max_iterations` for extrapolation
- Check data filtering thresholds

## Performance

- **Small grid** (100×100): ~1 minute
- **Medium grid** (500×500): ~5-10 minutes
- **Large grid** (1000×1000): ~20-30 minutes

Time depends on:
- Number of FESOM2 nodes
- Regular grid resolution
- Number of depth levels
- Number of neighbors (k)
- Whether smoothing is enabled

## References

- **FESOM-C**: https://github.com/kuivi/fesom-c ; Kuznetsov, I., Rabe, B., Androsov, A., Fang, Y.-C., Hoppmann, M., Quintanilla-Zurita, A., Harig, S., Tippenhauer, S., Schulz, K., Mohrholz, V., Fer, I., Fofonova, V., and Janout, M.: Dynamical reconstruction of the upper-ocean state in the central Arctic during the winter period of the MOSAiC expedition, Ocean Sci., 20, 759–777, https://doi.org/10.5194/os-20-759-2024, 2024.
- **FESOM2**: https://fesom2.readthedocs.io/
- **GSW**: Gibbs SeaWater Oceanographic Toolbox, https://www.teos-10.org/

## License

This tool is provided as-is for scientific research purposes.

## Author

Generated with Claude Code (2026-01-11)

## Support

For issues or questions:
1. Check the [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for details
2. Review the `config.yaml` settings
3. Check input data format and quality
