# FESOM-C Boundary Conditions Preparation Tool

A Python tool to prepare time series of Boundary Conditions (BC) for the FESOM-C ocean model from FESOM2 output data.

## Overview

This tool extracts data at open boundary nodes from FESOM2 output and interpolates it to FESOM-C boundary nodes across multiple time steps (months/years). It produces a CF-compliant NetCDF file with dimensions (time, depth, node) that FESOM-C can use for boundary forcing.

## Key Differences from IC Preparation

| Feature | IC Script | BC Script |
|---------|-----------|-----------|
| **Purpose** | Initial conditions (3D snapshot) | Boundary conditions (time series) |
| **Target nodes** | All mesh nodes | Only boundary nodes (index==2) |
| **Intermediate grid** | Regular lon-lat grid | Direct interpolation to nodes |
| **Time handling** | Single time step | Multiple years/months |
| **Output dims** | (depth, lat, lon) | (time, depth, node) |
| **Plotting** | Surface + profiles | Hovmuller + spatial BC |

## Features

- **Boundary-only processing**: Extracts only open boundary nodes (index==2)
- **Time series support**: Processes multiple years and months
- **Memory-efficient**: Spatial filtering reduces data by 99%+
- **KDTree interpolation**: Fast inverse distance weighting
- **Density stability**: GSW-based stability checking for all time steps
- **Hovmuller diagrams**: Time vs depth visualizations
- **Spatial BC plots**: T/S distribution along boundary
- **CF-compliant NetCDF**: Output ready for FESOM-C

## Installation

Same requirements as IC preparation tool. If you already set up the IC tool, you're ready to go:

```bash
source .venv/bin/activate  # If using virtual environment
```

Dependencies already installed:
- numpy, scipy, xarray, netCDF4
- pyproj, gsw, pyyaml
- matplotlib

## Quick Start

### 1. Configure BC preparation

Edit `config.yaml` or `config_test.yaml`:

```yaml
boundary_conditions:
  enabled: true  # Enable BC preparation
  years: [2020]  # Years to process
  months: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # All months
```

### 2. Run the script

```bash
# Test with config_test.yaml (faster, plots enabled)
.venv/bin/python prepare_bc_fesom.py --config config_test.yaml

# Or production run with config.yaml (plots disabled)
.venv/bin/python prepare_bc_fesom.py --config config.yaml
```

## Features

- **Boundary node extraction**: Automatically identifies open boundary nodes (index==2)
- **Spatial filtering**: Reduces FESOM2 data by 99% using boundary-region filtering
- **Multi-year/month processing**: Handles time series across multiple years
- **KDTree interpolation**: Fast IDW interpolation to boundary nodes
- **Density stability check**: Ensures physically stable water columns using GSW
- **CF-compliant NetCDF output**: Standard format for FESOM-C
- **Hovmuller diagrams**: Time vs depth visualization at selected points
- **Spatial plots**: BC distribution along boundary nodes

## Quick Start

### Basic usage (production)

```bash
# Edit config.yaml to enable BC preparation
# Set boundary_conditions.enabled: true

.venv/bin/python prepare_bc_fesom.py
```

### Test with sample data

```bash
# Uses config_test.yaml (BC enabled, all plots enabled)
.venv/bin/python prepare_bc_fesom.py --config config_test.yaml
```

## Configuration

BC configuration is in the `boundary_conditions` section of `config.yaml`:

```yaml
boundary_conditions:
  enabled: true  # Must be true to run BC preparation

  # Time period
  years: [2020]  # List of years to process
  months: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # All months

  # Input files
  fesom2_temp_pattern: "temp.fesom.{year}.nc"  # {year} replaced with actual year
  fesom2_salt_pattern: "salt.fesom.{year}.nc"

  # Interpolation
  interpolation:
    k_neighbors: 10
    use_idw: true
    idw_power: 2

  # Output
  output_file: "bc_fesomc_from_fesom2.nc"

  # Plotting
  create_hovmoller_plots: true
  create_spatial_plots: true
```

## Input Data Requirements

### FESOM-C Mesh
- **nod2d.out**: Node coordinates with boundary flag
  - Boundary nodes identified by `index==2`
  - Only these nodes are extracted

### FESOM2 Output
- **fesom.mesh.diag.nc**: Mesh coordinates
- **temp.fesom.YYYY.nc**: Temperature for each year
- **salt.fesom.YYYY.nc**: Salinity for each year
- Multiple years and months specified in config

## Output

NetCDF file with the following structure:

```
Dimensions:
  time: n_time_steps (e.g., 12 for 1 year monthly data)
  depth: n_depths (e.g., 47 levels)
  node: n_boundary (number of boundary nodes)

Coordinates:
  time(time): seconds since reference date
  depth(depth): depth levels [m]

Variables:
  node(node): 1-indexed boundary node numbers
  lon(node): longitude of boundary nodes [degrees_east]
  lat(node): latitude of boundary nodes [degrees_north]
  temp(time, depth, node): temperature [degree_Celsius]
  salt(time, depth, node): salinity [PSU]
```

The output file can be directly used by FESOM-C for boundary conditions.

## Algorithm Details

### Spatial Filtering

Critical for memory efficiency! The script:

1. Finds rectangular bounds around boundary nodes
2. Adds margin (default 2.0°)
3. Filters FESOM2 mesh to only load nodes in this region
4. **Reduces data by ~99%** (12.9M → ~84k nodes in test)

### Interpolation

Uses the same method as IC preparation:
1. Build KDTree from filtered FESOM2 coordinates (stereographic projection)
2. For each boundary node, find k nearest neighbors
3. Apply inverse distance weighting (IDW)
4. Process all time steps and depth levels

### Density Stability

The density stability check is **critical for BC data**:
- Uses GSW (Gibbs SeaWater) package
- Checks that density increases monotonically with depth
- Fixes instabilities by copying T/S from layer above
- Iterates until stable
- Note: High percentage of fixes (>50%) may indicate data quality issues

### Time Handling

- Each month uses time index 0 from the monthly file
- Time stored as 15th of each month (middle of month)
- Units: seconds since reference date (configurable)

## Output

NetCDF file with structure:

```
Dimensions:
  time: n_times (12 for 1 year)
  depth: n_depths (47 levels)
  node: n_boundary (203 in test case)

Variables:
  time(time): seconds since reference date
  depth(depth): depth levels [m]
  node(node): 1-indexed boundary node numbers
  lon(node): longitude of boundary nodes [degrees_east]
  lat(node): latitude of boundary nodes [degrees_north]
  temp(time, depth, node): temperature [degree_Celsius]
  salt(time, depth, node): salinity [PSU]
```

## Performance

**Test Results (1 year, 12 months):**
- Boundary nodes: 203
- Time steps: 12
- Depth levels: 47
- FESOM2 nodes: 12.9M → 83k after spatial filtering (99.4% reduction)
- Runtime: ~2 minutes (mostly interpolation and stability check)
- Memory usage: ~2-3 GB
- Output file: 0.4 MB

**Bottlenecks:**
1. Reading FESOM2 data: ~30 seconds (spatial filtering helps!)
2. Interpolation: ~30 seconds (203 boundary nodes × 12 months × 47 depths)
3. Density stability: ~60 seconds (high number of instabilities in this test case)

## Output Files

Running the script with `config_test.yaml` generates:

```
bc_fesomc_from_fesom2_test.nc     # NetCDF with BC time series
bc_hovmoller_test.png             # Hovmuller diagrams (time vs depth)
bc_spatial_test.png               # Spatial BC distribution plots
```

## Plotting Features

### Hovmuller Diagrams
Shows temporal evolution of T/S at selected boundary points:
- X-axis: Time (months)
- Y-axis: Depth (inverted)
- Colors: Temperature or Salinity values
- One row per random boundary point

### Spatial Distribution Plots
Shows spatial patterns at a random time/depth slice:
- **Top row**: T/S vs boundary node index
- **Middle row**: T/S vs longitude
- **Bottom row**: T/S vs latitude
- Colors indicate latitude or longitude

## Configuration Examples

### Quick Test (1 month)
```yaml
boundary_conditions:
  enabled: true
  years: [2020]
  months: [1]  # Only January
```

### Full Year
```yaml
boundary_conditions:
  enabled: true
  years: [2020]
  months: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
```

### Multi-Year Production
```yaml
boundary_conditions:
  enabled: true
  years: [2018, 2019, 2020]
  months: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
```

## Troubleshooting

### No boundary nodes found
- Check that nod2d.out has nodes with index==2
- Boundary nodes are typically at open boundaries (not coastlines)

### High percentage of density instabilities
- Normal for some datasets (especially near boundaries)
- Algorithm iteratively fixes all instabilities
- Check output data to verify it's physically reasonable

### Memory issues
- Reduce number of years/months processed at once
- Spatial filtering already reduces memory by 99%
- Process in chunks if needed (modify script)

### Slow interpolation
- Normal for large numbers of time steps
- Consider reducing k_neighbors for faster (but less accurate) interpolation
- Parallelization possible (future enhancement)

## Comparison with IC Script

| Feature | IC Script | BC Script |
|---------|-----------|-----------|
| Target | Full mesh | Boundary nodes only |
| Grid | Regular lon-lat | Direct to nodes |
| Time | Single snapshot | Time series |
| Output dims | (depth, lat, lon) | (time, depth, node) |
| File size | ~50 MB | ~0.4 MB per year |
| Runtime | ~10 min | ~2 min per year |
| Plots | Surface + profiles | Hovmuller + spatial |

## References

- **FESOM-C**: Finite Element Sea ice-Ocean Model - C version
- **FESOM2**: https://fesom2.readthedocs.io/
- **GSW**: Gibbs SeaWater Oceanographic Toolbox, https://www.teos-10.org/

## Author

Generated with Claude Code (2026-01-12)

## Support

For issues or questions:
1. Check [BC_IMPLEMENTATION_PLAN.md](BC_IMPLEMENTATION_PLAN.md) for technical details
2. Review config settings in `config.yaml` or `config_test.yaml`
3. Check input data format and availability
4. Verify boundary nodes exist in mesh (index==2)
