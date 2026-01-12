# Quick Start Guide

## Run the Tool

### Test Run (Fast - 2-3 minutes)
```bash
.venv/bin/python prepare_ic_fesom.py --config config_test.yaml
```

This creates:
- `ic_fesomc_from_fesom2_test.nc` - NetCDF file for FESOM-C
- `ic_surface_plot_test.png` - Surface temperature and salinity plots

### Production Run (Full Resolution)
```bash
.venv/bin/python prepare_ic_fesom.py --config config.yaml
```

This uses 500×500 grid instead of 100×100 (takes ~10-15 minutes).

## Check the Results

### View the NetCDF file
```bash
# Check structure
ncdump -h ic_fesomc_from_fesom2_test.nc

# With Python
.venv/bin/python -c "
import xarray as xr
ds = xr.open_dataset('ic_fesomc_from_fesom2_test.nc')
print(ds)
print('\nSurface temperature stats:')
print(ds.temp[0].describe())
"
```

### View the plot
```bash
# On macOS
open ic_surface_plot_test.png

# On Linux
xdg-open ic_surface_plot_test.png
```

## Customize for Your Data

### Edit config.yaml
```yaml
# Change input files
input:
  fesom2_temp_file: "temp.fesom.2021.nc"  # Use different year
  fesom2_time_index: 6  # Use July instead of January

# Change grid resolution
regular_grid:
  resolution: [200, 200]  # Finer grid

# Change interpolation
interpolation:
  k_neighbors: 15  # Use more neighbors

# Enable smoothing
smoothing:
  enabled: true

# Change output filename
output:
  output_file: "ic_summer_2021.nc"
```

Then run:
```bash
.venv/bin/python prepare_ic_fesom.py
```

## Command-Line Overrides

Don't want to edit config file? Use command-line arguments:

```bash
# Use 15 neighbors
.venv/bin/python prepare_ic_fesom.py --k-neighbors 15

# Enable smoothing
.venv/bin/python prepare_ic_fesom.py --smoothing

# With plots
.venv/bin/python prepare_ic_fesom.py --plot

# All together
.venv/bin/python prepare_ic_fesom.py --k-neighbors 15 --smoothing --plot
```

## Common Issues

### "No FESOM2 nodes found in region"
- Check that coordinate systems match (both in degrees)
- Increase `margin_degrees` in config

### Memory issues
- Reduce `resolution` in config (e.g., [100, 100])
- The spatial filtering already handles the biggest memory issue

### Slow processing
- Reduce grid resolution
- Reduce `k_neighbors` (try 5 instead of 10)
- Use `--no-smoothing` flag

## What's Next?

1. **Verify the output**: Load the NC file in your favorite tool (ncview, Panoply, Python)
2. **Check the statistics**: Make sure T/S ranges are reasonable
3. **Inspect the plots**: Look for any obvious artifacts
4. **Use with FESOM-C**: Copy the NC file to your FESOM-C input directory
5. **Run full resolution**: Once satisfied with test, run with production config

## File Locations

All files are in the current directory:
```
./prepare_ic_fesom.py           # Main script
./config.yaml                    # Production config
./config_test.yaml               # Test config
./README.md                      # Full documentation
./SUMMARY.md                     # Project summary
```

## Get Help

```bash
.venv/bin/python prepare_ic_fesom.py --help
```

## Congratulations! 🎉

Your FESOM-C Initial Conditions tool is ready to use!
