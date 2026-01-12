# Advanced Plotting Features Guide

## Overview

The FESOM-C IC preparation tool now includes three types of plots to visualize and validate the initial conditions:

1. **Surface plots** - Regular grid with mesh overlay
2. **Profile plots** - Vertical T/S profiles at selected points
3. **Native mesh plots** - Data on FESOM-C mesh elements

## Configuration

Enable these features in your `config.yaml`:

```yaml
diagnostics:
  # Basic surface plots (always recommended)
  create_plots: true
  plot_file: "ic_surface_plot.png"

  # T/S profile plots at random points
  create_profile_plots: true
  profile_plot_file: "ic_profiles.png"
  num_profile_points: 3  # Number of locations

  # Native FESOM-C mesh visualization
  create_native_mesh_plot: true
  native_mesh_plot_file: "ic_native_mesh.png"
```

## Plot Types

### 1. Surface Plots (`ic_surface_plot.png`)

**What it shows:**
- Left panel: Surface temperature on regular grid
- Right panel: Surface salinity on regular grid
- Black semi-transparent dots: All FESOM-C mesh nodes
- Red numbered circles: Profile point locations (if enabled)

**Use for:**
- Quick visual check of interpolated data quality
- Verifying mesh coverage
- Identifying where profiles are located

**Features:**
- Contour maps with colorbar
- Mesh nodes overlay (alpha=0.5)
- Profile points marked with red circles and numbers
- Latitude/Longitude axes

### 2. Profile Plots (`ic_profiles.png`)

**What it shows:**
- One panel per profile point (3 rows by default)
- Dual x-axes: Temperature (bottom, blue) and Salinity (top, red)
- Y-axis: Depth (inverted, increases downward)
- Points and connecting lines show T/S variation with depth

**Use for:**
- Checking vertical structure of water column
- Verifying density stability was applied correctly
- Comparing T/S characteristics at different locations

**Features:**
- Dual x-axes for simultaneous T and S visualization
- Color-coded axes (blue=T, red=S)
- Grid lines for easy reading
- Location coordinates and bathymetry depth in title
- **Y-axis automatically scaled** to actual node depth (+10m margin)
  - Avoids showing irrelevant deep water when node is shallow
  - E.g., 40m deep node shows 0-50m, not 0-6000m

**Example interpretation:**
- Temperature typically decreases with depth
- Salinity may increase, decrease, or remain constant
- Check for realistic oceanographic structure

### 3. Native Mesh Plots (`ic_native_mesh.png`)

**What it shows:**
- Left panel: Temperature on FESOM-C mesh elements
- Right panel: Salinity on FESOM-C mesh elements
- Triangular/quadrilateral mesh cells
- Data values at element centers (averaged from nodes)

**Use for:**
- Visualizing how data looks on the actual model mesh
- Checking for interpolation artifacts
- Verifying mesh structure
- Understanding spatial resolution variations

**Features:**
- PolyCollection rendering (fast for large meshes)
- Element-averaged values
- Same colormaps as surface plots
- Equal aspect ratio

**Technical notes:**
- Requires `elem2d.out` file
- Handles both triangles and quadrilaterals
- Automatically converts 1-indexed to 0-indexed

## How Profile Points are Selected

The tool automatically selects random points for profiles:

1. **Preference for internal nodes**: Avoids boundary nodes (obindex≠2)
2. **Random selection**: Different each run (use `np.random.seed()` for reproducibility)
3. **Printed coordinates**: Check console output for selected locations

Example output:
```
Selecting 3 random points for profile plots...
  Selected points: [3654 34603 12897]
    Point 1: lon=7.10°E, lat=53.59°N
    Point 2: lon=7.29°E, lat=54.98°N
    Point 3: lon=8.28°E, lat=55.33°N
```

## Performance

| Plot Type | File Size | Time | Memory |
|-----------|-----------|------|--------|
| Surface | ~340 KB | ~1 s | Low |
| Profiles | ~100 KB | ~2-5 s | Low |
| Native Mesh | ~700 KB | ~5-10 s | Medium |

Times are for typical mesh (~40k nodes). Native mesh plot time scales with number of elements.

## Examples

### Enable all plots in test config

```bash
# Edit config_test.yaml
diagnostics:
  create_plots: true
  create_profile_plots: true
  create_native_mesh_plot: true

# Run
.venv/bin/python prepare_ic_fesom.py --config config_test.yaml
```

### Disable advanced plots for production

```bash
# Edit config.yaml
diagnostics:
  create_plots: true          # Keep basic plots
  create_profile_plots: false  # Skip profiles (saves time)
  create_native_mesh_plot: false  # Skip mesh plot
```

## Interpreting the Plots

### Surface Plots Quality Checks

✅ **Good signs:**
- Smooth contours without artifacts
- All mesh nodes (black dots) covered by data
- Profile points (red circles) well-distributed
- No unexpected patterns or discontinuities

⚠️ **Warning signs:**
- Checkerboard patterns (may indicate interpolation issues)
- Large gaps in mesh coverage
- Sharp discontinuities (check data quality)

### Profile Plots Quality Checks

✅ **Good signs:**
- Smooth T/S curves with depth
- Temperature generally decreases with depth
- Density increases with depth (checked automatically)
- Realistic values for your region

⚠️ **Warning signs:**
- Erratic jumps in T or S
- Unrealistic values
- Temperature inversions (should be fixed by stability check)

### Native Mesh Plots Quality Checks

✅ **Good signs:**
- Smooth color transitions between elements
- Mesh structure visible and reasonable
- Similar patterns to surface plots
- No isolated extreme values

⚠️ **Warning signs:**
- Missing elements (NaN values)
- Discontinuities at mesh boundaries
- Different patterns than surface plots

## Tips

1. **Always check surface plots first** - They're the quickest way to spot issues

2. **Use profile plots for detailed validation** - Check a few key locations

3. **Use native mesh plots sparingly** - They're slower but show actual model representation

4. **Compare with source data** - If possible, compare with original FESOM2 fields

5. **Adjust number of profile points** - More points = better spatial coverage, but slower

6. **Set random seed for reproducibility**:
   ```python
   # Add to script if needed
   np.random.seed(42)
   ```

## Troubleshooting

### "matplotlib not available"
```bash
.venv/bin/pip install matplotlib
```

### "Elements file not found"
- Check that `fesomc/mesh/elem2d.out` exists
- Set correct path in config
- Or disable native mesh plots

### Profile plots look strange
- Check that regular grid resolution is adequate
- Verify interpolation settings
- Check source data quality

### Native mesh plot is slow
- Normal for large meshes (>50k elements)
- Consider disabling for routine runs
- Only needed for detailed validation

## Related Files

- [config.yaml](config.yaml) - Production configuration
- [config_test.yaml](config_test.yaml) - Test configuration (plots enabled)
- [README.md](README.md) - Main documentation
- [prepare_ic_fesom.py](prepare_ic_fesom.py) - Main script

## Viewing Plots

```bash
# macOS
open ic_surface_plot_test.png
open ic_profiles_test.png
open ic_native_mesh_test.png

# Linux
xdg-open ic_surface_plot_test.png
# or
eog ic_*.png

# Windows
start ic_surface_plot_test.png
```

Happy plotting! 📊🌊
