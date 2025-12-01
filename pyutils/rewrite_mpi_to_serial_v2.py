#!/usr/bin/env python3
"""
Script to convert an MPI-generated netCDF file from FESOM-C model into a corrected serial netCDF file using.

time_chunk_size - the number of time steps to read in one chunk (less memory -> smaller chunk size)

example of use :

srun python rewrite_mpi_to_serial_v2.py ./pechora_ice.nc   

Author: I.Kuznetsov (kuivi)
Date: 2025-03-25
"""
import sys
import os
import numpy as np
import xarray as xr

time_chunk_size = 10

# Check command-line arguments
if len(sys.argv) < 2:
    print("No input file given.")
    sys.exit(1)

fnamein = sys.argv[1]
if not os.path.isfile(fnamein):
    print("File not found:", fnamein)
    sys.exit(1)

# Use second argument if provided, otherwise default
if len(sys.argv) > 2:
    fnameout = sys.argv[2]
else:
    fnameout = fnamein[:-3] + '_corrected.nc'

print("InFile:", fnamein)
print("OutFile:", fnameout)

# Remove output file if it exists
if os.path.isfile(fnameout):
    os.remove(fnameout)

'''
# Open dataset with xarray using dask chunks.
# Adjust chunk sizes as appropriate for your dataset.
ds = xr.open_dataset(fnamein, decode_times=False)
# Define available memory (in bytes), for example, 2GB
available_memory = 18 * 1024**3  # 18GB in bytes
# To be safe, use only 60% of available memory for chunking
memory_budget = 0.6 * available_memory
# Example: Estimating memory per time slice for a single variable
# Let's assume:
#   - The 'node' dimension is used (all nodes are read each time step)
#   - Each element is a 32-bit float (4 bytes)
#   - You have two main variables to process
node_count = ds.sizes['node']
bytes_per_element = 4  # for float32
n_variables = 2  # adjust if you have more than 2 variables of interest
# Memory required for one time slice (over all nodes, for the two variables)
memory_per_time_slice = node_count * bytes_per_element * n_variables

# Compute the number of time steps per chunk
time_chunk_size = int(memory_budget // memory_per_time_slice)

time_chunk_size = 100
print("Estimated time chunk size:", time_chunk_size)

if (ds.sizes['time'] <= time_chunk_size):
    time_chunk_size = ds.sizes['time']
'''
ds = xr.open_dataset(fnamein, chunks={'time': time_chunk_size}, decode_times=False)

# Compute unique indices for 'node_idx' and 'elem_idx'
node_idx = ds['node_idx'].values
elem_idx = ds['elem_idx'].values

# np.unique returns sorted unique values and the indices of their first occurrence.
_, sortn_idx = np.unique(node_idx, return_index=True)
_, sorte_idx = np.unique(elem_idx, return_index=True)

# Reindex the entire dataset with drop=True on both dimensions
ds_reordered = ds.isel(node=sortn_idx, drop=True).isel(nele=sorte_idx, drop=True)

# Write out the new dataset
ds_reordered.to_netcdf(fnameout, engine='netcdf4', compute=True)#, parallel=True)
