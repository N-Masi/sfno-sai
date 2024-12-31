
# File path
import xarray as xr
import numpy as np
import sys
from pathlib import Path

def print_and_write(message, file=None):
    """Print to console and write to file if provided"""
    print(message)
    if file:
        file.write(message + '\n')

# File paths
nc_file =  "/teamspace/s3_connections/ncar-cesm2-arise-bucket/CESM2-WACCM-SSP245/b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.001/atm/proc/tseries/day_1/b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.001.cam.h1.T.20150101-20241231.nc"
output_file = "/teamspace/studios/this_studio/netcdf-vars.txt"

# Make sure output directory exists
Path(output_file).parent.mkdir(parents=True, exist_ok=True)

try:
    # Open the netCDF file
    ds = xr.open_dataset(nc_file)
    
    # Open text file for writing
    with open(output_file, 'w') as f:
        # Write file header
        header = f"NetCDF File Analysis for: {nc_file}\n" + "="*80
        print_and_write(header, f)
        
        # Dataset Info
        print_and_write("\n=== Dataset Info ===", f)
        print_and_write(str(ds), f)
        
        # Dimensions
        print_and_write("\n=== Dimensions ===", f)
        for dim in ds.dims:
            print_and_write(f"{dim}: {ds.dims[dim]}", f)
        
        # Variables
        print_and_write("\n=== Variables ===", f)
        for var in ds.variables:
            print_and_write(f"\nVariable: {var}", f)
            print_and_write(f"Shape: {ds[var].shape}", f)
            print_and_write("Attributes:", f)
            for attr, value in ds[var].attrs.items():
                print_and_write(f"  {attr}: {value}", f)
        
        # Coordinates
        print_and_write("\n=== Coordinates ===", f)
        print_and_write(str(ds.coords), f)

    print(f"\nOutput written to: {output_file}")

except Exception as e:
    error_msg = f"Error processing file: {str(e)}"
    print(error_msg)
    with open(output_file, 'a') as f:
        f.write(f"\nERROR: {error_msg}")
    
finally:
    if 'ds' in locals():
        ds.close()