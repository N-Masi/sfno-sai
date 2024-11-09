import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path

def get_pressure_levels(ds):
    """
    Calculate pressure levels from hybrid coordinates
    
    Parameters:
        ds: xarray Dataset containing hybrid level coordinates
    Returns:
        pressure levels in hPa
    """
    # Get reference pressure in hPa
    P0 = ds.P0 / 100  # Convert Pa to hPa
    
    # Get hybrid coefficients
    hyam = ds.hyam
    hybm = ds.hybm
    
    # Calculate approximate pressure levels at midpoints
    p_levels = P0 * (hyam + hybm)
    
    return p_levels

def find_nearest_levels(p_levels, target_levels):
    """
    Find indices of nearest pressure levels to targets
    
    Parameters:
        p_levels: array of actual pressure levels
        target_levels: array of desired pressure levels
    Returns:
        indices of nearest levels
    """
    indices = []
    for target in target_levels:
        idx = np.abs(p_levels - target).argmin()
        indices.append(idx)
    return indices

def write_level_info(file_path, output_file):
    """Write comprehensive level information to file"""
    
    with open(output_file, 'w') as f:
        # Write header
        f.write("CESM2 Pressure Level Analysis\n")
        f.write("=" * 50 + "\n\n")
        
        # Open dataset
        ds = xr.open_dataset(file_path)
        p_levels = get_pressure_levels(ds)
        
        # Target levels from paper
        target_levels = np.array([0.6, 10, 20, 30, 50, 70, 120, 200, 400, 600, 800, 1000])
        level_indices = find_nearest_levels(p_levels, target_levels)
        
        # Write target level mapping
        f.write("=== Target Level Mapping ===\n")
        f.write("Target (hPa) -> Actual (hPa) [Index] : Description\n")
        descriptions = [
            "Mesosphere",
            "Upper stratosphere",
            "Mid-stratosphere",
            "Lower stratosphere (SAI region)",
            "Lower stratosphere (SAI region)",
            "Tropopause",
            "Upper troposphere",
            "Mid-troposphere",
            "Lower troposphere",
            "Near surface",
            "Near surface",
            "Surface"
        ]
        
        for target, idx, desc in zip(target_levels, level_indices, descriptions):
            f.write(f"{target:>8.1f} -> {p_levels[idx]:>8.1f} [{idx:>2d}] : {desc}\n")
        
        # Write all level information
        f.write("\n=== Complete Level Information ===\n")
        f.write("Index | Pressure (hPa) | A Coef | B Coef\n")
        f.write("-" * 50 + "\n")
        
        for i in range(len(p_levels)):
            f.write(f"{i:>5d} | {p_levels[i]:>12.3f} | {ds.hyam[i].values:>7.5f} | {ds.hybm[i].values:>7.5f}\n")
            
        # Highlight the selected levels
        f.write("\n=== Selected Level Summary ===\n")
        f.write("These are the indices to use in your analysis:\n")
        f.write("Level indices array: " + str(level_indices) + "\n\n")
        f.write("When selecting data, use:\n")
        f.write("data.isel(lev=level_indices)\n")
        
        # Close dataset
        ds.close()

# File paths
nc_file = "/teamspace/s3_connections/ncar-cesm2-arise-bucket/CESM2-WACCM-SSP245/b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.001/atm/proc/tseries/day_1/b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.001.cam.h1.ACTNL.20650101-20741231.nc"
output_file = "/teamspace/studios/this_studio/pressure-levels.txt"

# Make sure output directory exists
Path(output_file).parent.mkdir(parents=True, exist_ok=True)

# Write the information
write_level_info(nc_file, output_file)

print(f"Level information has been written to: {output_file}")