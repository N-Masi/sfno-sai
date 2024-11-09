import os
import logging
from datetime import datetime
from pathlib import Path
import stat
import re
from collections import defaultdict

def extract_variable_info(filename: str) -> tuple:
    """
    Extract variable name and time period from filename.
    Handles multiple time formats and experiment names.
    """
    # More flexible pattern to match any path structure
    # Matches both formats: 
    # - b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.001.cam.h0.ADSNOW.201501-206412.nc
    # - b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.001.cam.h0.ADSNOW.201501-206412.nc
    patterns = [
        r'.*\.cam\.[h]\d+\.(\w+)\.(\d{6}-\d{6})\.nc$',  # monthly
        r'.*\.cam\.[h]\d+\.(\w+)\.(\d{8}-\d{8})\.nc$',  # daily
        r'.*\.cam\.[h]\d+\.(\w+)\.(\d{4}-\d{4})\.nc$'   # other formats
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            var_name, timeperiod = match.groups()
            # Extract frequency (h0, h1, etc.)
            freq_match = re.search(r'\.([h]\d+)\.', filename)
            freq = freq_match.group(1) if freq_match else 'unknown'
            return freq, var_name, timeperiod
            
    return None

def get_time_resolution(path: str) -> str:
    """Extract time resolution from path."""
    if 'month' in path:
        return 'monthly'
    elif 'day' in path:
        return 'daily'
    elif 'hour' in path:
        return 'hourly'
    else:
        return 'unknown'

def get_experiment_name(path: str) -> str:
    """Extract experiment name from path."""
    # Match different possible experiment patterns
    patterns = [
        r'ARISE-SAI-1\.5(?:-\d+)?',  # Matches ARISE-SAI-1.5 or ARISE-SAI-1.5-2045
        r'CESM2-WACCM-SSP245',
        r'SSP245-TSMLT-GAUSS-(?:DEFAULT|DELAYED-\d+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, path)
        if match:
            return match.group(0)
    return "unknown_experiment"

def list_directory_files(directory_path: str, 
                        output_file: str = None) -> None:
    """List all files with enhanced variable summary."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    try:
        dir_path = Path(directory_path)
        
        # Generate output filename if not provided
        if output_file is None:
            experiment = get_experiment_name(str(dir_path))
            time_res = get_time_resolution(str(dir_path))
            output_file = f'/teamspace/studios/this_studio/arise-vars_{experiment}_{time_res}.txt'
            
        output_path = Path(output_file)
        
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
            
        # Organize files
        var_files = defaultdict(list)
        time_periods = defaultdict(set)
        frequencies = defaultdict(set)
        
        logger.info(f"Reading directory: {directory_path}")
        files = sorted([f.name for f in dir_path.iterdir() if f.is_file() and f.name.endswith('.nc')])
        
        # Process files
        for file in files:
            info = extract_variable_info(file)
            if info:
                freq, var, timeperiod = info
                var_files[var].append(file)
                time_periods[var].add(timeperiod)
                frequencies[var].add(freq)
        
        logger.info(f"Found {len(var_files)} unique variables")
        
        # Write to output file
        with open(output_path, 'w') as f:
            # Write header
            f.write(f"# ARISE Variable Summary\n")
            f.write(f"# Generated on: {datetime.now()}\n")
            f.write(f"# Source: {directory_path}\n")
            f.write(f"# Experiment: {get_experiment_name(str(dir_path))}\n")
            f.write(f"# Time Resolution: {get_time_resolution(str(dir_path))}\n")
            f.write(f"# Total variables: {len(var_files)}\n")
            f.write(f"# Total files: {len(files)}\n\n")
            
            # Write variable summary
            f.write("=" * 120 + "\n")
            f.write(f"{'Variable':<20} {'Frequency':<12} {'# Files':<10} {'Time Range':<50}\n")
            f.write("=" * 120 + "\n")
            
            for var in sorted(var_files.keys()):
                time_range = sorted(time_periods[var])
                freq_list = sorted(frequencies[var])
                freq_str = ','.join(freq_list)
                f.write(f"{var:<20} {freq_str:<12} {len(var_files[var]):<10} "
                       f"{time_range[0]} → {time_range[-1]}\n")
            
            f.write("\n" + "=" * 120 + "\n\n")
            
            # Write detailed file listings by variable
            f.write("Detailed File Listings by Variable:\n\n")
            for var in sorted(var_files.keys()):
                f.write(f"\n### {var} ###\n")
                for file in sorted(var_files[var]):
                    f.write(f"{file}\n")
                f.write("\n")
        
        # Set permissions
        output_path.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IWGRP | stat.S_IROTH)
        
        logger.info(f"Successfully wrote summary to {output_file}")
        logger.info(f"Found variables: {', '.join(sorted(var_files.keys()))}")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    # Example paths for different experiments
    #        "/teamspace/s3_connections/ncar-cesm2-arise-bucket/ARISE-SAI-1.5-2045/b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DELAYED-2045.001/atm/proc/tseries/hour_3",
    #         "/teamspace/s3_connections/ncar-cesm2-arise-bucket/CESM2-WACCM-SSP245/b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.001/atm/proc/tseries/month_1"

    dir_paths = [
        "/teamspace/s3_connections/ncar-cesm2-arise-bucket/ARISE-SAI-1.5/b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.001/atm/proc/tseries/day_1",
    ]
    
    for dir_path in dir_paths:
        list_directory_files(dir_path)