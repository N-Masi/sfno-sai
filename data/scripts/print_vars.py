import s3fs
import xarray as xr
from pathlib import Path
from collections import defaultdict
import pandas as pd

def explore_arise_bucket(bucket_name="ncar-cesm2-arise", experiment="ARISE-SAI-1.5"):
    # Initialize S3 filesystem
    fs = s3fs.S3FileSystem(anon=True)
    
    # Get all files in the monthly time series directory
    base_path = f"{bucket_name}/{experiment}/b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.001/atm/proc/tseries/month_1/"
    files = fs.ls(base_path)
    
    # Filter for .nc files
    nc_files = [f for f in files if f.endswith('.nc')]
    
    # Dictionary to store variable info
    var_info = defaultdict(dict)
    
    print(f"\n{'='*80}")
    print(f"Exploring ARISE Dataset: {experiment}")
    print(f"{'='*80}\n")
    print(f"Number of .nc files found: {len(nc_files)}\n")
    
    # Process each NetCDF file
    for file_path in nc_files:
        try:
            with fs.open(file_path) as f:
                ds = xr.open_dataset(f)
                
                # Get the variable name from the filename
                file_name = Path(file_path).name
                var_name = file_name.split('.')[-2]  # Extract variable name from filename
                
                # Store information
                for var in ds.data_vars:
                    var_info[var_name] = {
                        'dimensions': list(ds[var].dims),
                        'shape': list(ds[var].shape),
                        'time_range': f"{ds.time.min().values} to {ds.time.max().values}",
                        'units': ds[var].attrs.get('units', 'Not specified'),
                        'long_name': ds[var].attrs.get('long_name', 'Not specified')
                    }
                
                print(f"\nVariable: {var_name}")
                print("-" * 40)
                print(f"Dimensions: {var_info[var_name]['dimensions']}")
                print(f"Shape: {var_info[var_name]['shape']}")
                print(f"Time Range: {var_info[var_name]['time_range']}")
                print(f"Units: {var_info[var_name]['units']}")
                print(f"Long Name: {var_info[var_name]['long_name']}")
                
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
    
    # Display total number of unique variables found
    print(f"\nTotal unique variables found: {len(var_info)}")
    
    return var_info

if __name__ == "__main__":
    var_info = explore_arise_bucket()
    
    # Optional: Create a summary DataFrame
    print("\nSummary of Available Variables:")
    print("=" * 80)
    
    summary_data = []
    for var_name, info in var_info.items():
        summary_data.append({
            'Variable': var_name,
            'Dimensions': ', '.join(info['dimensions']),
            'Shape': str(info['shape']),
            'Time Range': info['time_range']
        })
    
    if summary_data:
        df = pd.DataFrame(summary_data)
        print(df.to_string())
