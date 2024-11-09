import xarray as xr
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict
import re
from datetime import datetime
from collections import defaultdict
import dask
import shutil
from dask.distributed import Client, progress
from dask.diagnostics import ProgressBar
from tqdm import tqdm
import time
import zarr
from numcodecs import Blosc
import psutil
import os

# ============================================
# Final Datacube Structure:
# The final datacube combines all processed variables into a single xarray.Dataset.
# - Common dimensions: 'time', 'lat', 'lon'
# - Variables with vertical levels will include an additional 'lev' dimension.
#   For example:
#     - 4D Variable: ('time', 'lev', 'lat', 'lon')
#     - 3D Variable: ('time', 'lat', 'lon')
# This conditional structure ensures that variables are appropriately integrated
# based on their dimensionality.
# ============================================

# Add system resource logging
def log_system_resources():
    # CPU Information
    cpu_count = psutil.cpu_count(logical=False)  # Physical cores
    cpu_count_logical = psutil.cpu_count(logical=True)  # Logical cores
    cpu_freq = psutil.cpu_freq()
    
    # Memory Information
    memory = psutil.virtual_memory()
    memory_gb = memory.total / (1024 ** 3)  # Convert to GB
    
    # Create formatted string
    resource_info = [
        "=" * 40,
        "SYSTEM RESOURCES",
        "=" * 40,
        f"CPU Cores (Physical): {cpu_count}",
        f"CPU Cores (Logical): {cpu_count_logical}",
        f"CPU Frequency: Current={cpu_freq.current:.2f}MHz, Min={cpu_freq.min:.2f}MHz, Max={cpu_freq.max:.2f}MHz",
        f"Total RAM: {memory_gb:.2f} GB",
        f"Available RAM: {(memory.available / (1024 ** 3)):.2f} GB",
        f"RAM Usage: {memory.percent}%",
        "=" * 40,
    ]
    
    return "\n".join(resource_info)

class ARISEProcessor:
    def __init__(
        self,
        base_dir: str,
        output_dir: str,
        dask_workers: int = 8,
        threads_per_worker: int = 4,
        memory_limit: str = '24GB',
        compressor_name: str = 'zstd',
        compression_level: int = 3,
        shuffle: int = Blosc.SHUFFLE
    ):
        """Initialize ARISE data processor"""
        self.base_dir = Path(base_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Log system resources
        self.logger.info(log_system_resources())
        
        # Target pressure levels from paper
        self.target_levels = np.array([0.6, 10, 20, 30, 50, 70, 120, 200, 400, 600, 800, 1000])
        
        # Variables to process
        self.variables = ['T', 'Q', 'Uzm', 'Vzm', 'PS', 'TS', 'SO2', 
                         'BURDENSO4dn', 'SFso4_a1', 'SFso4_a2', 'SOLIN']
        
        # Dictionary to store processed data
        self.processed_data = {}
        
        # Compression settings
        self.compressor = Blosc(cname=compressor_name, clevel=compression_level, shuffle=shuffle)
        # Since we're processing one variable at a time, we don't need a dict of encodings
        # We'll create the encoding dict inside the process_variable method
        
        # Add these Dask configurations
        dask.config.set({
            'distributed.worker.memory.target': 0.8,  # 80% memory threshold
            'distributed.worker.memory.spill': 0.85,  # 85% memory spill to disk
            'distributed.worker.memory.pause': 0.90,  # 90% memory pause worker
            'distributed.worker.memory.terminate': 0.95,  # 95% memory terminate worker
            'distributed.comm.compression': 'zstd',
            'distributed.scheduler.work-stealing': True,  # Enable work stealing between workers
        })

        # Initialize Dask client with modified settings
        self.client = Client(
            n_workers=dask_workers,
            threads_per_worker=threads_per_worker,
            memory_limit=memory_limit,
            processes=True,  # Ensure we're using processes not threads
            silence_logs=logging.INFO
        )
        self.logger.info(f"Dask client initialized with {dask_workers} workers, {threads_per_worker} threads per worker, memory limit per worker: {memory_limit}")
        self.logger.info(f"Dask dashboard available at: {self.client.dashboard_link}")
    
    def get_pressure_levels(self, ds: xr.Dataset) -> np.ndarray:
        """
        Calculate pressure levels from hybrid coordinates
        
        Parameters:
            ds: xarray Dataset containing hybrid level coordinates
            
        Returns:
            ndarray: Pressure levels in hPa
        """
        self.logger.info("Calculating pressure levels...")
        
        try:
            # Check if required variables exist
            required_vars = ['P0', 'hyam', 'hybm']
            for var in required_vars:
                if var not in ds:
                    self.logger.error(f"Missing required variable: {var}")
                    raise KeyError(f"Dataset missing required variable: {var}")
            
            # Get reference pressure in hPa
            P0 = ds.P0 / 100  # Convert Pa to hPa
            self.logger.debug(f"Reference pressure P0: {P0.values} hPa")
            
            # Get hybrid coefficients
            hyam = ds.hyam
            hybm = ds.hybm
            
            self.logger.debug(f"Shape of hybrid coefficients - hyam: {hyam.shape}, hybm: {hybm.shape}")
            
            # Calculate approximate pressure levels at midpoints
            p_levels = P0 * (hyam + hybm)
            
            self.logger.info(f"Calculated {len(p_levels)} pressure levels")
            self.logger.debug(f"Pressure levels (hPa): {p_levels.values}")
            
            return p_levels.values
            
        except Exception as e:
            self.logger.error(f"Error calculating pressure levels: {str(e)}")
            raise

    def find_nearest_levels(self, p_levels):
        """
        Find indices of nearest pressure levels to targets

        Parameters:
            p_levels: array of actual pressure levels
        Returns:
            sorted list of unique indices of nearest levels
        """
        
        indices = []
        for target in self.target_levels:
            idx = np.abs(p_levels - target).argmin()
            indices.append(idx)
            self.logger.debug(f"Target pressure level {target} hPa matched with actual level {p_levels[idx]} hPa at index {idx}")
        
        # Ensure no duplicate indices
        unique_indices = sorted(set(indices))
        self.logger.info(f"Selected unique pressure level indices: {unique_indices}")
        return unique_indices

    def setup_logging(self):
        """Configure logging with more detailed formatting"""
        log_file = self.output_dir / f'arise_processing_{datetime.now():%Y%m%d_%H%M%S}.log'
        
        # Create a formatter that includes thread name
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(threadName)s - %(message)s'
        )
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        
        # Stream handler
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        
        # Configure logger
        logging.basicConfig(
            level=logging.INFO,
            handlers=[file_handler, stream_handler]
        )
        
        self.logger = logging.getLogger(__name__)
        
        # Log initial configuration
        self.logger.info("=" * 80)
        self.logger.info("ARISE Data Processing Started")
        self.logger.info(f"Base directory: {self.base_dir}")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Log file: {log_file}")
        self.logger.info("=" * 80)
    
    def analyze_available_variables(self) -> Dict[str, Dict]:
        """
        Analyze all available variables in the directory and their characteristics
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("ANALYZING AVAILABLE VARIABLES")
        self.logger.info("=" * 80)

        var_info = defaultdict(lambda: {
            'files': [],
            'total_size': 0,
            'time_periods': set(),
            'frequencies': set()
        })

        # Pattern to match any netCDF file
        pattern = re.compile(r'.*\.cam\.(h\d+)\.(\w+)\.(\d{8}-\d{8})\.nc$')
        
        total_files = 0
        total_size = 0

        self.logger.info("Scanning directory for .nc files...")
        
        for file in self.base_dir.glob('*.nc'):
            match = pattern.match(str(file))
            if match:
                freq, var_name, time_period = match.groups()
                file_size = file.stat().st_size / (1024 * 1024)  # Size in MB
                
                var_info[var_name]['files'].append(file.name)
                var_info[var_name]['total_size'] += file_size
                var_info[var_name]['time_periods'].add(time_period)
                var_info[var_name]['frequencies'].add(freq)
                
                total_files += 1
                total_size += file_size

        # Print summary
        self.logger.info("\nVARIABLE SUMMARY:")
        self.logger.info("-" * 80)
        self.logger.info(f"{'Variable':<20} {'Files':<8} {'Size (MB)':<12} {'Frequencies':<15} {'Time Periods'}")
        self.logger.info("-" * 80)

        for var_name, info in sorted(var_info.items()):
            self.logger.info(
                f"{var_name:<20} "
                f"{len(info['files']):<8} "
                f"{info['total_size']:,.2f} MB  "
                f"{','.join(info['frequencies']):<15} "
                f"{len(info['time_periods'])}"
            )

        self.logger.info("-" * 80)
        self.logger.info(f"Total variables: {len(var_info)}")
        self.logger.info(f"Total files: {total_files}")
        self.logger.info(f"Total size: {total_size:,.2f} MB")
        
        # Check for missing required variables
        missing_vars = set(self.variables) - set(var_info.keys())
        if missing_vars:
            self.logger.warning("\nMISSING REQUIRED VARIABLES:")
            for var in missing_vars:
                self.logger.warning(f"- {var}")
        
        self.logger.info("=" * 80 + "\n")
        
        return var_info

    def find_nc_files(self) -> Dict[str, List[Path]]:
        """Find all relevant .nc files and organize by variable"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("FINDING AND ORGANIZING NC FILES")
        self.logger.info("=" * 80)
        
        files_by_var = {var: [] for var in self.variables}
        
        # Pattern to match h1 files only
        pattern = re.compile(r'.*\.cam\.h1\.(\w+)\.\d{8}-\d{8}\.nc$')
        
        self.logger.info("Scanning for h1 files...")
        
        for file in self.base_dir.glob('*.nc'):
            match = pattern.match(str(file))
            if match:
                var_name = match.group(1)
                if var_name in self.variables:
                    files_by_var[var_name].append(file)
                    self.logger.debug(f"Found {var_name} file: {file.name}")
        
        # Log summary
        self.logger.info("\nFILE SUMMARY:")
        self.logger.info("-" * 50)
        for var, files in files_by_var.items():
            self.logger.info(f"{var:<20}: {len(files)} files")
        
        self.logger.info("=" * 80 + "\n")
        return files_by_var

    def process_variable(self, var_name: str, files: List[Path]) -> xr.DataArray:
        """Process a single variable across all its files using Dask for out-of-memory computation"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info(f"PROCESSING VARIABLE: {var_name}")
        self.logger.info("=" * 80)

        if not files:
            self.logger.warning(f"No files found for {var_name}")
            return None

        # Sort files by time period
        files = sorted(files)

        # Create Zarr store path
        zarr_path = self.output_dir / f'{var_name}.zarr'
        if zarr_path.exists():
            self.logger.info(f"Removing existing Zarr store: {zarr_path}")
            shutil.rmtree(zarr_path)

        # Initialize timers
        start_time = time.time()

        # Initialize ProgressBar for Dask tasks
        with ProgressBar():
            # Wrap the file processing loop with tqdm for a progress bar
            for i, file in enumerate(tqdm(files, desc=f"Processing {var_name}", unit="file"), 1):
                self.logger.info(f"\nProcessing file {i}/{len(files)}: {file.name}")

                try:
                    self.logger.info("Opening dataset with Dask...")
                    # Open with Dask - consistent chunking
                    ds = xr.open_dataset(file, chunks={'time': 200})  # Increased chunk size
                    self.logger.info(f"Opened dataset: {file.name}")
                    self.logger.info(f"Dataset dimensions: {ds[var_name].dims}")
                    self.logger.info(f"Dataset shape: {ds[var_name].shape}")
                    self.logger.info(f"Dataset chunks: {ds[var_name].chunks}")

                    # Log data type information
                    dtype = ds[var_name].dtype
                    self.logger.info(f"Data type for {var_name}: {dtype}")
                    self.logger.info(f"Data type size in bits: {dtype.itemsize * 8}")

                    # Check initial shape
                    initial_shape = ds[var_name].shape
                    self.logger.info(f"Initial shape: {initial_shape}")

                    # Handle pressure levels if 'lev' dimension exists
                    if 'lev' in ds[var_name].dims:
                        self.logger.info("Variable has 'lev' dimension. Computing pressure levels...")
                        p_levels = self.get_pressure_levels(ds)
                        level_indices = self.find_nearest_levels(p_levels)
                        self.logger.info("Selecting pressure levels...")
                        data = ds[var_name].isel(lev=level_indices)
                        self.logger.info(f"Shape after level selection: {data.shape}")
                    else:
                        self.logger.info("Variable does not have 'lev' dimension. No level selection needed.")
                        data = ds[var_name]
                        self.logger.info(f"Shape: {data.shape}")

                    # Define chunk dictionary based on presence of 'lev' dimension
                    if 'lev' in data.dims:
                        chunk_dict = {
                            'time': 100,   # Adjust as needed
                            'lat': 192,
                            'lon': 288,
                            'lev': 12
                        }
                    else:
                        chunk_dict = {
                            'time': 100,   # Adjust as needed
                            'lat': 192,
                            'lon': 288
                        }

                    self.logger.info(f"Re-chunking data with chunks: {chunk_dict}")
                    data = data.chunk(chunk_dict)
                    self.logger.debug(f"New chunks: {data.chunks}")

                    # Add memory tracking
                    current_mem = psutil.Process(os.getpid()).memory_info().rss / 1e9
                    self.logger.info(f"Current memory usage: {current_mem:.2f} GB")

                    # Estimate size of data in GB
                    data_size_gb = data.nbytes / (1024 ** 3)
                    self.logger.info(f"Size of data to write: {data_size_gb:.2f} GB")

                    # Define encoding with compressor
                    encoding = {var_name: {'compressor': self.compressor}}

                    # Write to Zarr incrementally with encoding
                    self.logger.info(f"Appending data to Zarr store: {zarr_path}")
                    # For the first file, initialize the Zarr store
                    if i == 1:
                        self.logger.info(f"Initializing Zarr store for {var_name} with mode='w'")
                        data.to_zarr(
                            zarr_path,
                            mode='w',
                            encoding=encoding,
                            compute=True,
                            safe_chunks=False  # Disable safe_chunks to allow variable chunk sizes
                        )
                        self.logger.info(f"Initialized Zarr store for {var_name}")
                    else:
                        self.logger.info(f"Appending to Zarr store for {var_name} with mode='a' and append_dim='time'")
                        data.to_zarr(
                            zarr_path,
                            mode='a',
                            append_dim='time',
                            compute=True,
                            safe_chunks=False  # Disable safe_chunks for appending
                        )
                        self.logger.info(f"Appended data to Zarr store for {var_name}")

                    self.logger.info(f"Successfully processed and written file {file.name} for variable {var_name}")

                except Exception as e:
                    self.logger.error(f"Error processing {file.name} for variable {var_name}: {str(e)}")
                    raise

        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        self.logger.info(f"Finished processing variable {var_name} in {elapsed_time / 60:.2f} minutes.")

        # After all files are processed, open the Zarr store
        self.logger.info("Opening concatenated Zarr store...")
        try:
            final_data = xr.open_zarr(zarr_path, consolidated=True, chunks={'time': 200})  # Adjusted chunk size
            self.logger.info(f"Opened Zarr store for {var_name}")
            self.logger.info(f"Final shape for {var_name}: {final_data.shape}")
            self.logger.info(f"Chunks: {final_data.chunks}")
        except Exception as e:
            self.logger.error(f"Error opening Zarr store for {var_name}: {str(e)}")
            raise

        return final_data


    def create_datacube(self) -> xr.Dataset:
            """Create final datacube combining all variables"""
            self.logger.info("\n" + "=" * 80)
            self.logger.info("CREATING FINAL DATACUBE")
            self.logger.info("=" * 80)
            
            # Analyze available variables first
            self.analyze_available_variables()
            
            # Find and process files
            self.logger.info("Finding NC files...")
            files_by_var = self.find_nc_files()
            
            # Process each variable
            for var in self.variables:
                self.logger.info(f"\n{'=' * 40}\nProcessing Variable: {var}\n{'=' * 40}")
                if files_by_var[var]:
                    try:
                        # Process the variable and retrieve the processed DataArray
                        self.logger.info(f"Processing variable {var} with {len(files_by_var[var])} files...")
                        processed_data = self.process_variable(var, files_by_var[var])
                        
                        if processed_data is not None:
                            # Store processed data
                            self.processed_data[var] = processed_data
                            self.logger.info(f"Stored processed data for variable {var}")
                        else:
                            self.logger.warning(f"No processed data for variable {var}")
                        
                    except Exception as e:
                        self.logger.error(f"Error processing {var}: {str(e)}")
                        raise
                else:
                    self.logger.warning(f"No files found for {var}")
            
            # Combine into final dataset
            self.logger.info("\nCombining variables into final dataset...")
            ds = xr.Dataset(self.processed_data)
            
            # Log final dimensions
            self.logger.info("\nFINAL DATACUBE DIMENSIONS:")
            self.logger.info("-" * 50)
            for dim, size in ds.dims.items():
                self.logger.info(f"{dim:<20}: {size:,}")
            
            # Log data variables and their shapes
            self.logger.info("\nFINAL DATACUBE VARIABLES:")
            self.logger.info("-" * 50)
            for var in ds.data_vars:
                self.logger.info(f"{var:<20}: shape={ds[var].shape}, chunks={ds[var].chunks}")
            
            self.logger.info("=" * 80 + "\n")
            return ds


    def run(self):
        """Main processing pipeline"""
        try:
            self.logger.info("=" * 80)
            self.logger.info("STARTING ARISE DATA PROCESSING PIPELINE")
            self.logger.info("=" * 80 + "\n")
            
            # Create datacube
            self.logger.info("Creating datacube...")
            ds = self.create_datacube()
            
            # Save to Zarr
            output_file = self.output_dir / 'arise_processed.zarr'
            if output_file.exists():
                self.logger.info(f"Removing existing final Zarr store: {output_file}")
                shutil.rmtree(output_file)
            self.logger.info(f"\nSaving final datacube to {output_file}...")
            
            # Define encoding for all variables in the Dataset
            encoding = {var: {'compressor': self.compressor} for var in ds.data_vars}
            
            # Initialize ProgressBar for writing final dataset
            with ProgressBar():
                ds.to_zarr(
                    output_file,
                    mode='w',
                    consolidated=True,
                    encoding=encoding,
                    compute=True,
                    safe_chunks=False  # Disable safe_chunks for the final write
                )
            
            self.logger.info("\nProcessing complete!")
            self.logger.info("=" * 80)
        
        except Exception as e:
            self.logger.error(f"Error during processing: {str(e)}", exc_info=True)
            raise
        finally:
            self.logger.info("Closing Dask client...")
            self.client.close()
            self.logger.info("Dask client closed.")
            
if __name__ == "__main__":
    base_dir = "/teamspace/s3_connections/ncar-cesm2-arise-bucket/ARISE-SAI-1.5/b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.001/atm/proc/tseries/day_1"
    output_dir = "/teamspace/studios/this_studio/arise_processed"
    
    processor = ARISEProcessor(base_dir, output_dir)
    processor.run()
