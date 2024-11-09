import xarray as xr
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict, Tuple
import re
from datetime import datetime
from collections import defaultdict
import dask
import shutil
from dask.distributed import Client, progress, as_completed
from dask.diagnostics import ProgressBar
from tqdm import tqdm
import time
import zarr
from numcodecs import Blosc
import psutil
import os
import cloudpickle  # Ensure cloudpickle is available for serialization

# ============================================
# Step-by-Step Plan:
#
# 1. **Identify Non-Picklable Objects**:
#    - Loggers, Blosc compressor instances, and Path objects.
#
# 2. **Refactor to Ensure Picklability**:
#    - Move `process_variable` outside the class.
#    - Pass only picklable objects (e.g., strings, lists, dicts).
#    - Reconstruct non-picklable objects within the function.
#    - Initialize loggers within the function.
#
# 3. **Update Task Submission**:
#    - Ensure all arguments passed to `process_variable` are picklable.
#    - Use strings for file paths instead of Path objects.
#
# 4. **Handle Logging Appropriately**:
#    - Each worker initializes its own logger to avoid serialization issues.
#
# 5. **Ensure Dask Environment Consistency**:
#    - All necessary libraries are available on worker nodes.
#
# ============================================

# ============================================
# Standalone Function: process_variable
# ============================================

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


def setup_logger(log_file: str) -> logging.Logger:
    """Configure and return a logger."""
    logger = logging.getLogger(log_file)
    logger.setLevel(logging.INFO)

    # Avoid adding multiple handlers to the logger
    if not logger.handlers:
        # Create a formatter that includes timestamp and log level
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )

        # File handler
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # Stream handler (optional, can be removed if not needed)
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)

    return logger

def process_variable(
    var_name: str,
    files: List[str],
    output_dir: str,
    target_levels: List[float],
    compressor_params: Dict,
    log_file: str
) -> xr.DataArray:
    """Standalone function to process a single variable across all its files."""
    # Initialize logger for this function
    logger = setup_logger(log_file)
    logger.info("\n" + "=" * 80)
    logger.info(f"PROCESSING VARIABLE: {var_name}")
    logger.info("=" * 80)

    if not files:
        logger.warning(f"No files found for {var_name}")
        return None

    # Sort files by time period
    files = sorted(files)

    # Create Zarr store path
    zarr_path = os.path.join(output_dir, f'{var_name}.zarr')
    if os.path.exists(zarr_path):
        logger.info(f"Removing existing Zarr store: {zarr_path}")
        shutil.rmtree(zarr_path)

    # Initialize Blosc compressor inside the function
    compressor = Blosc(cname=compressor_params['cname'],
                       clevel=compressor_params['clevel'],
                       shuffle=compressor_params['shuffle'])

    # Initialize timers
    start_time = time.time()

    # Initialize ProgressBar for Dask tasks
    with ProgressBar():
        # Wrap the file processing loop with tqdm for a progress bar
        for i, file in enumerate(tqdm(files, desc=f"Processing {var_name}", unit="file"), 1):
            logger.info(f"\nProcessing file {i}/{len(files)}: {file}")
            try:
                logger.info("Opening dataset with Dask...")
                # Open with Dask - consistent chunking
                ds = xr.open_dataset(file, chunks={'time': 200})  # Adjust chunk size as needed
                logger.info(f"Opened dataset: {file}")
                logger.info(f"Dataset dimensions: {ds[var_name].dims}")
                logger.info(f"Dataset shape: {ds[var_name].shape}")
                logger.info(f"Dataset chunks: {ds[var_name].chunks}")

                # Log data type information
                dtype = ds[var_name].dtype
                logger.info(f"Data type for {var_name}: {dtype}")
                logger.info(f"Data type size in bits: {dtype.itemsize * 8}")

                # Check initial shape
                initial_shape = ds[var_name].shape
                logger.info(f"Initial shape: {initial_shape}")

                # Handle pressure levels if 'lev' dimension exists
                if 'lev' in ds[var_name].dims:
                    logger.info("Variable has 'lev' dimension. Computing pressure levels...")

                    # Get hybrid coordinates
                    required_vars = ['P0', 'hyam', 'hybm']
                    for var in required_vars:
                        if var not in ds:
                            logger.error(f"Missing required variable: {var}")
                            raise KeyError(f"Dataset missing required variable: {var}")

                    # Get reference pressure in hPa
                    P0 = ds.P0.values / 100  # Convert Pa to hPa
                    logger.debug(f"Reference pressure P0: {P0} hPa")

                    # Get hybrid coefficients
                    hyam = ds.hyam.values
                    hybm = ds.hybm.values

                    logger.debug(f"Shape of hybrid coefficients - hyam: {hyam.shape}, hybm: {hybm.shape}")

                    # Calculate approximate pressure levels at midpoints
                    p_levels = P0 * (hyam + hybm)

                    logger.info(f"Calculated {len(p_levels)} pressure levels")
                    logger.debug(f"Pressure levels (hPa): {p_levels}")

                    # Find nearest target levels
                    indices = []
                    for target in target_levels:
                        idx = np.abs(p_levels - target).argmin()
                        indices.append(idx)
                        logger.debug(f"Target pressure level {target} hPa matched with actual level {p_levels[idx]} hPa at index {idx}")

                    # Ensure no duplicate indices
                    unique_indices = sorted(set(indices))
                    logger.info(f"Selected unique pressure level indices: {unique_indices}")

                    logger.info("Selecting pressure levels...")
                    data = ds[var_name].isel(lev=unique_indices)
                    logger.info(f"Shape after level selection: {data.shape}")
                else:
                    logger.info("Variable does not have 'lev' dimension. No level selection needed.")
                    data = ds[var_name]
                    logger.info(f"Shape: {data.shape}")

                # Define chunk dictionary based on presence of 'lev' dimension
                if 'lev' in data.dims:
                    chunk_dict = {
                        'time': 100,   # Adjust as needed
                        'lat': 192,
                        'lon': 288,
                        'lev': len(target_levels)  # Adjust based on selected levels
                    }
                else:
                    chunk_dict = {
                        'time': 100,   # Adjust as needed
                        'lat': 192,
                        'lon': 288
                    }

                logger.info(f"Re-chunking data with chunks: {chunk_dict}")
                data = data.chunk(chunk_dict)
                logger.debug(f"New chunks: {data.chunks}")

                # Add memory tracking
                current_mem = psutil.Process(os.getpid()).memory_info().rss / 1e9
                logger.info(f"Current memory usage: {current_mem:.2f} GB")

                # Estimate size of data in GB
                data_size_gb = data.nbytes / (1024 ** 3)
                logger.info(f"Size of data to write: {data_size_gb:.2f} GB")

                # Define encoding with compressor
                encoding = {var_name: {'compressor': compressor}}

                # Write to Zarr incrementally with encoding
                logger.info(f"Appending data to Zarr store: {zarr_path}")
                # For the first file, initialize the Zarr store
                if i == 1:
                    logger.info(f"Initializing Zarr store for {var_name} with mode='w'")
                    data.to_zarr(
                        zarr_path,
                        mode='w',
                        encoding=encoding,
                        compute=True,
                        safe_chunks=False  # Disable safe_chunks to allow variable chunk sizes
                    )
                    logger.info(f"Initialized Zarr store for {var_name}")
                else:
                    logger.info(f"Appending to Zarr store for {var_name} with mode='a' and append_dim='time'")
                    data.to_zarr(
                        zarr_path,
                        mode='a',
                        append_dim='time',
                        compute=True,
                        safe_chunks=False  # Disable safe_chunks for appending
                    )
                    logger.info(f"Appended data to Zarr store for {var_name}")

                logger.info(f"Successfully processed and written file {file} for variable {var_name}")

            except Exception as e:
                logger.error(f"Error processing {file} for variable {var_name}: {str(e)}")
                raise

    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    logger.info(f"Finished processing variable {var_name} in {elapsed_time / 60:.2f} minutes.")

    # After all files are processed, open the Zarr store
    logger.info("Opening concatenated Zarr store...")
    try:
        final_data = xr.open_zarr(zarr_path, consolidated=True, chunks={'time': 200})  # Adjusted chunk size
        logger.info(f"Opened Zarr store for {var_name}")
        logger.info(f"Final shape for {var_name}: {final_data[var_name].shape}")
        logger.info(f"Chunks for {var_name}: {final_data[var_name].chunks}")
    except Exception as e:
        logger.error(f"Error opening Zarr store for {var_name}: {str(e)}")
        raise

    return final_data[var_name]

# ============================================
# ARISEProcessor Class
# ============================================

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
        self.base_dir = base_dir  # Keep as string for picklability
        self.output_dir = output_dir  # Keep as string for picklability

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        # Setup logging for the main process
        log_file = os.path.join(self.output_dir, f'arise_processing_main_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        self.logger = setup_logger(log_file)

        # Log system resources
        self.logger.info(log_system_resources())

        # Target pressure levels from paper
        self.target_levels = [0.6, 10, 20, 30, 50, 70, 120, 200, 400, 600, 800, 1000]

        # Variables to process
        self.variables = ['T', 'Q', 'Uzm', 'Vzm', 'PS', 'TS', 'SO2', 
                         'BURDENSO4dn', 'SFso4_a1', 'SFso4_a2', 'SOLIN']

        # Dictionary to store processed data
        self.processed_data = {}

        # Compression settings parameters (as dict for picklability)
        self.compressor_params = {
            'cname': compressor_name,
            'clevel': compression_level,
            'shuffle': shuffle
        }

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

        for file in Path(self.base_dir).glob('*.nc'):
            match = pattern.match(str(file))
            if match:
                freq, var_name, time_period = match.groups()
                file_size = file.stat().st_size / (1024 * 1024)  # Size in MB

                var_info[var_name]['files'].append(str(file))
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

    def find_nc_files(self) -> Dict[str, List[str]]:
        """Find all relevant .nc files and organize by variable"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("FINDING AND ORGANIZING NC FILES")
        self.logger.info("=" * 80)

        files_by_var = {var: [] for var in self.variables}

        # Pattern to match h1 files only
        pattern = re.compile(r'.*\.cam\.h1\.(\w+)\.\d{8}-\d{8}\.nc$')

        self.logger.info("Scanning for h1 files...")

        for file in Path(self.base_dir).glob('*.nc'):
            match = pattern.match(str(file))
            if match:
                var_name = match.group(1)
                if var_name in self.variables:
                    files_by_var[var_name].append(str(file))
                    self.logger.debug(f"Found {var_name} file: {file.name}")

        # Log summary
        self.logger.info("\nFILE SUMMARY:")
        self.logger.info("-" * 50)
        for var, files in files_by_var.items():
            self.logger.info(f"{var:<20}: {len(files)} files")

        self.logger.info("=" * 80 + "\n")
        return files_by_var

    def create_datacube(self) -> xr.Dataset:
        """Create final datacube combining all variables"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("CREATING FINAL DATACUBE")
        self.logger.info("=" * 80)

        # Analyze available variables first
        var_info = self.analyze_available_variables()

        # Find and process files
        self.logger.info("Finding NC files...")
        files_by_var = self.find_nc_files()

        # Prepare a list to hold futures
        futures = []

        # Submit processing tasks for each variable
        for var in self.variables:
            if files_by_var[var]:
                # Create a unique log file for each variable
                log_file = os.path.join(self.output_dir, f'process_{var}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

                try:
                    # Submit the process_variable task to Dask
                    future = self.client.submit(
                        process_variable,
                        var,
                        files_by_var[var],
                        self.output_dir,
                        self.target_levels,
                        self.compressor_params,
                        log_file
                    )
                    futures.append((var, future))
                    self.logger.info(f"Submitted processing task for variable {var}")
                except Exception as e:
                    self.logger.error(f"Error submitting task for {var}: {str(e)}")
                    raise
            else:
                self.logger.warning(f"No files found for {var}")

        # Collect results as they complete
        self.logger.info("\nWaiting for all variable processing tasks to complete...")
        for var, future in futures:
            try:
                processed_data = future.result()
                if processed_data is not None:
                    # Store processed data
                    self.processed_data[var] = processed_data
                    self.logger.info(f"Stored processed data for variable {var}")
                else:
                    self.logger.warning(f"No processed data for variable {var}")
            except Exception as e:
                self.logger.error(f"Error processing {var}: {str(e)}")
                raise

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
            output_file = os.path.join(self.output_dir, 'arise_processed.zarr')
            if os.path.exists(output_file):
                self.logger.info(f"Removing existing final Zarr store: {output_file}")
                shutil.rmtree(output_file)
            self.logger.info(f"\nSaving final datacube to {output_file}...")

            # Define encoding for all variables in the Dataset
            # Reconstruct compressor objects for encoding
            compressor_dict = {var: {'compressor': Blosc(**self.compressor_params)} for var in ds.data_vars}

            # Initialize ProgressBar for writing final dataset
            with ProgressBar():
                ds.to_zarr(
                    output_file,
                    mode='w',
                    consolidated=True,
                    encoding=compressor_dict,
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

# ============================================
# Main Execution
# ============================================

if __name__ == "__main__":
    base_dir = "/teamspace/s3_connections/ncar-cesm2-arise-bucket/ARISE-SAI-1.5/b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.001/atm/proc/tseries/day_1"
    output_dir = "/teamspace/studios/this_studio/arise_processed"

    processor = ARISEProcessor(base_dir, output_dir)
    processor.run()
