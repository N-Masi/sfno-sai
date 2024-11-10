import xarray as xr
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict, Tuple
import re
from datetime import datetime, timedelta
from collections import defaultdict
import shutil
from tqdm import tqdm
import time
import zarr
from numcodecs import Blosc
import psutil
import os
import torch  # Assuming PyTorch is used for tensor operations
import concurrent.futures  # Added for parallel processing

# ============================================
# Enhanced ARISEProcessor with Parallel Variable Processing
# ============================================

def log_system_resources() -> str:
    """Log current system resources for monitoring."""
    cpu_count = psutil.cpu_count(logical=False)  # Physical cores
    cpu_count_logical = psutil.cpu_count(logical=True)  # Logical cores
    cpu_freq = psutil.cpu_freq()

    memory = psutil.virtual_memory()
    memory_gb = memory.total / (1024 ** 3)  # Convert to GB

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
        compressor_name: str = 'zstd',
        compression_level: int = 3,
        shuffle: int = Blosc.SHUFFLE
    ):
        """
        Initialize the ARISE data processor with batching capabilities.

        Parameters:
            base_dir (str): Directory containing the .nc files.
            output_dir (str): Directory to store processed data and logs.
            compressor_name (str): Compressor for Zarr encoding.
            compression_level (int): Compression level for Zarr encoding.
            shuffle (int): Shuffle parameter for Zarr encoding.
        """
        self.base_dir = Path(base_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.setup_logging()

        # Log system resources
        # self.logger.info(log_system_resources())

        # Target pressure levels from paper
        self.target_levels = np.array([0.6, 10, 20, 30, 50, 70, 120, 200, 400, 600, 800, 1000])

        # Variables to process
        self.variables = ['T', 'Q', 'PS', 'TS', 'SO2', 
                         'BURDENSO4dn', 'SFso4_a1', 'SFso4_a2', 'SOLIN']

        # Scan and map files
        self.files_by_var = self.find_nc_files()
        self.day_to_file_map = self.build_day_to_file_map()

        # Initialize batch parameters
        self.total_days = self.calculate_total_days()
        # self.logger.info(f"Total days available for processing: {self.total_days}")

        # Compression settings
        self.compressor = Blosc(cname=compressor_name, clevel=compression_level, shuffle=shuffle)

    def setup_logging(self):
        """Configure logging with detailed formatting."""
        log_file = self.output_dir / f'arise_processing_{datetime.now():%Y%m%d_%H%M%S}.log'

        # Create a formatter that includes thread name and time
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
        # logging.basicConfig(
        #     level=logging.INFO,
        #     handlers=[file_handler, stream_handler]
        # )

        # self.logger = logging.getLogger(__name__)

        # Log initial configuration
        # self.logger.info("=" * 80)
        # self.logger.info("ARISE Data Processing Initialized")
        # self.logger.info(f"Base directory: {self.base_dir}")
        # self.logger.info(f"Output directory: {self.output_dir}")
        # self.logger.info(f"Log file: {log_file}")
        # self.logger.info("=" * 80)

    def find_nc_files(self) -> Dict[str, List[Path]]:
        """Find all relevant .nc files and organize by variable"""
        # self.logger.info("\n" + "=" * 80)
        # self.logger.info("FINDING AND ORGANIZING NC FILES")
        # self.logger.info("=" * 80)

        files_by_var = {var: [] for var in self.variables}

        # Pattern to match h1 files only (daily data)
        pattern = re.compile(r'.*\.cam\.h1\.(\w+)\.\d{8}-\d{8}\.nc$')

        # self.logger.info("Scanning for h1 files...")

        for file in self.base_dir.glob('*.nc'):
            match = pattern.match(str(file))
            if match:
                var_name = match.group(1)
                if var_name in self.variables:
                    files_by_var[var_name].append(file)
                    # self.logger.debug(f"Found {var_name} file: {file.name}")

        # Log summary
        # self.logger.info("\nFILE SUMMARY:")
        # self.logger.info("-" * 50)
        # for var, files in files_by_var.items():
            # self.logger.info(f"{var:<20}: {len(files)} files")
        # self.logger.info("=" * 80 + "\n")
        return files_by_var

    def build_day_to_file_map(self) -> Dict[str, Dict]:
        """
        Build a mapping from each day to its corresponding file and variables.

        Returns:
            Dict[str, Dict]: Mapping from day (YYYYMMDD) to variable-specific files.
        """
        # self.logger.info("BUILDING DAY TO FILE MAP")
        day_to_file = {}

        for var, files in self.files_by_var.items():
            for file in files:
                # Extract time range from filename
                match = re.search(r'\.(\d{8})-(\d{8})\.nc$', file.name)
                if match:
                    start_date_str, end_date_str = match.groups()
                    start_date = datetime.strptime(start_date_str, '%Y%m%d')
                    end_date = datetime.strptime(end_date_str, '%Y%m%d')

                    current_date = start_date
                    while current_date <= end_date:
                        day_str = current_date.strftime('%Y%m%d')
                        if day_str not in day_to_file:
                            day_to_file[day_str] = {}
                        if var not in day_to_file[day_str]:
                            day_to_file[day_str][var] = []
                        day_to_file[day_str][var].append(file)
                        current_date += timedelta(days=1)
                # else:
                    # self.logger.warning(f"Filename does not match expected pattern: {file.name}")

        # self.logger.info(f"Total unique days mapped: {len(day_to_file)}")
        return day_to_file

    def calculate_total_days(self) -> int:
        """
        Calculate the total number of unique days available across all variables.

        Returns:
            int: Total number of days.
        """
        unique_days = set(self.day_to_file_map.keys())
        return len(unique_days)

    def get_pressure_levels(self, ds: xr.Dataset) -> np.ndarray:
        """
        Calculate pressure levels from hybrid coordinates

        Parameters:
            ds: xarray Dataset containing hybrid level coordinates

        Returns:
            ndarray: Pressure levels in hPa
        """
        # self.logger.info("Calculating pressure levels...")

        try:
            # Check if required variables exist
            required_vars = ['P0', 'hyam', 'hybm']
            for var in required_vars:
                if var not in ds:
                    # self.logger.error(f"Missing required variable: {var}")
                    raise KeyError(f"Dataset missing required variable: {var}")

            # Get reference pressure in hPa
            P0 = ds.P0 / 100  # Convert Pa to hPa
            # self.logger.debug(f"Reference pressure P0: {P0.values} hPa")

            # Get hybrid coefficients
            hyam = ds.hyam
            hybm = ds.hybm

            # self.logger.debug(f"Shape of hybrid coefficients - hyam: {hyam.shape}, hybm: {hybm.shape}")

            # Calculate approximate pressure levels at midpoints
            p_levels = P0 * (hyam + hybm)

            # self.logger.info(f"Calculated {len(p_levels)} pressure levels")
            # self.logger.debug(f"Pressure levels (hPa): {p_levels.values}")

            return p_levels.values

        except Exception as e:
            # self.logger.error(f"Error calculating pressure levels: {str(e)}")
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
            # self.logger.debug(f"Target pressure level {target} hPa matched with actual level {p_levels[idx]} hPa at index {idx}")

        # Ensure no duplicate indices
        unique_indices = sorted(set(indices))
        # self.logger.info(f"Selected unique pressure level indices: {unique_indices}")
        return unique_indices

    def analyze_available_variables(self) -> Dict[str, Dict]:
        """
        Analyze all available variables in the directory and their characteristics
        """
        # self.logger.info("\n" + "=" * 80)
        # self.logger.info("ANALYZING AVAILABLE VARIABLES")
        # self.logger.info("=" * 80)

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

        # self.logger.info("Scanning directory for .nc files...")

        for file in self.base_dir.glob('*.nc'):
            match = pattern.match(str(file))
            if match:
                freq, var_name, time_period = match.groups()
                file_size = file.stat().st_size / (1024 * 1024)  # Size in MB

                # var_info[var_name]['files'].append(file.name)
                # var_info[var_name]['total_size'] += file_size
                # var_info[var_name]['time_periods'].add(time_period)
                # var_info[var_name]['frequencies'].add(freq)

                total_files += 1
                total_size += file_size

        # Print summary
        # self.logger.info("\nVARIABLE SUMMARY:")
        # self.logger.info("-" * 80)
        # self.logger.info(f"{'Variable':<20} {'Files':<8} {'Size (MB)':<12} {'Frequencies':<15} {'Time Periods'}")
        # self.logger.info("-" * 80)

        # for var_name, info in sorted(var_info.items()):
            # self.logger.info(
            #     f"{var_name:<20} "
            #     f"{len(info['files']):<8} "
            #     f"{info['total_size']:,.2f} MB  "
            #     f"{','.join(info['frequencies']):<15} "
            #     f"{len(info['time_periods'])}"
            # )

        # self.logger.info("-" * 80)
        # self.logger.info(f"Total variables: {len(var_info)}")
        # self.logger.info(f"Total files: {total_files}")
        # self.logger.info(f"Total size: {total_size:,.2f} MB")

        # Check for missing required variables
        missing_vars = set(self.variables) - set(var_info.keys())
        # if missing_vars:
            # self.logger.warning("\nMISSING REQUIRED VARIABLES:")
            # for var in missing_vars:
                # self.logger.warning(f"- {var}")

        # self.logger.info("=" * 80 + "\n")

        return var_info

    def group_days_by_file(self, var: str, days: List[str]) -> Dict[Path, List[str]]:
        """
        Group a list of days by their corresponding files for a specific variable.

        Parameters:
            var (str): Variable name.
            days (List[str]): List of day strings in 'YYYYMMDD' format.

        Returns:
            Dict[Path, List[str]]: Mapping from file paths to lists of days.
        """
        files_map = defaultdict(list)
        for day in days:
            files = self.day_to_file_map.get(day, {}).get(var, [])
            if not files:
                # self.logger.error(f"No file found for Variable: {var} on Day: {day}")
                raise FileNotFoundError(f"No file found for Variable: {var} on Day: {day}")
            # Assuming one file per variable per day
            files_map[files[0]].append(day)
        return files_map

    def get_day_by_index(self, index: int) -> str:
        """
        Retrieve the day string (YYYYMMDD) corresponding to a given index.

        Parameters:
            index (int): Index of the day.

        Returns:
            str: Day string in 'YYYYMMDD' format.
        """
        if index < 0 or index >= self.total_days:
            # self.logger.error(f"Index {index} out of bounds for total days {self.total_days}")
            raise IndexError(f"Index {index} out of bounds for total days {self.total_days}")

        # Assuming days are sorted
        sorted_days = sorted(self.day_to_file_map.keys())
        return sorted_days[index]

    def get_batch(self, start_idx: int, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve a batch of data for forecasting.

        Parameters:
            start_idx (int): Starting index for the batch.
            batch_size (int): Number of samples in the batch.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: X and Y tensors for the batch.
        """
        # self.logger.info(f"Retrieving batch: Start index = {start_idx}, Batch size = {batch_size}")

        if start_idx + batch_size + 1 > self.total_days:
            # self.logger.error("Batch exceeds available data range.")
            raise IndexError("Requested batch exceeds available data range.")

        # Generate list of day strings for X and Y
        X_days = [self.get_day_by_index(i) for i in range(start_idx, start_idx + batch_size)]
        Y_days = [self.get_day_by_index(i + 1) for i in range(start_idx, start_idx + batch_size)]

        # self.logger.debug(f"X_days: {X_days}")
        # self.logger.debug(f"Y_days: {Y_days}")

        # Initialize dictionaries to hold data arrays for X and Y
        X_data = defaultdict(list)
        Y_data = defaultdict(list)

        # Load data for X and Y
        for var in self.variables:
            # self.logger.info(f"\n{'=' * 40}\nLoading Variable: {var}\n{'=' * 40}")
            X_var_list = []
            Y_var_list = []

            # Group X_days and Y_days by file to minimize I/O
            X_files_map = self.group_days_by_file(var, X_days)
            Y_files_map = self.group_days_by_file(var, Y_days)

            # Load X data
            for file, days in X_files_map.items():
                try:
                    # self.logger.info(f"Opening file: {file.name} for Variable: {var}")
                    ds = xr.open_dataset(file)
                    data = ds[var]
                    # self.logger.info(f"Original shape of {var} in {file.name}: {data.shape}")

                    # Check if 'lev' dimension exists
                    if 'lev' in data.dims:
                        # self.logger.info("Variable has 'lev' dimension. Calculating pressure levels and filtering...")
                        p_levels = self.get_pressure_levels(ds)
                        level_indices = self.find_nearest_levels(p_levels)
                        # self.logger.info("Selecting target pressure levels...")
                        data = data.isel(lev=level_indices)
                        # self.logger.info(f"Shape after pressure level selection: {data.shape}")
                    else:
                        # First select the days
                        # self.logger.info(f"Selecting days: {days}")
                        file_days_formatted = [datetime.strptime(day, '%Y%m%d').date() for day in days]
                        
                        time_indices = [
                            i for i, t in enumerate(data['time'].values) 
                            if (t.year, t.month, t.day) in [
                                (d.year, d.month, d.day) for d in file_days_formatted
                            ]
                        ]
                        
                        if not time_indices:
                            # self.logger.warning(f"No matching days found in file: {file.name} for Variable: {var}")
                            continue
                            
                        data = data.isel(time=time_indices)
                        # self.logger.info(f"Shape after selecting days: {data.shape}")
                        
                        # Then add the singleton dimension
                        # self.logger.info("Adding singleton 'lev' dimension...")
                        data = data.expand_dims({'lev': 1}, axis=1)
                        # self.logger.info(f"Shape after adding singleton 'lev' dimension: {data.shape}")

                    # Select required days within the file
                    # self.logger.info(f"Selecting days: {days}")
                    # Convert day strings to datetime objects for comparison
                    file_days_formatted = [datetime.strptime(day, '%Y%m%d').date() for day in days]
                    
                    # Modified time selection logic for cftime dates
                    time_indices = [
                        i for i, t in enumerate(data['time'].values) 
                        if (t.year, t.month, t.day) in [
                            (d.year, d.month, d.day) for d in file_days_formatted
                        ]
                    ]

                    if not time_indices:
                        # self.logger.warning(f"No matching days found in file: {file.name} for Variable: {var}")
                        continue
                    selected_data = data.isel(time=time_indices)
                    # self.logger.info(f"Selected data shape for {var} from file {file.name}: {selected_data.shape}")

                    # Convert to NumPy array and append to list
                    X_var_list.append(selected_data.values)
                    ds.close()
                except Exception as e:
                    # self.logger.error(f"Error loading X data for Variable: {var} from file: {file.name} - {e}")
                    raise

            # Load Y data
            for file, days in Y_files_map.items():
                try:
                    # self.logger.info(f"Opening file: {file.name} for Variable: {var}")
                    ds = xr.open_dataset(file)
                    data = ds[var]
                    # self.logger.info(f"Original shape of {var} in {file.name}: {data.shape}")

                    # Check if 'lev' dimension exists
                    if 'lev' in data.dims:
                        # self.logger.info("Variable has 'lev' dimension. Calculating pressure levels and filtering...")
                        p_levels = self.get_pressure_levels(ds)
                        level_indices = self.find_nearest_levels(p_levels)
                        # self.logger.info("Selecting target pressure levels...")
                        data = data.isel(lev=level_indices)
                        # self.logger.info(f"Shape after pressure level selection: {data.shape}")
                    else:
                        # First select the days
                        # self.logger.info(f"Selecting days: {days}")
                        file_days_formatted = [datetime.strptime(day, '%Y%m%d').date() for day in days]
                        
                        time_indices = [
                            i for i, t in enumerate(data['time'].values) 
                            if (t.year, t.month, t.day) in [
                                (d.year, d.month, d.day) for d in file_days_formatted
                            ]
                        ]
                        
                        if not time_indices:
                            # self.logger.warning(f"No matching days found in file: {file.name} for Variable: {var}")
                            continue
                            
                        data = data.isel(time=time_indices)
                        # self.logger.info(f"Shape after selecting days: {data.shape}")
                        
                        # Then add the singleton dimension
                        # self.logger.info("Adding singleton 'lev' dimension...")
                        data = data.expand_dims({'lev': 1}, axis=1)
                        # self.logger.info(f"Shape after adding singleton 'lev' dimension: {data.shape}")

                    # Select required days within the file
                    # self.logger.info(f"Selecting days: {days}")
                    file_days_formatted = [datetime.strptime(day, '%Y%m%d').date() for day in days]
                    
                    # Modified time selection logic for cftime dates
                    time_indices = [
                        i for i, t in enumerate(data['time'].values) 
                        if (t.year, t.month, t.day) in [
                            (d.year, d.month, d.day) for d in file_days_formatted
                        ]
                    ]

                    # if not time_indices:
                        # self.logger.warning(f"No matching days found in file: {file.name} for Variable: {var}")
                        # continue
                    selected_data = data.isel(time=time_indices)
                    # self.logger.info(f"Selected data shape for {var} from file {file.name}: {selected_data.shape}")

                    # Convert to NumPy array and append to list
                    Y_var_list.append(selected_data.values)
                    ds.close()
                except Exception as e:
                    # self.logger.error(f"Error loading Y data for Variable: {var} from file: {file.name} - {e}")
                    raise

            # Concatenate data from all files for the current variable
            if X_var_list:
                X_data[var] = np.concatenate(X_var_list, axis=0)  # Shape: (batch_size, features, lat, lon)
                # self.logger.info(f"Final concatenated X shape for {var}: {X_data[var].shape}")
            # else:
                # self.logger.warning(f"No X data loaded for Variable: {var}")

            if Y_var_list:
                Y_data[var] = np.concatenate(Y_var_list, axis=0)
                # self.logger.info(f"Final concatenated Y shape for {var}: {Y_data[var].shape}")
            # else:
                # self.logger.warning(f"No Y data loaded for Variable: {var}")

        # Convert dictionaries to tensors after loading all variables
        X_tensor_list = []
        Y_tensor_list = []
        for var in self.variables:
            if var in X_data:
                tensor_X = torch.from_numpy(X_data[var])  # Shape: (batch_size, features, lat, lon)
                X_tensor_list.append(tensor_X)
            # else:
                # self.logger.warning(f"Variable {var} missing in X_data.")

            if var in Y_data:
                tensor_Y = torch.from_numpy(Y_data[var])  # Shape: (batch_size, features, lat, lon)
                Y_tensor_list.append(tensor_Y)
            # else:
                # self.logger.warning(f"Variable {var} missing in Y_data.")

        # Concatenate features across variables
        if X_tensor_list:
            X_tensor = torch.cat(X_tensor_list, dim=1)  # Shape: (batch_size, total_features, lat, lon)
            # self.logger.info(f"Concatenated X tensor shape: {X_tensor.shape}")
        else:
            # self.logger.error("No X data available to form tensor.")
            raise ValueError("No X data available to form tensor.")

        if Y_tensor_list:
            Y_tensor = torch.cat(Y_tensor_list, dim=1)  # Shape: (batch_size, total_features, lat, lon)
            # self.logger.info(f"Concatenated Y tensor shape: {Y_tensor.shape}")
        else:
            # self.logger.error("No Y data available to form tensor.")
            raise ValueError("No Y data available to form tensor.")

        # self.logger.info(f"Final Batch X shape: {X_tensor.shape}")
        # self.logger.info(f"Final Batch Y shape: {Y_tensor.shape}")

        return X_tensor, Y_tensor

    def run_batch_processing(self, batch_indices: List[Tuple[int, int]]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Process multiple batches given their indices.

        Parameters:
            batch_indices (List[Tuple[int, int]]): List of (start_idx, batch_size) tuples.

        Returns:
            List[Tuple[torch.Tensor, torch.Tensor]]: List of (X, Y) tensors for each batch.
        """
        # self.logger.info("Starting batch processing...")
        batches = []
        for start_idx, batch_size in batch_indices:
            try:
                X, Y = self.get_batch(start_idx, batch_size)
                batches.append((X, Y))
                # self.logger.info(f"Successfully processed batch starting at index {start_idx}")
            except Exception as e:
                # self.logger.error(f"Error processing batch starting at index {start_idx}: {str(e)}")
                continue
        return batches


    def process_variable(self, var_name: str, files: List[Path]) -> xr.DataArray:
        """Process a single variable across all its files"""
        # self.logger.info("\n" + "=" * 80)
        # self.logger.info(f"PROCESSING VARIABLE: {var_name}")
        # self.logger.info("=" * 80)

        if not files:
            # self.logger.warning(f"No files found for {var_name}")
            return None

        # Sort files by time period
        files = sorted(files)

        # Create Zarr store path
        zarr_path = self.output_dir / f'{var_name}.zarr'
        if zarr_path.exists():
            # self.logger.info(f"Removing existing Zarr store: {zarr_path}")
            shutil.rmtree(zarr_path)

        # Initialize timers
        start_time = time.time()

        # Initialize ProgressBar for file processing
        with tqdm(files, desc=f"Processing {var_name}", unit="file") as pbar:
            for i, file in enumerate(pbar, 1):
                pbar.set_postfix(file=file.name)
                # self.logger.info(f"\nProcessing file {i}/{len(files)}: {file.name}")

                try:
                    # self.logger.info("Opening dataset with xarray...")
                    # Open dataset without Dask since we're handling batching manually
                    ds = xr.open_dataset(file)
                    # self.logger.info(f"Opened dataset: {file.name}")
                    # self.logger.info(f"Dataset dimensions: {ds[var_name].dims}")
                    # self.logger.info(f"Dataset shape: {ds[var_name].shape}")

                    # Log data type information
                    dtype = ds[var_name].dtype
                    # self.logger.info(f"Data type for {var_name}: {dtype}")
                    # self.logger.info(f"Data type size in bits: {dtype.itemsize * 8}")

                    # Handle pressure levels if 'lev' dimension exists
                    if 'lev' in ds[var_name].dims:
                        # self.logger.info("Variable has 'lev' dimension. Calculating pressure levels and filtering...")
                        p_levels = self.get_pressure_levels(ds)
                        level_indices = self.find_nearest_levels(p_levels)
                        # self.logger.info("Selecting target pressure levels...")
                        data = ds[var_name].isel(lev=level_indices)
                        # self.logger.info(f"Shape after pressure level selection: {data.shape}")
                    else:
                        # self.logger.info("Variable does not have 'lev' dimension. Adding singleton 'lev' dimension...")
                        data = ds[var_name].expand_dims('lev')
                        # self.logger.info(f"Shape after adding singleton 'lev' dimension: {data.shape}")

                    # Define chunk dictionary based on presence of 'lev' dimension
                    if 'lev' in data.dims:
                        chunk_dict = {
                            'time': 100,   # Adjust as needed
                            'lat': 192,
                            'lon': 288,
                            'lev': data.dims.index('lev')
                        }
                    else:
                        chunk_dict = {
                            'time': 100,   # Adjust as needed
                            'lat': 192,
                            'lon': 288
                        }

                    # self.logger.info(f"Re-chunking data with chunks: {chunk_dict}")
                    data = data.chunk(chunk_dict)
                    # self.logger.debug(f"New chunks: {data.chunks}")

                    # Add memory tracking
                    current_mem = psutil.Process(os.getpid()).memory_info().rss / 1e9
                    # self.logger.info(f"Current memory usage: {current_mem:.2f} GB")

                    # Estimate size of data in GB
                    data_size_gb = data.nbytes / (1024 ** 3)
                    # self.logger.info(f"Size of data to write: {data_size_gb:.2f} GB")

                    # Define encoding with compressor
                    encoding = {var_name: {'compressor': self.compressor}}

                    # Write to Zarr incrementally with encoding
                    if i == 1:
                        # self.logger.info(f"Initializing Zarr store for {var_name} with mode='w'")
                        data.to_zarr(
                            zarr_path,
                            mode='w',
                            encoding=encoding,
                            compute=True,
                            safe_chunks=False  # Disable safe_chunks to allow variable chunk sizes
                        )
                        # self.logger.info(f"Initialized Zarr store for {var_name}")
                    else:
                        # self.logger.info(f"Appending to Zarr store for {var_name} with mode='a' and append_dim='time'")
                        data.to_zarr(
                            zarr_path,
                            mode='a',
                            append_dim='time',
                            compute=True,
                            safe_chunks=False  # Disable safe_chunks for appending
                        )
                        # self.logger.info(f"Appended data to Zarr store for {var_name}")

                    # self.logger.info(f"Successfully processed and written file {file.name} for variable {var_name}")

                except Exception as e:
                    # self.logger.error(f"Error processing {file.name} for variable {var_name}: {str(e)}")
                    raise

        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        # self.logger.info(f"Finished processing variable {var_name} in {elapsed_time / 60:.2f} minutes.")

        # After all files are processed, open the Zarr store
        # self.logger.info("Opening concatenated Zarr store...")
        try:
            final_data = xr.open_zarr(zarr_path, consolidated=True, chunks={'time': 200})  # Adjusted chunk size
            # self.logger.info(f"Opened Zarr store for {var_name}")
            # self.logger.info(f"Final shape for {var_name}: {final_data.shape}")
            # self.logger.info(f"Chunks: {final_data.chunks}")
        except Exception as e:
            # self.logger.error(f"Error opening Zarr store for {var_name}: {str(e)}")
            raise

        return final_data

    def process_all_batches(self, batch_size: int) -> None:
        """
        Process all data in batches and save them to Zarr stores.

        Parameters:
            batch_size (int): Number of samples in each batch.
        """
        # self.logger.info("Processing all batches and saving to Zarr...")
        num_batches = self.total_days // batch_size
        batch_indices = [(i * batch_size, batch_size) for i in range(num_batches)]

        for start_idx, size in tqdm(batch_indices, desc="Processing Batches"):
            try:
                X, Y = self.get_batch(start_idx, size)
                # Define Zarr store paths for X and Y
                zarr_X_path = self.output_dir / f'batch_X_{start_idx:05d}.zarr'
                zarr_Y_path = self.output_dir / f'batch_Y_{start_idx:05d}.zarr'

                # Define encoding with compressor
                encoding = {'zarr': {'compressor': self.compressor}}

                # Convert tensors back to xarray DataArrays for saving
                X_da = xr.DataArray(
                    X.numpy(),
                    dims=('batch', 'singleton', 'feature', 'lat', 'lon'),
                    name='X'
                )
                Y_da = xr.DataArray(
                    Y.numpy(),
                    dims=('batch', 'singleton', 'feature', 'lat', 'lon'),
                    name='Y'
                )

                # Save to Zarr
                X_da.to_zarr(zarr_X_path, mode='w', encoding=encoding)
                Y_da.to_zarr(zarr_Y_path, mode='w', encoding=encoding)

                # self.logger.info(f"Saved batch {start_idx} to Zarr stores.")
            except Exception as e:
                # self.logger.error(f"Failed to process and save batch starting at {start_idx}: {str(e)}")
                continue

        # self.logger.info("All batches processed and saved.")

    def close(self):
        """Placeholder for any cleanup if necessary."""
        # self.logger.info("Processing complete. No resources to clean up.")

    def __del__(self):
        """Ensure any necessary cleanup upon deletion."""
        self.close()

# Example Usage
if __name__ == "__main__":
    base_dir = "/teamspace/s3_connections/ncar-cesm2-arise-bucket/ARISE-SAI-1.5/b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.001/atm/proc/tseries/day_1"
    output_dir = "/teamspace/studios/this_studio/arise_processed"

    # Initialize processor
    processor = ARISEProcessor(base_dir, output_dir)

    # Define batch parameters
    batch_size = 4  # For demonstration; adjust as needed
    num_batches = 1  # For demonstration; adjust as needed
    batch_indices = [(i * batch_size, batch_size) for i in range(num_batches)]

    # Retrieve and process batches
    batches = processor.run_batch_processing(batch_indices)

    # Example: Access the first batch
    if batches:
        X_batch, Y_batch = batches[0]
        print(f"First batch X shape: {X_batch.shape}")
        print(f"First batch Y shape: {Y_batch.shape}")

    # Alternatively, process and save all batches
    # processor.process_all_batches(batch_size=32)

    # Create the final datacube with parallel processing
    print("Final datacube created successfully.")

    # Cleanup
    processor.close()
