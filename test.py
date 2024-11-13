import torch
from torch.utils.data import Dataset, DataLoader
import xarray as xr
import s3fs
import numpy as np

from typing import List, Tuple, Dict
import logging
from pathlib import Path
import time

# Enhanced logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class ARISEBatchDataset(Dataset):
    def __init__(self, 
                 simulation_number: str = "001",
                 device: str = "cuda",
                 verbose: bool = True,
                 batch_size: int = 4):
        """
        Args:
            simulation_number: ARISE simulation number (001-009)
            device: torch device for tensors
            verbose: whether to print detailed progress
            batch_size: size of batches to process
        """
        logger.info(f"Initializing ARISEBatchDataset with simulation {simulation_number} on {device}")
        start_time = time.time()
        
        self.device = device
        self.verbose = verbose
        self.batch_size = batch_size
        
        logger.info("Connecting to S3 filesystem...")
        self.fs = s3fs.S3FileSystem(anon=True)
        self.s3_bucket = "ncar-cesm2-arise"
        self.simulation_number = simulation_number
        
        # Define paths with proper simulation number
        logger.info(f"Setting up paths for simulation {simulation_number}")
        base_path = f"ARISE-SAI-1.5/b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.{simulation_number}/atm/proc/tseries/month_1/"
        self.paths = {
            'AODVISstdn': (f"{base_path}b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.{simulation_number}.cam.h0.AODVISstdn.203501-206912.nc", False, True),
            'SST': (f"{base_path}b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.{simulation_number}.cam.h0.SST.203501-206912.nc", False, True),
            'SOLIN': (f"{base_path}b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.{simulation_number}.cam.h0.SOLIN.203501-206912.nc", False, True),
            'Q': (f"{base_path}b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.{simulation_number}.cam.h0.Q.203501-206912.nc", True, False),
            'U': (f"{base_path}b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.{simulation_number}.cam.h0.U.203501-206912.nc", True, False),
            'V': (f"{base_path}b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.{simulation_number}.cam.h0.V.203501-206912.nc", True, False),
            'PS': (f"{base_path}b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.{simulation_number}.cam.h0.PS.203501-206912.nc", False, False),
            'T': (f"{base_path}b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.{simulation_number}.cam.h0.T.203501-206912.nc", True, False),
            'TS': (f"{base_path}b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.{simulation_number}.cam.h0.TS.203501-206912.nc", False, False)
        }
        
        logger.info("Setting up vertical level indices...")
        self.vert_level_indices = [24, 36, 39, 41, 44, 46, 49, 52, 56, 59, 62, 69]
        self.data_cache = {}
        
        logger.info("Initializing dataset length...")
        self._initialize_length()
        
        # Calculate number of full batches
        self.num_batches = self.length // self.batch_size
        self.last_batch_size = self.length % self.batch_size
        
        init_time = time.time() - start_time
        logger.info(f"Dataset initialization completed in {init_time:.2f} seconds")
        logger.info(f"Total timesteps: {self.length}")
        logger.info(f"Number of full batches: {self.num_batches}")
        logger.info(f"Last batch size: {self.last_batch_size}")
        
    def _initialize_length(self):
        """Initialize dataset length using first variable"""
        first_var = next(iter(self.paths.keys()))
        path, has_levels, _ = self.paths[first_var]
        full_path = f"{self.s3_bucket}/{path}"
        logger.info(f"Determining dataset length from {first_var} at {full_path}")
        
        try:
            with self.fs.open(full_path) as f:
                ds = xr.open_dataset(f)
                self.length = len(ds.time) - 1  # -1 because we need pairs
                logger.info(f"Dataset length initialized: {self.length + 1} timestamps ({self.length} usable pairs)")
        except Exception as e:
            logger.error(f"Failed to initialize length: {str(e)}")
            raise
            
    def _load_variable_batch(self, var_name: str, start_idx: int, batch_size: int) -> torch.Tensor:
        """Load a batch of timesteps for a variable"""
        start_time = time.time()
        path, has_levels, _ = self.paths[var_name]
        full_path = f"{self.s3_bucket}/{path}"
        
        if self.verbose:
            logger.debug(f"Loading {var_name} batch starting at index {start_idx}")
        
        try:
            with self.fs.open(full_path) as f:
                ds = xr.open_dataset(f)
                if has_levels:
                    data = ds[var_name].isel(
                        time=slice(start_idx, start_idx + batch_size + 1),  # +1 for t+1 prediction
                        lev=self.vert_level_indices
                    )
                    data = data.transpose('time', 'lev', 'lat', 'lon')
                else:
                    data = ds[var_name].isel(time=slice(start_idx, start_idx + batch_size + 1))
                    data = data.expand_dims('lev')
                    
                tensor = torch.from_numpy(data.values).float()
                if self.device == "cuda":
                    tensor = tensor.cuda()
                
                load_time = time.time() - start_time
                if self.verbose:
                    logger.debug(f"Loaded {var_name} batch in {load_time:.3f} seconds. Tensor shape: {tensor.shape}")
                return tensor
                
        except Exception as e:
            logger.error(f"Error loading {var_name} batch at index {start_idx}: {str(e)}")
            raise

    def __len__(self) -> int:
        return self.num_batches + (1 if self.last_batch_size > 0 else 0)

    def __getitem__(self, batch_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start_time = time.time()
        logger.debug(f"Processing batch {batch_idx}")
        
        # Determine batch size for this iteration
        is_last_batch = batch_idx == self.num_batches
        current_batch_size = self.last_batch_size if is_last_batch else self.batch_size
        start_idx = batch_idx * self.batch_size
        
        # Load input features (t)
        logger.debug(f"Loading input features for batch starting at index {start_idx}")
        x_tensors = []
        for var_name, (_, _, forcing_only) in self.paths.items():
            x_tensor = self._load_variable_batch(var_name, start_idx, current_batch_size)
            x_tensors.append(x_tensor[:-1])  # Remove t+1 timestep
        
        # Load output features (t+1)
        logger.debug(f"Loading output features for batch")
        y_tensors = []
        for var_name, (_, _, forcing_only) in self.paths.items():
            if not forcing_only:
                y_tensor = self._load_variable_batch(var_name, start_idx, current_batch_size)
                y_tensors.append(y_tensor[1:])  # Use t+1 timesteps
        
        # Combine tensors
        X = torch.cat([t.reshape(current_batch_size, -1, 192, 288) for t in x_tensors], dim=1)
        Y = torch.cat([t.reshape(current_batch_size, -1, 192, 288) for t in y_tensors], dim=1)
        
        total_time = time.time() - start_time
        logger.debug(f"Total time to load batch {batch_idx}: {total_time:.3f} seconds")
        
        return X, Y

def get_arise_batch_dataloader(simulation_number: str = "001",
                             batch_size: int = 4,
                             device: str = "cuda",
                             verbose: bool = True) -> DataLoader:
    """Create a DataLoader for ARISE climate data with efficient batch processing."""
    logger.info(f"Creating BatchDataLoader with batch_size={batch_size} on {device}")
    
    dataset = ARISEBatchDataset(
        simulation_number=simulation_number,
        device=device,
        verbose=verbose,
        batch_size=batch_size
    )
    
    logger.info("Initializing BatchDataLoader...")
    loader = DataLoader(
        dataset,
        batch_size=None,  # We handle batching in the dataset
        shuffle=True,
        num_workers=4,
        pin_memory=(device=="cuda")
    )
    
    logger.info(f"BatchDataLoader created successfully. {len(dataset)} batches")
    return loader

if __name__ == "__main__":
    logger.info("Starting ARISE batch data loader test...")
    
    try:
        start_time = time.time()
        batch_size = 4
        
        logger.info(f"Creating test dataset with batch_size={batch_size}...")
        test_dataset = ARISEBatchDataset(
            simulation_number="001",
            verbose=True,
            batch_size=batch_size
        )
        
        logger.info(f"{'='*50}")
        logger.info(f"Dataset contains {len(test_dataset)} batches")
        
        # Test first batch
        logger.info("Loading first batch...")
        batch_start = time.time()
        X1, Y1 = test_dataset[0]
        batch1_time = time.time() - batch_start
        logger.info(f"Successfully loaded first batch")
        logger.info(f"First batch input shape: {X1.shape}")
        logger.info(f"First batch output shape: {Y1.shape}")
        logger.info(f"First batch load time: {batch1_time:.2f} seconds")
        
        total_time = time.time() - start_time
        logger.info(f"\nTest completed successfully in {total_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}", exc_info=True)