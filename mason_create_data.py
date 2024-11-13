import torch
from torch.utils.data import Dataset, DataLoader
import xarray as xr
import s3fs
import numpy as np
from typing import List, Tuple, Dict
import logging
from pathlib import Path
import time
from tqdm import tqdm

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

class ARISEDataProcessor:
    def __init__(
        self,
        simulation_number: str = "001",
        device: str = "cuda",
        batch_size: int = 4,
        output_dir: str = "./processed_data",
        verbose: bool = True
    ):
        """
        Initialize the ARISE data processor
        
        Args:
            simulation_number: ARISE simulation number (001-009)
            device: torch device for processing
            batch_size: size of batches to process
            output_dir: directory to save processed tensors
            verbose: whether to print detailed progress
        """
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.dataset = ARISEBatchDataset(
            simulation_number=simulation_number,
            device=device,
            verbose=verbose,
            batch_size=batch_size
        )
        
        # Initialize empty tensors to store complete data
        self.total_samples = self.dataset.length
        logger.info(f"Initializing storage for {self.total_samples} samples")
        
        # Get shapes from first batch
        X_sample, Y_sample = self.dataset[0]
        self.X_channels = X_sample.shape[1]
        self.Y_channels = Y_sample.shape[1]
        
        self.processed_file = self.output_dir / f"arise_{simulation_number}_processed.pt"
        self.metadata_file = self.output_dir / f"arise_{simulation_number}_metadata.pt"
        
    def process_and_save(self, checkpoint_frequency: int = 10):
        """
        Process all data batches and save to disk with checkpointing
        
        Args:
            checkpoint_frequency: Save checkpoint every N batches
        """
        logger.info("Starting data processing...")
        start_time = time.time()
        
        # Initialize tensors on CPU to save memory
        X_complete = torch.zeros(
            (self.total_samples, self.X_channels, 192, 288),
            dtype=torch.float32
        )
        Y_complete = torch.zeros(
            (self.total_samples, self.Y_channels, 192, 288),
            dtype=torch.float32
        )
        
        current_idx = 0
        
        try:
            for batch_idx in tqdm(range(len(self.dataset)), desc="Processing batches"):
                X_batch, Y_batch = self.dataset[batch_idx]
                
                # Move batch tensors to CPU if they're on GPU
                if X_batch.device.type == "cuda":
                    X_batch = X_batch.cpu()
                    Y_batch = Y_batch.cpu()
                
                # Get current batch size (might be smaller for last batch)
                batch_size = X_batch.shape[0]
                
                # Store in complete tensors
                X_complete[current_idx:current_idx + batch_size] = X_batch
                Y_complete[current_idx:current_idx + batch_size] = Y_batch
                
                current_idx += batch_size
                
                # Save checkpoint if needed
                if (batch_idx + 1) % checkpoint_frequency == 0:
                    self._save_checkpoint(X_complete, Y_complete, batch_idx + 1)
            
            # Final save
            self._save_final(X_complete, Y_complete)
            
            total_time = time.time() - start_time
            logger.info(f"Processing completed in {total_time:.2f} seconds")
            logger.info(f"Processed data saved to {self.processed_file}")
            logger.info(f"Metadata saved to {self.metadata_file}")
            
        except Exception as e:
            logger.error(f"Error during processing: {str(e)}", exc_info=True)
            raise
    
    def _save_checkpoint(self, X: torch.Tensor, Y: torch.Tensor, batch_num: int):
        """Save processing checkpoint"""
        checkpoint_file = self.output_dir / f"checkpoint_batch_{batch_num}.pt"
        torch.save({
            'X': X,
            'Y': Y,
            'batch_num': batch_num
        }, checkpoint_file)
        logger.info(f"Saved checkpoint after batch {batch_num}")
    
    def _save_final(self, X: torch.Tensor, Y: torch.Tensor):
        """Save final processed data and metadata"""
        # Save processed data
        torch.save({
            'X': X,
            'Y': Y
        }, self.processed_file)
        
        # Save metadata
        metadata = {
            'total_samples': self.total_samples,
            'X_channels': self.X_channels,
            'Y_channels': self.Y_channels,
            'spatial_dims': (192, 288),
            'variables': list(self.dataset.paths.keys()),
            'variable_metadata': self.dataset.paths,
            'vertical_levels': self.dataset.vert_level_indices
        }
        torch.save(metadata, self.metadata_file)

def load_processed_data(data_dir: str, simulation_number: str = "001"):
    """
    Load processed ARISE data from disk
    
    Args:
        data_dir: Directory containing processed data
        simulation_number: Simulation number to load
    
    Returns:
        tuple: (X, Y) tensors and metadata dictionary
    """
    data_path = Path(data_dir) / f"arise_{simulation_number}_processed.pt"
    metadata_path = Path(data_dir) / f"arise_{simulation_number}_metadata.pt"
    
    logger.info(f"Loading processed data from {data_path}")
    data = torch.load(data_path)
    metadata = torch.load(metadata_path)
    
    return data['X'], data['Y'], metadata

def get_arise_batch_dataloader(simulation_number: str = "001",
                             batch_size: int = 10,
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
    """
    Process all ARISE simulations (001-009)
    """
    # Set up parameters
    batch_size = 4
    output_dir = "./arise_processed_data"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Process simulations 001-009
    for sim_num in range(2, 10):
        simulation_number = f"{sim_num:03d}"  # Format as 001, 002, etc.
        
        try:
            logger.info(f"\nStarting ARISE data processing pipeline for simulation {simulation_number}...")
            
            # Initialize processor
            processor = ARISEDataProcessor(
                simulation_number=simulation_number,
                batch_size=batch_size,
                output_dir=output_dir,
                device=device,
                verbose=True
            )
            
            # Process and save data with checkpoints every 5 batches
            logger.info(f"Beginning data processing for simulation {simulation_number}...")
            processor.process_and_save(checkpoint_frequency=5)
            
            # Load and verify processed data
            logger.info(f"Loading processed data to verify simulation {simulation_number}...")
            X, Y, metadata = load_processed_data(output_dir, simulation_number)
            
            # Print summary information
            logger.info(f"\nProcessed Data Summary for Simulation {simulation_number}:")
            logger.info(f"X tensor shape: {X.shape}")
            logger.info(f"Y tensor shape: {Y.shape}")
            logger.info(f"Total samples: {metadata['total_samples']}")
            logger.info(f"Input channels: {metadata['X_channels']}")
            logger.info(f"Output channels: {metadata['Y_channels']}")
            logger.info(f"Spatial dimensions: {metadata['spatial_dims']}")
            logger.info(f"Variables processed: {', '.join(metadata['variables'])}")
            
            # Optional: Clean up checkpoint files
            checkpoint_files = list(Path(output_dir).glob(f"checkpoint_batch_*_{simulation_number}.pt"))
            if checkpoint_files:
                logger.info(f"\nCleaning up checkpoint files for simulation {simulation_number}...")
                for checkpoint in checkpoint_files:
                    checkpoint.unlink()
                logger.info(f"Removed {len(checkpoint_files)} checkpoint files")
            
            logger.info(f"\nProcessing pipeline completed successfully for simulation {simulation_number}!")
            
        except Exception as e:
            logger.error(f"Processing pipeline failed for simulation {simulation_number}!", exc_info=True)
            logger.error("Continuing with next simulation...")
            continue
    
    logger.info("\nAll simulations processing completed!")