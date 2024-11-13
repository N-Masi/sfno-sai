import torch
from torch.utils.data import Dataset, DataLoader
import logging
from pathlib import Path
from typing import Tuple, Dict
import numpy as np
import torch.multiprocessing as mp
import warnings

# Suppress FutureWarning if necessary
warnings.filterwarnings("ignore", category=FutureWarning)

# Set multiprocessing start method
if mp.get_start_method(allow_none=True) != 'spawn':
    mp.set_start_method('spawn', force=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def cpu_collate_fn(batch):
    """Collate function without device transfer"""
    x_batch, y_batch = zip(*batch)
    x_tensor = torch.stack(x_batch)
    y_tensor = torch.stack(y_batch)
    return x_tensor, y_tensor

class ProcessedARISEDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        simulation_number: str = "001",
        train_fraction: float = 0.8,
        split: str = "train"
    ):
        self.split = split
        
        logger.info(f"Loading processed ARISE data from {data_dir}")
        data_path = Path(data_dir) / f"arise_{simulation_number}_processed.pt"
        metadata_path = Path(data_dir) / f"arise_{simulation_number}_metadata.pt"
        
        # Load tensors on CPU to prevent device conflicts
        data = torch.load(data_path, map_location='cpu')
        self.X = data['X']
        self.Y = data['Y']
        self.metadata = torch.load(metadata_path, map_location='cpu')
        
        # Calculate split indices
        total_samples = len(self.X)
        split_idx = int(total_samples * train_fraction)
        
        if split == 'train':
            self.indices = np.arange(split_idx)
        else:  # validation
            self.indices = np.arange(split_idx, total_samples)
            
        logger.info(f"Loaded {self.split} split with {len(self.indices)} samples")
        logger.info(f"X shape: {self.X.shape}, Y shape: {self.Y.shape}")
        
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        real_idx = self.indices[idx]
        x = self.X[real_idx]
        y = self.Y[real_idx]
        assert x.device.type == 'cpu', "Input tensor X is not on CPU"
        assert y.device.type == 'cpu', "Output tensor Y is not on CPU"
        return x, y

def get_arise_dataloaders(
    data_dir: str,
    simulation_number: str = "001",
    batch_size: int = 32,
    train_fraction: float = 0.8,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders for processed ARISE data."""
    
    # Create datasets
    train_dataset = ProcessedARISEDataset(
        data_dir=data_dir,
        simulation_number=simulation_number,
        train_fraction=train_fraction,
        split='train'
    )
    
    val_dataset = ProcessedARISEDataset(
        data_dir=data_dir,
        simulation_number=simulation_number,
        train_fraction=train_fraction,
        split='val'
    )
    
    # Create dataloaders with pin_memory based on CUDA availability
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=cpu_collate_fn,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=cpu_collate_fn,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader

if __name__ == "__main__":
    # Example usage
    data_dir = "./arise_processed_data"
    batch_size = 4
    
    try:
        logger.info("Creating ARISE dataloaders...")
        train_loader, val_loader = get_arise_dataloaders(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=4
        )
        
        logger.info(f"Created dataloaders:")
        logger.info(f"Training batches: {len(train_loader)}")
        logger.info(f"Validation batches: {len(val_loader)}")
        
        # Test loading a batch
        logger.info("\nTesting batch loading...")
        train_batch = next(iter(train_loader))
        X, Y = train_batch
        
        if torch.cuda.is_available():
            X = X.cuda(non_blocking=True)
            Y = Y.cuda(non_blocking=True)
        
        logger.info(f"Loaded batch shapes:")
        logger.info(f"X: {X.shape}")
        logger.info(f"Y: {Y.shape}")
        logger.info(f"Device: {X.device}")
        
    except Exception as e:
        logger.error("Error in dataloader test!", exc_info=True)
        raise
