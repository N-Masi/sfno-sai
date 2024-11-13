import torch
from torch.utils.data import Dataset, DataLoader
import logging
from pathlib import Path
from typing import Tuple, List
import warnings
import os
import wandb

# Import custom modules (ensure these are available in your environment)
from ace_helpers import get_ace_sto_sfno, get_ace_optimizer, get_ace_lr_scheduler, AceLoss
from modulus.launch.logging import LaunchLogger, initialize_wandb

# Suppress FutureWarning from torch.load if necessary
warnings.filterwarnings("ignore", category=FutureWarning)

# ============================ Configuration ============================ #

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Data configuration
DATA_DIR = "./arise_processed_data"  # Directory where processed data is stored
SIM_NUMS_TRAIN = ["001"]
ACE_BATCH_SIZE = 4
SINGLE_SIM_EPOCHS = 10  # Define as per your requirement

# Model configuration
IMG_SHAPE = (192, 288)
IN_CHANS = 48
OUT_CHANS = 48

# Logging and W&B configuration
WANDB_PROJECT = "SFNO_test"
WANDB_ENTITY = "nickmasi"
WANDB_RUN_NAME = "Masons train.py"

# Vertical level indices (as per your friend's script)
VERT_LEVEL_INDICES = [24, 36, 39, 41, 44, 46, 49, 52, 56, 59, 62, 69]

# Checkpoint configuration
CHECKPOINT_DIR = "./checkpoints"
SAVE_FREQUENCY = 5  # Save every N epochs

# ============================ Dataset Definition ============================ #

class ProcessedARISEDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        simulation_numbers: List[str],
        train_fraction: float = 0.8,
        split: str = "train"
    ):
        """
        Args:
            data_dir: Directory containing processed data.
            simulation_numbers: List of simulation numbers to include.
            train_fraction: Fraction of data to use for training.
            split: "train" or "val".
        """
        self.split = split
        self.simulation_numbers = simulation_numbers
        self.data_dir = Path(data_dir)
        self.train_fraction = train_fraction

        logger.info(f"Initializing ProcessedARISEDataset for split: {self.split}")
        
        # Load single simulation data
        data_path = self.data_dir / "arise_001_processed_norm.pt"
        metadata_path = self.data_dir / "arise_001_metadata.pt"
        
        logger.info(f"Loading data from {data_path}")
        data = torch.load(data_path, map_location='cpu')
        metadata = torch.load(metadata_path, map_location='cpu')
        
        self.X = data['X']
        self.Y = data['Y']
        
        logger.info(f"Loaded data - X shape: {self.X.shape}, Y shape: {self.Y.shape}")
        
        # Calculate split indices
        total_samples = len(self.X)
        split_idx = int(total_samples * self.train_fraction)
        
        if self.split == 'train':
            self.X = self.X[:split_idx]
            self.Y = self.Y[:split_idx]
        else:  # validation
            self.X = self.X[split_idx:]
            self.Y = self.Y[split_idx:]
        
        logger.info(f"{self.split.capitalize()} split - X shape: {self.X.shape}, Y shape: {self.Y.shape}")
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.Y[idx]

# ============================ DataLoader Setup ============================ #

def get_arise_dataloaders(
    data_dir: str,
    simulation_numbers: List[str],
    batch_size: int = 32,
    train_fraction: float = 0.8,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders for processed ARISE data."""
    
    # Create datasets
    train_dataset = ProcessedARISEDataset(
        data_dir=data_dir,
        simulation_numbers=simulation_numbers,
        train_fraction=train_fraction,
        split='train'
    )
    
    val_dataset = ProcessedARISEDataset(
        data_dir=data_dir,
        simulation_numbers=simulation_numbers,
        train_fraction=train_fraction,
        split='val'
    )
    
    # Define collate function (if any specific processing is needed)
    def collate_fn(batch):
        X_batch, Y_batch = zip(*batch)
        X_tensor = torch.stack(X_batch)
        Y_tensor = torch.stack(Y_batch)
        return X_tensor, Y_tensor
    
    # Create dataloaders with pin_memory based on CUDA availability
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available()
    )
    
    logger.info(f"Created DataLoaders - Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    return train_loader, val_loader

# ============================ Training Setup ============================ #

def train_model(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    loss_fn: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 10
):
    """Train the model using the provided dataloaders."""
    
    logger.info("Starting Training Process...")
    best_val_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        with LaunchLogger("train", epoch=epoch, mini_batch_log_freq=10) as launchlog:
            logger.info(f"Epoch {epoch}/{epochs}")
            
            # Training Phase
            model.train()
            running_loss = 0.0
            for batch_idx, (X, Y) in enumerate(train_loader, 1):
                X = X.to(DEVICE, non_blocking=True)
                Y = Y.to(DEVICE, non_blocking=True)
                
                optimizer.zero_grad()
                preds = model(X)
                loss = loss_fn(preds, Y)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                launchlog.log_minibatch({"Loss": loss.item()})
                
                if batch_idx % 10 == 0:
                    avg_loss = running_loss / 10
                    logger.info(f"  Batch {batch_idx} - Avg Loss: {avg_loss:.4f}")
                    running_loss = 0.0
            
            # Learning rate scheduling
            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]
            logger.info(f"  Learning Rate updated to: {current_lr}")
            launchlog.log_epoch({"Learning Rate": current_lr})
            
            # Validation Phase
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_val, Y_val in val_loader:
                    X_val = X_val.to(DEVICE, non_blocking=True)
                    Y_val = Y_val.to(DEVICE, non_blocking=True)
                    preds_val = model(X_val)
                    loss_val = loss_fn(preds_val, Y_val)
                    val_loss += loss_val.item()
            
            avg_val_loss = val_loss / len(val_loader)
            logger.info(f"  Validation Loss: {avg_val_loss:.4f}")
            launchlog.log_epoch({"Validation Loss": avg_val_loss})
            
            # Save best model as W&B artifact
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                
                # Create checkpoint dictionary
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': best_val_loss,
                }
                
                # Save checkpoint to temporary file
                temp_path = f"temp_best_model.pt"
                torch.save(checkpoint, temp_path)
                
                # Create and log W&B artifact
                artifact = wandb.Artifact(
                    name=f"model-best",
                    type="model",
                    description=f"Best model checkpoint (val_loss: {best_val_loss:.4f})"
                )
                artifact.add_file(temp_path)
                wandb.log_artifact(artifact)
                
                # Clean up temporary file
                os.remove(temp_path)
                logger.info(f"Saved best model checkpoint to W&B (val_loss: {best_val_loss:.4f})")
            
            # Periodic checkpoint saving
            if epoch % SAVE_FREQUENCY == 0:
                # Create checkpoint dictionary
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': avg_val_loss,
                }
                
                # Save checkpoint to temporary file
                temp_path = f"temp_checkpoint_epoch_{epoch}.pt"
                torch.save(checkpoint, temp_path)
                
                # Create and log W&B artifact
                artifact = wandb.Artifact(
                    name=f"model-checkpoint-epoch-{epoch}",
                    type="model",
                    description=f"Model checkpoint at epoch {epoch}"
                )
                artifact.add_file(temp_path)
                wandb.log_artifact(artifact)
                
                # Clean up temporary file
                os.remove(temp_path)
                logger.info(f"Saved periodic checkpoint to W&B (epoch {epoch})")
            
            logger.info("-" * 50)
    
    logger.info("Training Completed Successfully!")

# ============================ Main Execution ============================ #

if __name__ == "__main__":
    # Initialize logging
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger("main")
    
    # Initialize W&B and LaunchLogger
    initialize_wandb(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=WANDB_RUN_NAME,
        mode="online",
        resume="never",
    )
    LaunchLogger.initialize(use_wandb=True)
    logger.info("W&B Initialized and Logging Started.")
    
    # Create DataLoaders
    try:
        logger.info("Setting up DataLoaders...")
        train_loader, val_loader = get_arise_dataloaders(
            data_dir=DATA_DIR,
            simulation_numbers=SIM_NUMS_TRAIN,
            batch_size=ACE_BATCH_SIZE,
            train_fraction=0.8,
            num_workers=4
        )
        logger.info("DataLoaders are ready.")
    except Exception as e:
        logger.exception("Failed to create DataLoaders!")
        raise
    
    # Initialize Model, Optimizer, Scheduler, and Loss Function
    try:
        logger.info("Initializing Model, Optimizer, Scheduler, and Loss Function...")
        model = get_ace_sto_sfno(img_shape=IMG_SHAPE, in_chans=IN_CHANS, out_chans=OUT_CHANS, device=DEVICE)
        model.to(DEVICE)
        optimizer = get_ace_optimizer(model)
        scheduler = get_ace_lr_scheduler(optimizer)
        loss_fn = AceLoss()
        logger.info("Model and training components initialized successfully.")
    except Exception as e:
        logger.exception("Failed to initialize training components!")
        raise
    
    # Start Training
    try:
        train_model(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=SINGLE_SIM_EPOCHS  # Define number of epochs per simulation
        )
    except Exception as e:
        logger.exception("An error occurred during training!")
        raise
    
    logger.info("All processes completed successfully!")
