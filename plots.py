import torch
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
import traceback

def setup_logging(log_file: Path) -> logging.Logger:
    """Configure logging to output both to console and file."""
    logger = logging.getLogger("NormalizedDataPlotter")
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

def load_normalized_data(data_path: Path) -> Dict[str, torch.Tensor]:
    """Load the normalized data from disk."""
    logger.info(f"Loading normalized data from {data_path}")
    data = torch.load(data_path, map_location='cpu')
    return data

def create_distribution_plots(data: Dict[str, torch.Tensor], output_dir: Path):
    """Create distribution plots for each variable."""
    try:
        logger.info("Creating distribution plots...")
        
        for key in ['X', 'Y']:
            if isinstance(data[key], dict):
                for var_name, tensor in data[key].items():
                    plt.figure(figsize=(12, 6))
                    
                    # Flatten the tensor for histogram
                    values = tensor.flatten().numpy()
                    
                    # Create histogram with KDE
                    sns.histplot(values, kde=True)
                    plt.title(f'Distribution of {key}-{var_name}')
                    plt.xlabel('Value')
                    plt.ylabel('Count')
                    
                    # Add statistical annotations
                    plt.axvline(np.mean(values), color='r', linestyle='--', label=f'Mean: {np.mean(values):.3f}')
                    plt.axvline(np.median(values), color='g', linestyle='--', label=f'Median: {np.median(values):.3f}')
                    plt.legend()
                    
                    # Save plot
                    plt.savefig(output_dir / f'distribution_{key}_{var_name}.png', dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    logger.info(f"Created distribution plot for {key}-{var_name}")
    
    except Exception as e:
        logger.error(f"Error in create_distribution_plots: {str(e)}")
        logger.error(traceback.format_exc())

def create_temporal_statistics_plots(data: Dict[str, torch.Tensor], output_dir: Path):
    """Create plots showing temporal statistics."""
    try:
        logger.info("Creating temporal statistics plots...")
        
        for key in ['X', 'Y']:
            if isinstance(data[key], dict):
                for var_name, tensor in data[key].items():
                    plt.figure(figsize=(15, 10))
                    
                    # Calculate temporal statistics
                    temporal_mean = torch.mean(tensor, dim=(1,2,3) if len(tensor.shape) == 4 else (1,2)).numpy()
                    temporal_std = torch.std(tensor, dim=(1,2,3) if len(tensor.shape) == 4 else (1,2)).numpy()
                    
                    # Create subplot for mean
                    plt.subplot(2, 1, 1)
                    plt.plot(temporal_mean, label='Mean')
                    plt.title(f'Temporal Mean of {key}-{var_name}')
                    plt.xlabel('Time Step')
                    plt.ylabel('Mean Value')
                    plt.grid(True)
                    
                    # Create subplot for standard deviation
                    plt.subplot(2, 1, 2)
                    plt.plot(temporal_std, label='Std Dev', color='orange')
                    plt.title(f'Temporal Standard Deviation of {key}-{var_name}')
                    plt.xlabel('Time Step')
                    plt.ylabel('Std Dev Value')
                    plt.grid(True)
                    
                    plt.tight_layout()
                    plt.savefig(output_dir / f'temporal_stats_{key}_{var_name}.png', dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    logger.info(f"Created temporal statistics plot for {key}-{var_name}")
    
    except Exception as e:
        logger.error(f"Error in create_temporal_statistics_plots: {str(e)}")
        logger.error(traceback.format_exc())

def create_spatial_correlation_plots(data: Dict[str, torch.Tensor], output_dir: Path):
    """Create spatial correlation plots."""
    try:
        logger.info("Creating spatial correlation plots...")
        
        for key in ['X', 'Y']:
            if isinstance(data[key], dict):
                for var_name, tensor in data[key].items():
                    # Take a sample time step
                    sample_time = tensor[0]
                    
                    if len(sample_time.shape) == 3:  # Has levels
                        for level in range(sample_time.shape[0]):
                            plt.figure(figsize=(10, 8))
                            plt.imshow(sample_time[level], cmap='RdBu_r')
                            plt.colorbar(label='Value')
                            plt.title(f'Spatial Pattern {key}-{var_name} (Level {level})')
                            plt.savefig(output_dir / f'spatial_pattern_{key}_{var_name}_level_{level}.png', 
                                      dpi=300, bbox_inches='tight')
                            plt.close()
                    else:
                        plt.figure(figsize=(10, 8))
                        plt.imshow(sample_time, cmap='RdBu_r')
                        plt.colorbar(label='Value')
                        plt.title(f'Spatial Pattern {key}-{var_name}')
                        plt.savefig(output_dir / f'spatial_pattern_{key}_{var_name}.png', 
                                  dpi=300, bbox_inches='tight')
                        plt.close()
                    
                    logger.info(f"Created spatial correlation plot for {key}-{var_name}")
    
    except Exception as e:
        logger.error(f"Error in create_spatial_correlation_plots: {str(e)}")
        logger.error(traceback.format_exc())

def create_level_correlation_plots(data: Dict[str, torch.Tensor], output_dir: Path):
    """Create correlation plots between levels for 3D variables."""
    try:
        logger.info("Creating level correlation plots...")
        
        for key in ['X', 'Y']:
            if isinstance(data[key], dict):
                for var_name, tensor in data[key].items():
                    if len(tensor.shape) == 4:  # Only for 3D variables
                        # Calculate correlation between levels
                        n_levels = tensor.shape[1]
                        corr_matrix = torch.zeros((n_levels, n_levels))
                        
                        for i in range(n_levels):
                            for j in range(n_levels):
                                # Flatten spatial dimensions and calculate correlation
                                level_i = tensor[:, i].flatten()
                                level_j = tensor[:, j].flatten()
                                corr = torch.corrcoef(torch.stack([level_i, level_j]))[0,1]
                                corr_matrix[i,j] = corr
                        
                        plt.figure(figsize=(10, 8))
                        sns.heatmap(corr_matrix.numpy(), cmap='RdBu_r', center=0, vmin=-1, vmax=1)
                        plt.title(f'Level Correlation Matrix {key}-{var_name}')
                        plt.xlabel('Level')
                        plt.ylabel('Level')
                        plt.savefig(output_dir / f'level_correlation_{key}_{var_name}.png', 
                                  dpi=300, bbox_inches='tight')
                        plt.close()
                        
                        logger.info(f"Created level correlation plot for {key}-{var_name}")
    
    except Exception as e:
        logger.error(f"Error in create_level_correlation_plots: {str(e)}")
        logger.error(traceback.format_exc())

def create_summary_statistics(data: Dict[str, torch.Tensor], output_dir: Path):
    """Create and save summary statistics."""
    try:
        logger.info("Creating summary statistics...")
        
        with open(output_dir / 'summary_statistics.txt', 'w') as f:
            f.write("=== Summary Statistics ===\n\n")
            
            for key in ['X', 'Y']:
                if isinstance(data[key], dict):
                    for var_name, tensor in data[key].items():
                        f.write(f"\nVariable: {key}-{var_name}\n")
                        f.write(f"Shape: {tensor.shape}\n")
                        f.write(f"Mean: {tensor.mean().item():.6f}\n")
                        f.write(f"Std Dev: {tensor.std().item():.6f}\n")
                        f.write(f"Min: {tensor.min().item():.6f}\n")
                        f.write(f"Max: {tensor.max().item():.6f}\n")
                        f.write(f"25th percentile: {torch.quantile(tensor.flatten(), 0.25).item():.6f}\n")
                        f.write(f"Median: {torch.median(tensor.flatten()).item():.6f}\n")
                        f.write(f"75th percentile: {torch.quantile(tensor.flatten(), 0.75).item():.6f}\n")
                        f.write("-" * 50 + "\n")
        
        logger.info("Created summary statistics file")
    
    except Exception as e:
        logger.error(f"Error in create_summary_statistics: {str(e)}")
        logger.error(traceback.format_exc())

def print_created_files(output_dir: Path):
    """Print all files created during the visualization process."""
    try:
        logger.info("Files created during visualization:")
        for file_path in output_dir.glob('*'):
            logger.info(f"  - {file_path.name}")
            
        # Also save the list to a file
        with open(output_dir / 'file_inventory.txt', 'w') as f:
            f.write("=== Created Files ===\n\n")
            for file_path in sorted(output_dir.glob('*')):
                f.write(f"{file_path.name}\n")
                
    except Exception as e:
        logger.error(f"Error in print_created_files: {str(e)}")
        logger.error(traceback.format_exc())

def main():
    """
    Main function to create and save various plots and statistics for normalized data.
    
    Inputs:
        - Normalized data path: /teamspace/studios/this_studio/arise_processed_data/arise_001_processed_norm.pt
    
    Outputs:
        - Various plots and statistics saved to: /teamspace/studios/this_studio/arise_processed_data/visualization/
    """
    try:
        # Setup paths
        data_dir = Path("/teamspace/studios/this_studio/arise_processed_data")
        input_path = data_dir / "arise_001_processed_norm.pt"
        output_dir = data_dir / "visualization"
        log_file = output_dir / "plotting.log"
        
        # Create output directory
        output_dir.mkdir(exist_ok=True)
        
        # Setup logging
        global logger
        logger = setup_logging(log_file)
        logger.info("Starting visualization process...")
        
        # Load data
        data = load_normalized_data(input_path)
        
        # Create various plots and statistics
        create_distribution_plots(data, output_dir)
        create_temporal_statistics_plots(data, output_dir)
        create_spatial_correlation_plots(data, output_dir)
        create_level_correlation_plots(data, output_dir)
        create_summary_statistics(data, output_dir)
        
        # Add this line to print all created files
        print_created_files(output_dir)
        
        logger.info("Visualization process completed successfully.")
        
    except Exception as e:
        logger.error("An error occurred during the visualization process.")
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()