import torch
import logging
from pathlib import Path
from typing import Dict
from climate_normalizer import ClimateNormalizer

def setup_logging():
    """Configure logging for the normalization process."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_processed_data(data_path: Path) -> Dict[str, torch.Tensor]:
    """
    Load the processed ARISE data from a .pt file.

    Args:
        data_path (Path): Path to the processed data file.

    Returns:
        Dict[str, torch.Tensor]: Dictionary containing 'X' and 'Y' tensors.
    """
    logger.info(f"Loading processed data from {data_path}")
    data = torch.load(data_path, map_location='cpu')
    return data

def save_normalized_data(normalized_data: Dict[str, torch.Tensor], output_path: Path):
    """
    Save the normalized data to a .pt file.

    Args:
        normalized_data (Dict[str, torch.Tensor]): Dictionary containing normalized 'X' and 'Y' tensors.
        output_path (Path): Path to save the normalized data.
    """
    logger.info(f"Saving normalized data to {output_path}")
    torch.save(normalized_data, output_path)
    logger.info("Normalized data saved successfully.")

def save_normalizer(normalizer: ClimateNormalizer, stats_path: Path):
    """
    Save the ClimateNormalizer instance to a file for future use.

    Args:
        normalizer (ClimateNormalizer): The ClimateNormalizer instance with fitted statistics.
        stats_path (Path): Path to save the normalizer statistics.
    """
    logger.info(f"Saving ClimateNormalizer statistics to {stats_path}")
    torch.save(normalizer, stats_path)
    logger.info("ClimateNormalizer statistics saved successfully.")

def normalize_dataset(data: Dict[str, torch.Tensor], normalizer: ClimateNormalizer) -> Dict[str, torch.Tensor]:
    """
    Normalize the dataset using the provided ClimateNormalizer.

    Args:
        data (Dict[str, torch.Tensor]): Dictionary containing 'X' and 'Y' tensors.
        normalizer (ClimateNormalizer): Fitted ClimateNormalizer instance.

    Returns:
        Dict[str, torch.Tensor]: Dictionary containing normalized 'X' and 'Y' tensors.
    """
    logger.info("Starting normalization of the dataset...")
    
    normalized_X = {}
    normalized_Y = {}
    
    # Assuming 'X' and 'Y' contain multiple variables
    # Modify this section based on the actual structure of 'X' and 'Y'
    if isinstance(data['X'], dict) and isinstance(data['Y'], dict):
        logger.info("Detected multiple variables in 'X' and 'Y'. Normalizing each variable individually.")
        for var_name, tensor in data['X'].items():
            normalized_X[var_name] = normalizer.normalize(tensor, var_name, mode='residual')
        for var_name, tensor in data['Y'].items():
            normalized_Y[var_name] = normalizer.normalize(tensor, var_name, mode='residual')
    else:
        logger.info("Detected single variable in 'X' and 'Y'. Normalizing directly.")
        normalized_X = normalizer.normalize(data['X'], 'X', mode='residual')
        normalized_Y = normalizer.normalize(data['Y'], 'Y', mode='residual')
    
    logger.info("Normalization completed.")
    return {'X': normalized_X, 'Y': normalized_Y}

def main():
    """
    Main function to perform normalization on the ARISE processed data.
    
    Plan:
        1. Setup logging.
        2. Define input and output file paths.
        3. Load the processed data.
        4. Initialize and fit the ClimateNormalizer.
        5. Normalize the data.
        6. Save the normalized data and normalizer statistics.
    
    Inputs:
        - Input data path: /teamspace/studios/this_studio/arise_processed_data/arise_001_processed.pt
    
    Outputs:
        - Normalized data path: /teamspace/studios/this_studio/arise_processed_data/arise_001_processed_norm.pt
        - Normalizer statistics path: /teamspace/studios/this_studio/arise_processed_data/climate_normalizer_stats.pt
    """
    try:
        # Step 1: Setup logging
        global logger
        logger = setup_logging()
        logger.info("Starting data normalization process.")
        
        # Step 2: Define file paths
        data_dir = Path("/teamspace/studios/this_studio/arise_processed_data")
        input_data_path = data_dir / "arise_001_processed.pt"
        output_data_path = data_dir / "arise_001_processed_norm.pt"
        normalizer_stats_path = data_dir / "climate_normalizer_stats.pt"
        
        # Step 3: Load the processed data
        data = load_processed_data(input_data_path)
        
        # Step 4: Initialize and fit the ClimateNormalizer
        normalizer = ClimateNormalizer()
        
        # Prepare variables dictionary for fitting
        variables = {}
        if isinstance(data['X'], dict):
            variables.update(data['X'])
        else:
            variables['X'] = data['X']
        
        if isinstance(data['Y'], dict):
            variables.update(data['Y'])
        else:
            variables['Y'] = data['Y']
        
        logger.info("Fitting the ClimateNormalizer with the dataset.")
        normalizer.fit_multiple(variables)
        logger.info("ClimateNormalizer fitting completed.")
        
        # Step 5: Normalize the data
        normalized_data = normalize_dataset(data, normalizer)
        
        # Step 6: Save the normalized data and normalizer statistics
        save_normalized_data(normalized_data, output_data_path)
        save_normalizer(normalizer, normalizer_stats_path)
        
        logger.info("Data normalization process completed successfully.")
    
    except Exception as e:
        logger.error("An error occurred during the normalization process.", exc_info=True)
        raise

if __name__ == "__main__":
    main()