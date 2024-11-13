import torch
import logging
from pathlib import Path
from typing import Dict, Any
from climate_normalizier import ClimateNormalizer

def setup_logging(log_file: Path) -> logging.Logger:
    """
    Configure logging to output both to console and a log file.

    Args:
        log_file (Path): Path to the log file.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger("VerifyNormalization")
    logger.setLevel(logging.INFO)

    # Formatter for log messages
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Stream handler for console output
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # File handler for log file
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

def load_normalizer(stats_path: Path) -> ClimateNormalizer:
    """
    Load the ClimateNormalizer instance from a .pt file.

    Args:
        stats_path (Path): Path to the normalizer statistics file.

    Returns:
        ClimateNormalizer: Loaded normalizer instance.
    """
    logger.info(f"Loading ClimateNormalizer statistics from {stats_path}")
    normalizer = torch.load(stats_path, map_location='cpu')
    if not isinstance(normalizer, ClimateNormalizer):
        logger.error("Loaded object is not an instance of ClimateNormalizer.")
        raise TypeError("Invalid normalizer object.")
    logger.info("ClimateNormalizer loaded successfully.")
    return normalizer

def load_normalized_data(data_path: Path) -> Dict[str, torch.Tensor]:
    """
    Load the normalized ARISE data from a .pt file.

    Args:
        data_path (Path): Path to the normalized data file.

    Returns:
        Dict[str, torch.Tensor]: Dictionary containing normalized 'X' and 'Y' tensors.
    """
    logger.info(f"Loading normalized data from {data_path}")
    data = torch.load(data_path, map_location='cpu')
    if not isinstance(data, Dict):
        logger.error("Loaded data is not a dictionary.")
        raise ValueError("Invalid data format.")
    logger.info("Normalized data loaded successfully.")
    return data

def print_normalizer_stats(normalizer: ClimateNormalizer, output_file: Path):
    """
    Print the normalization statistics to the output file.

    Args:
        normalizer (ClimateNormalizer): The loaded normalizer instance.
        output_file (Path): Path to the output text file.
    """
    logger.info(f"Printing normalizer statistics to {output_file}")
    with open(output_file, 'a') as f:
        f.write("=== ClimateNormalizer Statistics ===\n\n")
        for var_name, stats in normalizer.stats.items():
            f.write(f"Variable: {var_name}\n")
            for stat_key, stat_value in stats.items():
                # Detach and convert to list for better readability
                stat_list = stat_value.detach().cpu().numpy().tolist()
                f.write(f"  {stat_key}: {stat_list}\n")
            f.write("\n")
    logger.info("Normalizer statistics printed successfully.")

def calculate_and_print_statistics(
    data: Dict[str, torch.Tensor],
    normalizer: ClimateNormalizer,
    output_file: Path
):
    """
    Calculate statistics on the normalized data and print them to the output file.

    Args:
        data (Dict[str, torch.Tensor]): Dictionary containing normalized 'X' and 'Y' tensors.
        normalizer (ClimateNormalizer): The loaded normalizer instance.
        output_file (Path): Path to the output text file.
    """
    logger.info("Calculating statistics on normalized data.")
    with open(output_file, 'a') as f:
        f.write("=== Statistics on Normalized Data ===\n\n")
        for key in ['X', 'Y']:
            if key not in data:
                logger.warning(f"Key '{key}' not found in data. Skipping.")
                continue
            tensor = data[key]
            logger.info(f"Calculating statistics for '{key}'")
            if isinstance(tensor, Dict):
                for var_name, var_tensor in tensor.items():
                    f.write(f"Variable: {var_name} ({key})\n")
                    mean = var_tensor.mean().item()
                    std = var_tensor.std().item()
                    f.write(f"  Mean: {mean:.6f}\n")
                    f.write(f"  Std Dev: {std:.6f}\n\n")
            else:
                f.write(f"Variable: {key}\n")
                mean = tensor.mean().item()
                std = tensor.std().item()
                f.write(f"  Mean: {mean:.6f}\n")
                f.write(f"  Std Dev: {std:.6f}\n\n")
    logger.info("Statistics on normalized data calculated and printed successfully.")

def main():
    """
    Main function to verify normalization statistics.

    Plan:
        1. Setup logging.
        2. Define input and output file paths.
        3. Load the ClimateNormalizer statistics.
        4. Load the normalized data.
        5. Print normalizer statistics to a .txt file.
        6. Calculate and print statistics on the normalized data.
        7. Ensure all outputs are well-formatted in the .txt file.

    Inputs:
        - Normalizer statistics path: /teamspace/studios/this_studio/arise_processed_data/climate_normalizer_stats.pt
        - Normalized data path: /teamspace/studios/this_studio/arise_processed_data/arise_001_processed_norm.pt

    Outputs:
        - Verification report path: /teamspace/studios/this_studio/arise_processed_data/normalization_verification_report.txt
    """
    try:
        # Step 1: Setup logging
        log_file = Path("/teamspace/studios/this_studio/arise_processed_data/verification.log")
        global logger
        logger = setup_logging(log_file)
        logger.info("Starting normalization verification process.")

        # Step 2: Define file paths
        data_dir = Path("/teamspace/studios/this_studio/arise_processed_data")
        stats_path = data_dir / "climate_normalizer_stats.pt"
        normalized_data_path = data_dir / "arise_001_processed_norm.pt"
        report_path = data_dir / "normalization_verification_report.txt"

        # Clear the report file if it exists
        if report_path.exists():
            report_path.unlink()
            logger.info(f"Existing report file {report_path} deleted.")

        # Step 3: Load the ClimateNormalizer statistics
        normalizer = load_normalizer(stats_path)

        # Step 4: Load the normalized data
        normalized_data = load_normalized_data(normalized_data_path)

        # Step 5: Print normalizer statistics to the report
        print_normalizer_stats(normalizer, report_path)

        # Step 6: Calculate and print statistics on the normalized data
        calculate_and_print_statistics(normalized_data, normalizer, report_path)

        logger.info(f"Normalization verification report saved to {report_path}")
        logger.info("Normalization verification process completed successfully.")

    except Exception as e:
        logger.error("An error occurred during the normalization verification process.", exc_info=True)
        raise

if __name__ == "__main__":
    main()