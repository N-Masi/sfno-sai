if __name__ == "__main__":
    """
    Process all ARISE simulations (001-009)
    """
    # Set up parameters
    batch_size = 4
    output_dir = "./arise_processed_data"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Process simulations 001-009
    for sim_num in range(1, 10):
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