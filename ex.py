import subprocess
import logging
from pathlib import Path
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from typing import List, Dict, Optional
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class S3BucketExplorer:
    """Explorer class for NCAR-CESM2-ARISE S3 bucket"""
    
    def __init__(self, bucket_name: str = 'ncar-cesm2-arise'):
        self.logger = logging.getLogger(__name__)
        self.bucket_name = bucket_name
        self.output_dir = Path('bucket_contents')
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize S3 client
        self.s3_client = boto3.client(
            's3',
            config=Config(signature_version=UNSIGNED)
        )
    
    def explore_with_aws_cli(self) -> None:
        """
        Use AWS CLI to explore bucket contents
        Saves output to aws_cli_listing.txt
        """
        try:
            output_file = self.output_dir / 'aws_cli_listing.txt'
            
            # Run AWS CLI command and capture output
            command = f"aws s3 ls --no-sign-request s3://{self.bucket_name}/ --recursive"
            self.logger.info(f"Running command: {command}")
            
            result = subprocess.run(
                command.split(),
                capture_output=True,
                text=True
            )
            
            # Write output to file
            with open(output_file, 'w') as f:
                f.write(result.stdout)
                
            self.logger.info(f"AWS CLI listing saved to {output_file}")
            
        except Exception as e:
            self.logger.error(f"Error in AWS CLI exploration: {str(e)}")
            raise
    
    def explore_with_boto3(self, prefix: str = '') -> None:
        """
        Use boto3 to explore bucket contents
        Saves hierarchical structure to json file
        
        Args:
            prefix: Optional prefix to filter bucket contents
        """
        try:
            structure = self._build_directory_structure(prefix)
            
            output_file = self.output_dir / 'bucket_structure.json'
            with open(output_file, 'w') as f:
                json.dump(structure, f, indent=2)
                
            self.logger.info(f"Bucket structure saved to {output_file}")
            
            # Also save a simplified listing
            self._save_simplified_listing(structure)
            
        except Exception as e:
            self.logger.error(f"Error in boto3 exploration: {str(e)}")
            raise
    
    def _build_directory_structure(self, prefix: str) -> Dict:
        """Build hierarchical directory structure from bucket contents"""
        structure = {}
        paginator = self.s3_client.get_paginator('list_objects_v2')
        
        for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
            for obj in page.get('Contents', []):
                path_parts = obj['Key'].split('/')
                current_dict = structure
                
                # Build nested dictionary structure
                for part in path_parts[:-1]:
                    if part not in current_dict:
                        current_dict[part] = {}
                    current_dict = current_dict[part]
                    
                # Add file to structure
                if path_parts[-1]:  # If not empty string
                    current_dict[path_parts[-1]] = obj['Size']
        
        return structure
    
    def _save_simplified_listing(self, structure: Dict) -> None:
        """Save a simplified, hierarchical listing of bucket contents"""
        output_file = self.output_dir / 'simplified_listing.txt'
        
        def write_structure(d: Dict, file, indent: int = 0):
            for key, value in sorted(d.items()):
                if isinstance(value, dict):
                    file.write('  ' * indent + f'/{key}\n')
                    write_structure(value, file, indent + 1)
                else:
                    file.write('  ' * indent + f'{key} ({value} bytes)\n')
        
        with open(output_file, 'w') as f:
            write_structure(structure, f)
            
        self.logger.info(f"Simplified listing saved to {output_file}")
    
    def analyze_file_types(self) -> None:
        """Analyze and summarize file types in the bucket"""
        try:
            file_types = {}
            paginator = self.s3_client.get_paginator('list_objects_v2')
            
            for page in paginator.paginate(Bucket=self.bucket_name):
                for obj in page.get('Contents', []):
                    ext = Path(obj['Key']).suffix
                    if ext:
                        file_types[ext] = file_types.get(ext, 0) + 1
            
            # Save analysis
            output_file = self.output_dir / 'file_type_analysis.txt'
            with open(output_file, 'w') as f:
                f.write("File Type Analysis:\n")
                for ext, count in sorted(file_types.items(), key=lambda x: x[1], reverse=True):
                    f.write(f"{ext}: {count} files\n")
                    
            self.logger.info(f"File type analysis saved to {output_file}")
            
        except Exception as e:
            self.logger.error(f"Error in file type analysis: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    explorer = S3BucketExplorer()
    
    # Explore using AWS CLI
    explorer.explore_with_aws_cli()
    
    # Explore using boto3
    explorer.explore_with_boto3()
    
    # Analyze file types
    explorer.analyze_file_types()