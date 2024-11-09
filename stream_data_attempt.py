import torch
from torch.utils.data import Dataset
import xarray as xr
import s3fs
from torch.utils.data import DataLoader
import pdb

class NetCDFS3Dataset(Dataset):
    def __init__(self, s3_bucket, s3_path, variables, transform=None):
        """
        PyTorch Dataset for loading NetCDF data from an S3 bucket on the fly.
        
        Parameters:
        - s3_bucket: The name of the S3 bucket.
        - s3_path: The path to the NetCDF file in the S3 bucket.
        - variables: List of variable names to load from the NetCDF file.
        - transform: Any PyTorch transformations to apply on each sample.
        """
        self.s3_bucket = s3_bucket
        self.s3_path = s3_path
        self.variables = variables
        self.transform = transform
        self.s3_fs = s3fs.S3FileSystem(anon=False)
        
        # Open the dataset (using lazy loading to avoid loading into memory)
        s3_file_url = f"{s3_bucket}/{s3_path}"
        self.dataset = xr.open_dataset(s3_file_url, engine="h5netcdf", chunks="auto")
        
        # Assume the data dimension ordering is (time, level, height, width)
        # Adjust these depending on the structure of your NetCDF file
        self.time_steps = self.dataset.dims["time"]

    def __len__(self):
        return self.time_steps
    
    def __getitem__(self, idx):
        # Load data lazily for the selected time step across all variables
        pdb.set_trace()
        data = self.dataset.isel(time=idx)[self.variables].to_array().values

        # Transform or normalize if needed
        if self.transform:
            data = self.transform(data)

        # Convert to torch tensor
        return torch.tensor(data, dtype=torch.float32)

    def close(self):
        """Closes the NetCDF dataset to free up resources."""
        self.dataset.close()

# Define dataset parameters
s3_bucket = "ncar-cesm2-arise" #"https://ncar-cesm2-arise.s3.amazonaws.com"
s3_path = "ARISE-SAI-1.5/b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.001/atm/proc/tseries/day_1/b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.001.cam.h1.T.20350101-20441231.nc"
variables = ["temperature", "pressure", "humidity"]

fs = s3fs.S3FileSystem(anon=True)
aws_url = f"s3://{s3_bucket}/{s3_path}"

with fs.open(aws_url) as fileObj:
  ds = xr.open_dataset(fileObj, engine='h5netcdf')

# Create the dataset
# dataset = NetCDFS3Dataset(s3_bucket=s3_bucket, s3_path=s3_path, variables=variables)

# # Use DataLoader to iterate over the dataset in batches
batch_size = 4  # Adjust based on memory and model requirements
data_loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

# # Example of iterating over batches in training loop
for epoch in range(1):
    for batch in data_loader:
        # Your training loop here, e.g.:
        # outputs = model(batch)
        # loss = loss_fn(outputs, targets)
        # loss.backward()
        # optimizer.step()
        pass