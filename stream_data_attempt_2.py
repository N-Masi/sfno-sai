import torch
from torch.utils.data import Dataset
import xarray as xr
import s3fs
from torch.utils.data import DataLoader
import xbatcher
import pdb

fs = s3fs.S3FileSystem()
s3_bucket = "ncar-cesm2-arise"
s3_path = "ARISE-SAI-1.5/b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.001/atm/proc/tseries/day_1/b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.001.cam.h1.T.20350101-20441231.nc"
s3_file_url = f"{s3_bucket}/{s3_path}"

with fs.open(s3_file_url) as infile:
    print("opening")
    with xr.open_dataset(infile, engine="h5netcdf") as ds:
        pdb.set_trace()
        bgen = xbatcher.BatchGenerator(ds, {'time': 4})
        for batch in bgen:
            pdb.set_trace()
            break
