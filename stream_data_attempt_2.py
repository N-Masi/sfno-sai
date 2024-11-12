import torch
from torch.utils.data import Dataset
import xarray as xr
import s3fs
from torch.utils.data import DataLoader
import xbatcher
import pdb

fs = s3fs.S3FileSystem(anon=True)
s3_bucket = "ncar-cesm2-arise"
#s3_path = "ARISE-SAI-1.5/b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.001/atm/proc/tseries/day_1/b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.001.cam.h1.T.20350101-20441231.nc"
s3_path = "ARISE-SAI-1.5/b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.001/atm/proc/tseries/month_1/b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.001.cam.h0.T.203501-206912.nc"
s3_file_url = f"s3://{s3_bucket}/{s3_path}"

with fs.open(s3_file_url) as infile:
    print("opening")
    with xr.open_dataset(infile, engine="h5netcdf") as ds:
        temp_explorer = ds.isel(time=[0,1,2,3],lev=0)[['T']]
        #pdb.set_trace()
        bgen = xbatcher.BatchGenerator(temp_explorer, {'time': 2})
        for batch in bgen:
            pdb.set_trace()
            print(temp_explorer)
            print(temp_explorer.shape)
            break
