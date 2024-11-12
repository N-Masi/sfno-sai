import torch
from torch.utils.data import Dataset
import xarray as xr
import s3fs
from torch.utils.data import DataLoader
import xbatcher
import pdb
from ace_helpers import *

fs = s3fs.S3FileSystem(anon=True)
s3_bucket = "ncar-cesm2-arise"

# tuples of (path to monthly data for variable, whether different vertical levels are needed, forcing-only?)
s3_path_T = ("ARISE-SAI-1.5/b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.001/atm/proc/tseries/month_1/b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.001.cam.h0.T.203501-206912.nc", True, False) # temperature
s3_path_Q = ("ARISE-SAI-1.5/b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.001/atm/proc/tseries/month_1/b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.001.cam.h0.Q.203501-206912.nc", True, False) # specific humidity
s3_path_PS = ("ARISE-SAI-1.5/b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.001/atm/proc/tseries/month_1/b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.001.cam.h0.PS.203501-206912.nc", False, False) # surface pressure
# TODO: get other variables (windspeed & AOD)
variable_paths = [s3_path_T, s3_path_Q, s3_path_PS]

vert_level_indices = [24, 36, 39, 41, 44, 46, 49, 52, 56, 59, 62, 69]
''' Indices into the 70 vertical levels of the lev dimension.
Different vertical levels used for, among others, T & Q.
Index to corresponding pressure (hPa):
    24: 0.5534699885174632
    36: 10.70705009624362
    39: 20.056750625371933
    41: 29.7279991209507
    44: 51.67749896645546
    46: 73.75095784664154
    49: 121.54724076390266
    52: 197.9080867022276
    56: 379.10090386867523
    59: 609.7786948084831
    62: 820.8583686500788
    69: 992.556095123291
'''

# outer nested loops: for each epoch, for each simulation 001-009:
    # instantiate X & Y as empty torch.utils.data.Dataset | Y is shifted one timestep ahead of X
    # for variable in variable paths
        # open s3 file
        # if varialbe[1]: 
            # for each vertical level, append with shape (all_time_steps, 1, 192, 288) to X & Y (not to Y if forcing-only) [have to reshape both to switch time & lev]
        # else:
            # append to X & Y (not to Y if forcing-only)
    # make dataloaders of X & Y with batch_size ACE_BATCH_SIZE=4 | rough estimates that X & Y will each be 5.3Gb, should be doable on OSCAR
    # train for this data, logging to w&b
