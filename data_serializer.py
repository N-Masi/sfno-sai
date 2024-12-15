import torch
from torch.utils.data import TensorDataset, DataLoader
import xarray as xr
import s3fs
from ace_helpers import *
from climate_normalizer import ClimateNormalizer
import wandb
import os
import pdb
import random
import numpy as np
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--forcing_vars", type=str, nargs="*", default=["AODVISstdn", "SOLIN"], help="forcing (input-only) variables to use")
parser.add_argument("-p", "--prognostic_vars", type=str, nargs="*", default=["T", "Q", "U", "V", "PS", "TS"], help="prognostic (input & output) variables to use")
parser.add_argument("-d", "--diagnostic_vars", type=str, nargs="*", default=["SHFLX", "LHFLX", "PRECT"], help="diagnostic (output-only) variables to use")
args = parser.parse_args()

fs = s3fs.S3FileSystem(anon=True)
s3_bucket = "ncar-cesm2-arise"
s3_path_chunk_1 = "ARISE-SAI-1.5/b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT."
s3_path_chunk_2 = "/atm/proc/tseries/month_1/b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT."

variables = []
# tuples of (path to monthly data for variable, variable name, whether different vertical levels are needed, variable mode)
# input only (forcings):
if "AODVISstdn" in args.forcing_vars:
    variables.append((".cam.h0.AODVISstdn.203501-206912.nc", "AODVISstdn", False, "forcing")) # this is the signal of the aerosol injections by controller
if "SST" in args.forcing_vars:
    variables.append((".cam.h0.SST.203501-206912.nc", "SST", False, "forcing")) # sea surface/skin temp (different from TS)
if "SOLIN"  in args.forcing_vars:
    variables.append((".cam.h0.SOLIN.203501-206912.nc", "SOLIN", False, "forcing")) # downward shortwave radiative flux at TOA
print(f"Forcing variables being used: {args.forcing_vars}")
# input & output:
if "T" in args.prognostic_vars:
    variables.append((".cam.h0.T.203501-206912.nc", "T", True, "prognostic")) # temperature
if "Q" in args.prognostic_vars:
    variables.append((".cam.h0.Q.203501-206912.nc", "Q", True, "prognostic")) # specific humidity
if "U" in args.prognostic_vars:
    variables.append((".cam.h0.U.203501-206912.nc", "U", True, "prognostic")) # zonal wind
if "V" in args.prognostic_vars:
    variables.append((".cam.h0.V.203501-206912.nc", "V", True, "prognostic")) # meridonal wind
if "PS" in args.prognostic_vars:
    variables.append((".cam.h0.PS.203501-206912.nc", "PS", False, "prognostic")) # surface pressure
if "TS" in args.prognostic_vars:
    variables.append((".cam.h0.TS.203501-206912.nc", "TS", False, "prognostic")) # surface temperature of land or sea-ice (radiative)
print(f"Prognostic variables being used: {args.prognostic_vars}")
# output only:
if "SHFLX" in args.diagnostic_vars:
    variables.append((".cam.h0.SHFLX.203501-206912.nc", "SHFLX", False, "diagnostic")) # surface sensible heat flux
if "LHFLX" in args.diagnostic_vars:
    variables.append((".cam.h0.LHFLX.203501-206912.nc", "LHFLX", False, "diagnostic")) # surfact latent  heat flux
if "PRECT" in args.diagnostic_vars:
    variables.append((".cam.h0.PRECT.203501-206912.nc", "PRECT", False, "diagnostic")) # precipitation?
print(f"Diagnostic variables being used: {args.diagnostic_vars}")

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

ENSEMBLE = ["001"]
normalizer = ClimateNormalizer()

data = torch.empty(0)
data_to_norm = {}
for sim_num in ENSEMBLE:
    print(f"Starting member {sim_num}")
    for var_path, var_name, vert_levels, variable_mode in variables:
        s3_file_url = f"s3://{s3_bucket}/{s3_path_chunk_1}{sim_num}{s3_path_chunk_2}{sim_num}{var_path}"
        # open s3 file
        with fs.open(s3_file_url) as varfile:
            with xr.open_dataset(varfile, engine="h5netcdf") as ds: # ignore error on lightning.ai
                # for each vertical level, append with shape (all_time_steps, 1, 192, 288) to X & Y (not to Y if forcing-only) [have to reshape both to switch time & lev]
                if vert_levels:
                    data_to_norm[var_name] = torch.from_numpy(ds[[var_name]].isel(lev=vert_level_indices).to_array().values).reshape(-1, len(vert_level_indices), 192, 288)
                else:
                    data_to_norm[var_name] = torch.from_numpy(ds[[var_name]].to_array().values).reshape(-1, 1, 192, 288)
        formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"Done loading in {var_name} @ {formatted_time}")
    normalizer.fit_multiple(data_to_norm)
    print("Done fitting normalizer")
    for var_path, var_name, vert_levels, variable_mode in variables:
        normed_data = normalizer.normalize(data_to_norm[var_name], var_name, 'residual')
        data = torch.concat((data, normed_data), dim=1)
        data_to_norm[var_name] = None # clear some memory
        print(f"Done normalizing {var_name}")
    data_to_norm = {} # clear some memory
    print(f"Done loading in all data for member {sim_num}, tensor shape: {data.shape}")
    filepath = f"data/normed_data_{data.shape[1]}_chans_{sim_num}.pt"
    torch.save(data, filepath)
    formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"Done serializing {sim_num} to {filepath} @ {formatted_time}")
print("All done!")
