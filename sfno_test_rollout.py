import torch
from torch.utils.data import TensorDataset, DataLoader
import xarray as xr
import s3fs
from ace_helpers import *
from climate_normalizer import ClimateNormalizer
from modulus.launch.logging import LaunchLogger, PythonLogger, initialize_wandb
import wandb
import os
import pdb
import random
import numpy as np
import argparse
import wandb

parser = argparse.ArgumentParser()
parser.add_argument("run_name", type=str, help="name of the run/script as it will appear in w&b")
parser.add_argument("model_dir", type=str, help="directory to saved checkpoint (.pt) of trained model to use")
parser.add_argument("-s", "--seed", type=int, default=2952, help="randomizing seed")
parser.add_argument("-c", "--device", default="cuda", choices=["cuda", "cpu"], help="device to run on")
parser.add_argument("-D", "--dataset", default="arise", choices=["arise"], help="dataset to use")
parser.add_argument("-f", "--forcing_vars", type=str, nargs="*", default=["AODVISstdn", "SOLIN"], help="forcing (input-only) variables to use")
parser.add_argument("-p", "--prognostic_vars", type=str, nargs="*", default=["T", "Q", "U", "V", "PS", "TS"], help="prognostic (input & output) variables to use")
parser.add_argument("-d", "--diagnostic_vars", type=str, nargs="*", default=["SHFLX", "LHFLX", "PRECT"], help="diagnostic (output-only) variables to use")
parser.add_argument("-i", "--in_chans", type=int, default=52, help="# of in channels for the SFNO = #(forcing channels) + #(prognostic channels); 1 channel per variable per vertical level")
parser.add_argument("-o", "--out_chans", type=int, default=53, help="# of out channels for the SFNO = #(diagnostic channels) + #(prognostic channels); 1 channel per variable per vertical level")
parser.add_argument("-m", "--scale_factor", type=int, default=1, help="scale_factor in SFNO model, higher scale_factor multiplicatively decreases the threshold of frequency modes kept after spherical harmonic transform")
parser.add_argument("-r", "--drop_rate", type=float, default=0, help="dropout rate applied to SFNO encoder and the SFNO sFourier layer MLPs")
parser.add_argument("-T", "--test_members", type=str, nargs="*", default=["004"], help="ensemble members to use for testing")
args = parser.parse_args()

random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
DEVICE=args.device

# logging for w&b
logger = PythonLogger("main")
initialize_wandb(
    project="SFNO_test",
    entity="nickmasi",
    name='TEST:'+args.run_name,
    mode="online",
    resume="never", # use this to separate runs?
)
LaunchLogger.initialize(use_wandb=True)
logger.info("Starting up")
logger.info("Task: TEST --- full rollout of predictions w/o teacher forcing")
logger.info(f"RUNNING: {args.run_name}")

# connection to s3 for streaming data
fs = s3fs.S3FileSystem(anon=True)
if args.dataset == "arise":
    s3_bucket = "ncar-cesm2-arise"
    s3_path_chunk_1 = "ARISE-SAI-1.5/b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT."
    s3_path_chunk_2 = "/atm/proc/tseries/month_1/b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT."
    logger.info("Using ARISE-SAI-1.5 dataset")

variables = []
# tuples of (path to monthly data for variable, variable name, whether different vertical levels are needed, variable mode)
# input only (forcings):
if "AODVISstdn" in args.forcing_vars:
    variables.append((".cam.h0.AODVISstdn.203501-206912.nc", "AODVISstdn", False, "forcing")) # this is the signal of the aerosol injections by controller
if "SST" in args.forcing_vars:
    variables.append((".cam.h0.SST.203501-206912.nc", "SST", False, "forcing")) # sea surface/skin temp (different from TS)
if "SOLIN"  in args.forcing_vars:
    variables.append((".cam.h0.SOLIN.203501-206912.nc", "SOLIN", False, "forcing")) # downward shortwave radiative flux at TOA
logger.info(f"Forcing variables being used: {args.forcing_vars}")
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
logger.info(f"Prognostic variables being used: {args.prognostic_vars}")
# output only:
if "SHFLX" in args.diagnostic_vars:
    variables.append((".cam.h0.SHFLX.203501-206912.nc", "SHFLX", False, "diagnostic")) # surface sensible heat flux
if "LHFLX" in args.diagnostic_vars:
    variables.append((".cam.h0.LHFLX.203501-206912.nc", "LHFLX", False, "diagnostic")) # surfact latent  heat flux
if "PRECT" in args.diagnostic_vars:
    variables.append((".cam.h0.PRECT.203501-206912.nc", "PRECT", False, "diagnostic")) # precipitation?
logger.info(f"Diagnostic variables being used: {args.diagnostic_vars}")
# TODO: radiative flux

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

# initiate model
logger.info(f"SFNO model hyperparams: in_chans={args.in_chans}, out_chans={args.out_chans}, scale_factor={args.scale_factor}, drop_rate={args.drop_rate}")
model = get_ace_sto_sfno(img_shape=(192,288), in_chans=args.in_chans, out_chans=args.out_chans, scale_factor=args.scale_factor,  dropout=args.drop_rate, device=DEVICE)
logger.info(f"Loading in trained model: {args.model_dir}")
checkpoint = torch.load(args.model_dir, map_location=DEVICE)
logger.info("Transferring state dict from saved checkpoint to model")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
logger.info("Model loaded successfully")
loss_fn = AceLoss()
normalizer = ClimateNormalizer()

# load data
SIM_NUMS_TEST = args.test_members
logger.info(f"Loading testing data from ensemble: {SIM_NUMS_TEST}")
real_data = torch.empty(0)
data_to_norm = {}
for sim_num in SIM_NUMS_TEST:
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
        logger.info(f"Done loading in {var_name}")
    normalizer.fit_multiple(data_to_norm)
    logger.info("Done fitting normalizer")
    for var_path, var_name, vert_levels, variable_mode in variables:
        normed_data = normalizer.normalize(data_to_norm[var_name], var_name, 'residual')
        real_data = torch.concat((real_data, normed_data), dim=1)
        data_to_norm[var_name] = None # clear some memory
        logger.info(f"Done normalizing {var_name}")
    data_to_norm = {} # clear some memory
real_data = real_data.to(DEVICE, non_blocking=True)
logger.info("Testing data saved and preprocessed")

# NOTE: this functionality depends on the variable groups being appended to the variables list in this order
num_forcing_chans = 0
num_prognostic_chans = 0
num_diagnostic_chans = 0
for var_path, var_name, vert_levels, variable_mode in variables:
	num_chans = len(vert_level_indices) if vert_levels else 1
	if variable_mode == 'forcing':
		num_forcing_chans += num_chans
	elif variable_mode == 'prognostic':
		num_prognostic_chans += num_chans
	elif variable_mode == 'diagnostic':
		num_diagnostic_chans += num_chans
real_forcings = real_data[:, :num_forcing_chans, :, :]
real_outputs = real_data[:, num_forcing_chans:, :, :]
real_data = torch.empty(0)
logger.info("Finished separating data into variable modes")

# test with autoregressive rollout
logger.info("Beginning rollout")
curr_pred = real_outputs[0].reshape((1, -1, 192, 288))
with LaunchLogger("test", mini_batch_log_freq=1) as launchlog:
    with torch.no_grad():
        for t in range(1, 420):
            curr_pred = model(torch.concat((real_forcings[t-1].reshape((1, -1, 192, 288)), curr_pred[:, :num_prognostic_chans, :, :]), dim=1))
            loss = loss_fn(curr_pred, real_outputs[t].reshape((1, -1, 192, 288)))
            loss += loss.item()
            launchlog.log_minibatch({"Test Loss": loss.detach().cpu().numpy()})

logger.info("Finished Testing!")
