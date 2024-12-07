import torch
from torch.utils.data import TensorDataset, DataLoader
import xarray as xr
import s3fs
from ace_helpers import *
from mim_helpers import *
from climate_normalizer import ClimateNormalizer
from modulus.launch.logging import LaunchLogger, PythonLogger, initialize_wandb
import wandb
import os
import pdb
import random
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("run_name", type=str, help="name of the run/script as it will appear in w&b")
parser.add_argument("-s", "--seed", type=int, default=2952, help="randomizing seed")
parser.add_argument("-d", "--device", default="cuda", choices=["cuda", "cpu"], help="device to run on")
parser.add_argument("-p", "--prognostic_vars", type=str, nargs="*", default=["T", "Q", "U", "V", "PS", "TS"], help="prognostic (input & output) variables to use")
parser.add_argument("-c", "--channels", type=int, default=50, help="# of in channels in the 'image'; 1 channel per variable per vertical level")
parser.add_argument("-m", "--scale_factor", type=int, default=1, help="scale_factor in SFNO model, higher scale_factor multiplicatively decreases the threshold of frequency modes kept after spherical harmonic transform")
parser.add_argument("-r", "--drop_rate", type=float, default=0, help="dropout rate applied to SFNO encoder and the SFNO's Fourier layer MLPs")
parser.add_argument("-t", "--train_members", type=str, nargs="+", default=["001", "006", "002", "007", "005", "010"], help="ensemble members to use for training on")
parser.add_argument("-T", "--test_members", type=str, nargs="*", default=["004"], help="ensemble members to use for testing")
parser.add_argument("-v", "--val_members", type=str, nargs="+", default=["003"], help="ensemble members to use for validation")
parser.add_argument("-e", "--member_epochs", type=int, default=3, help="number of times each ensemble member is trained on")
parser.add_argument("-q", "--checkpoint_freq", type=int, default=1, help="after how many ensembles should a model checkpoint be saved on w&b")
parser.add_argument("-a", "--same_mask_across_chans", type=bool, default=True, help="whether to mask the same patches of gridpoints on all channels (default True)")
parser.add_argument("-M", "--masking_ratio", type=float, default=0.5, help="percentage of gridpoints to mask")
parser.add_argument("-P", "--patch_size", type=int, default=32, help="masked patches will have size PxP")
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
    name=args.run_name,
    mode="online",
    resume="never", # use this to separate runs?
)
LaunchLogger.initialize(use_wandb=True)
logger.info("Starting up")
logger.info("Task: masked image modeling")
logger.info(f"RUNNING: {args.run_name}")

# connection to s3 for streaming data
fs = s3fs.S3FileSystem(anon=True)
s3_bucket = "ncar-cesm2-arise"
s3_path_chunk_1 = "ARISE-SAI-1.5/b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT."
s3_path_chunk_2 = "/atm/proc/tseries/month_1/b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT."
logger.info("Using ARISE-SAI-1.5 dataset")

variables = []
# tuples of (path to monthly data for variable, variable name, whether different vertical levels are needed, variable mode)
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
logger.info(f"SFNO model hyperparameters: in_chans={args.channels}, out_chans={args.channels}, scale_factor={args.scale_factor}, drop_rate={args.drop_rate}")
logger.info(f"Masking hyperparameters: masking_ratio={args.masking_ratio}, patch_size={args.patch_size}x{args.patch_size}, using the same mask on all channels of a var: {args.same_mask_across_chans}")
model = get_ace_sto_sfno(img_shape=(192,288), in_chans=args.channels, out_chans=args.channels, scale_factor=args.scale_factor, dropout=args.drop_rate, device=DEVICE)
optimizer = get_ace_optimizer(model)
scheduler = get_ace_lr_scheduler(optimizer)
loss_fn = MIMLoss()
normalizer = ClimateNormalizer()

SIM_NUMS_TRAIN = args.train_members
SIM_NUMS_VAL = args.val_members
SIM_NUMS_TEST = args.test_members
logger.info(f"Training on simulations: {SIM_NUMS_TRAIN}")
logger.info(f"Validating on simulations: {SIM_NUMS_VAL}")
logger.info(f"Number of times each ensemble member is trained on: {args.member_epochs}. Total number of epochs equals {len(SIM_NUMS_TRAIN)}*{args.member_epochs}={len(SIM_NUMS_TRAIN)*args.member_epochs}.")

# create validation data
logger.info(f"Loading validation data")
X_val = torch.empty(0)
data_to_norm = {}
for sim_num in SIM_NUMS_VAL:
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
        if variable_mode != "diagnostic":
            X_val = torch.concat((X_val, normed_data), dim=1)
        data_to_norm[var_name] = None # clear some memory
        logger.info(f"Done normalizing {var_name}")
    data_to_norm = {} # clear some memory
val_tds = TensorDataset(X_val)
val_loader = DataLoader(val_tds, batch_size=ACE_BATCH_SIZE)
X_val = torch.empty(0)
logger.info("Validation data saved and preprocessed")

# outer nested loops: for each simulation 001-008, for each epoch:
for i, sim_num in enumerate(SIM_NUMS_TRAIN): 

    logger.info(f"Loading simulation {sim_num}")

    # append data (with dim=1) for each variable to X & Y
    X = torch.empty(0)
    data_to_norm = {}
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
        if variable_mode != "diagnostic":
            X = torch.concat((X, normed_data), dim=1)
        data_to_norm[var_name] = None # clear some memory
        logger.info(f"Done normalizing {var_name}")
    data_to_norm = {} # clear some memory
    logger.info("Preprocessing complete")

    tds = TensorDataset(X)
    data_loader = DataLoader(tds, batch_size=ACE_BATCH_SIZE, shuffle=True)
    X = torch.empty(0)

    # train for this data multiple times (epochs), logging to w&b
    for single_sim_epoch in range(args.member_epochs):
        epoch = single_sim_epoch+(i*args.member_epochs)
        with LaunchLogger("train", epoch=epoch, mini_batch_log_freq=1) as launchlog:
            model.train()
            for batch in data_loader:
                torch.cuda.empty_cache()
                optimizer.zero_grad()
                mask = get_mim_mask(len(batch[0]), args.channels, args.same_mask_across_chans, args.masking_ratio, args.patch_size, args.seed).to(DEVICE, non_blocking=True, dtype=torch.int)
                x = batch[0].to(DEVICE, non_blocking=True)
                pred = model(x * (1 - mask))
                loss = loss_fn(pred, x, mask)
                loss.backward()
                optimizer.step()
                launchlog.log_minibatch({"Loss": loss.detach().cpu().numpy()})
            launchlog.log_epoch({"Learning Rate": optimizer.param_groups[0]["lr"]})
            scheduler.step()
            logger.info(f"Epoch {epoch} training done")

            # validation for this epoch
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    mask = get_mim_mask(len(batch[0]), args.channels, args.same_mask_across_chans, args.masking_ratio, args.patch_size, args.seed).to(DEVICE, non_blocking=True, dtype=torch.int)
                    x = batch[0].to(DEVICE, non_blocking=True)
                    preds_val = model(x * (1 - mask))
                    loss_val = loss_fn(preds_val, x, mask)
                    val_loss += loss_val.item()
            avg_val_loss = val_loss / len(val_loader)
            logger.info(f"Validation loss: {avg_val_loss}")
            launchlog.log_epoch({"Validation Loss": avg_val_loss})

    # save checkpoint of model to w&b after all #(single_sim_epoch) epochs on one simulation
    if (i+1)%args.checkpoint_freq == 0:
        checkpoint = {
            'epoch': (i+1)*args.member_epochs-1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }
        tmp_path = f"temp_checkpoint_sim_{sim_num}.pt"
        torch.save(checkpoint, tmp_path)
        artifact = wandb.Artifact(
            name=f"model-checkpoint-sim-{sim_num}",
            type="model",
            description=f"Model checkpoint after training on sim {sim_num} from run {args.run_name}"
        )
        artifact.add_file(tmp_path)
        wandb.log_artifact(artifact)
        os.remove(tmp_path) # Clean up temporary file
        logger.info(f"Model checkpoint after training on sim {sim_num} (after epoch {(i+1)*args.member_epochs-1}) saved to w&b")

logger.info("Finished Training!")
