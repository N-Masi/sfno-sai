import torch
from torch.utils.data import TensorDataset, DataLoader
import xarray as xr
import s3fs
from ace_helpers import *
from climate_normalizier import ClimateNormalizer
from modulus.launch.logging import LaunchLogger, PythonLogger, initialize_wandb
import wandb
import os
import pdb

DEVICE="cuda"

fs = s3fs.S3FileSystem(anon=True)
s3_bucket = "ncar-cesm2-arise"
s3_path_prefix = "ARISE-SAI-1.5/b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.001/atm/proc/tseries/month_1/b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT."

# tuples of (path to monthly data for variable, variable name, whether different vertical levels are needed, forcing-only?)
# input only (forcings):
s3_path_AODVISstdn = (".cam.h0.AODVISstdn.203501-206912.nc", "AODVISstdn", False, True) # this is the signal of the aerosol injections by controller
s3_path_SST = (".cam.h0.SST.203501-206912.nc", "SST", False, True) # sea surface/skin temp (different from TS)
s3_path_SOLIN = (".cam.h0.SOLIN.203501-206912.nc", "SOLIN", False, True) # downward shortwave radiative flux at TOA
# input & output:
s3_path_T = (".cam.h0.T.203501-206912.nc", "T", True, False) # temperature
s3_path_Q = (".cam.h0.Q.203501-206912.nc", "Q", True, False) # specific humidity
s3_path_U = (".cam.h0.U.203501-206912.nc", "U", True, False) # zonal wind
s3_path_V = (".cam.h0.V.203501-206912.nc", "V", True, False) # meridonal wind
s3_path_PS = (".cam.h0.PS.203501-206912.nc", "PS", False, False) # surface pressure
s3_path_TS = (".cam.h0.TS.203501-206912.nc", "TS", False, False) # surface temperature of land or sea-ice (radiative)
# TODO: do we need diagnostic (output-only) variables? Mostly seem to be radiation
variables = [s3_path_AODVISstdn,s3_path_SST, s3_path_SOLIN, s3_path_T, s3_path_Q, s3_path_U, s3_path_V, s3_path_PS, s3_path_TS]
#variables = [s3_path_AODVISstdn, s3_path_T]

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

#model = get_ace_sto_sfno(img_shape=(192,288), in_chans=13, out_chans=12, device=DEVICE)
model = get_ace_sto_sfno(img_shape=(192,288), in_chans=53, out_chans=50, device=DEVICE)
optimizer = get_ace_optimizer(model)
scheduler = get_ace_lr_scheduler(optimizer)
loss_fn = AceLoss()
normalizer = ClimateNormalizer()

# logging for w&b
logger = PythonLogger("main")
initialize_wandb(
    project="SFNO_test",
    entity="nickmasi",
    name="Train with stream_all_vars",
    mode="online",
    resume="never", # use this to separate runs?
)
LaunchLogger.initialize(use_wandb=True)
logger.info("Starting up")

SIM_NUMS_TRAIN = ["001", "002", "003", "004", "005", "006", "007", "008"]
SIM_NUMS_VAL = ["009"]
SIM_NUMS_TEST = ["010"]

# outer nested loops: for each simulation 001-008, for each epoch:
for i, sim_num in enumerate(SIM_NUMS_TRAIN): 

    logger.info(f"Loading simulation {sim_num}")

    # instantiate X & Y as empty tensors
    # Y needs to be shifted one timestep ahead of X
    X = torch.empty(0)
    Y = torch.empty(0)

    # append data (with dim=1) for each variable to X & Y
    for var_path, var_name, vert_levels, forcing_only in variables:
        s3_file_url = f"s3://{s3_bucket}/{s3_path_prefix}{sim_num}{var_path}"
        # open s3 file
        with fs.open(s3_file_url) as varfile:
            with xr.open_dataset(varfile, engine="h5netcdf") as ds: # ignore error on lightning.ai
                # for each vertical level, append with shape (all_time_steps, 1, 192, 288) to X & Y (not to Y if forcing-only) [have to reshape both to switch time & lev]
                if vert_levels:
                    data_xarray = ds.isel(lev=vert_level_indices)[[var_name]]
                else:
                    data_xarray = ds.isel()[[var_name]]
                num_time_steps = len(data_xarray.time)
                data = torch.from_numpy(data_xarray.to_array().values).reshape(num_time_steps, -1, 192, 288)
                normalizer.fit(data, var_name)
                normed_data = normalizer.normalize(data, var_name, 'residual')
                X = torch.concat((X, normed_data[:-1]), dim=1)
                if not forcing_only:
                    Y = torch.concat((Y, normed_data[1:]), dim=1)
        logger.info(f"Done loading {var_name}")

    # make dataloader out of (X, Y) with batch_size ACE_BATCH_SIZE=4
    # rough estimates that X & Y will each be 5.3Gb, should be doable on OSCAR
    tds = TensorDataset(X, Y)
    data_loader = DataLoader(tds, batch_size=ACE_BATCH_SIZE, shuffle=True) # TODO: shuffle?

    # train for this data multiple times (epochs), logging to w&b
    for single_sim_epoch in range(SINGLE_SIM_EPOCHS): #
        epoch = single_sim_epoch*(i+1)
        with LaunchLogger("train", epoch=epoch, mini_batch_log_freq=1) as launchlog:
            for batch in data_loader:
                torch.cuda.empty_cache()
                optimizer.zero_grad()
                pred = model(batch[0].to(DEVICE))
                loss = loss_fn(pred, batch[1].to(DEVICE))
                loss.backward()
                optimizer.step()
                launchlog.log_minibatch({"Loss": loss.detach().cpu().numpy()})
            launchlog.log_epoch({"Learning Rate": optimizer.param_groups[0]["lr"]})
            scheduler.step()
        logger.info(f"Epoch {epoch} done")

    # save checkpoint of model to w&b
    checkpoint = {
        'epoch': (i+1)*SINGLE_SIM_EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }
    tmp_path = f"temp_checkpoint_sim_{sim_num}.pt"
    torch.save(checkpoint, tmp_path)
    artifact = wandb.Artifact(
        name=f"model-checkpoint-sim-{sim_num}",
        type="model",
        description=f"Model checkpoint after training on sim {sim_num}"
    )
    artifact.add_file(tmp_path)
    wandb.log_artifact(artifact)
    os.remove(tmp_path) # Clean up temporary file
    logger.info(f"Model checkpoint after training on sim {sim_num} saved to w&b")

logger.info("Finished Training!")
