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

parser = argparse.ArgumentParser()
parser.add_argument("run_name", type=str, help="name of the run/script as it will appear in w&b")
parser.add_argument("mim_model_path", type=str, help="filepath to pretrained (on MIM) model whose encoder to use")
parser.add_argument("-s", "--seed", type=int, default=2952, help="randomizing seed")
parser.add_argument("-c", "--device", default="cuda", choices=["cuda", "cpu"], help="device to run on")
parser.add_argument("-m", "--scale_factor", type=int, default=1, help="scale_factor in SFNO model, higher scale_factor multiplicatively decreases the threshold of frequency modes kept after spherical harmonic transform")
parser.add_argument("-r", "--drop_rate", type=float, default=0, help="dropout rate applied to SFNO encoder and the SFNO sFourier layer MLPs")
parser.add_argument("-t", "--train_members", type=str, nargs="+", default=["001", "006", "002", "007", "005", "010"], help="ensemble members to use for training on")
parser.add_argument("-T", "--test_members", type=str, nargs="*", default=["004"], help="ensemble members to use for testing")
parser.add_argument("-v", "--val_members", type=str, nargs="+", default=["003"], help="ensemble members to use for validation")
parser.add_argument("-e", "--member_epochs", type=int, default=3, help="number of times each ensemble member is trained on")
parser.add_argument("-q", "--checkpoint_freq", type=int, default=1, help="after how many ensembles should a model checkpoint be saved on w&b")
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
logger.info("Task: next-timestep climate prediction")
logger.info(f"RUNNING: {args.run_name}")

# data stuff from oscar /data dir
logger.info("Using ARISE-SAI-1.5 dataset")
logger.info('Forcings: ["AODVISstdn", "SOLIN"] (2 channels)')
logger.info('Prognostics: ["T", "Q", "U", "V", "PS", "TS"] (50 channels)')
logger.info('Diagnostics: ["SHFLX", "LHFLX", "PRECT"] (3 channels)')
DATA_FILEPATH_STEM = "../sfno-sai/data/normed_data_55_chans_"

# initiate model
logger.info(f"SFNO model hyperparams: in_chans=52, out_chans=53, scale_factor={args.scale_factor}, drop_rate={args.drop_rate}")
model = get_ace_sto_sfno(img_shape=(192,288), in_chans=52, out_chans=53, scale_factor=args.scale_factor,  dropout=args.drop_rate, device=DEVICE)
optimizer = get_ace_optimizer(model)
scheduler = get_ace_lr_scheduler(optimizer)
loss_fn = AceLoss()
normalizer = ClimateNormalizer()

# get pretrained encoder
# e.g., args.mim_model_path = "artifacts/model-mim-0.3-16-same/temp_checkpoint_sim_010.pt"
logger.info(f"Loading in pretrained model: {args.mim_model_path}")
checkpoint = torch.load(args.mim_model_path, weights_only=True)
logger.info("Pretrained model loaded!")
enc_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if 'encoder' in k}
new_dict = {k: v for k, v in model.state_dict().items() if not ('encoder' in k)}
new_dict.update(enc_dict)
model.load_state_dict(new_dict)
logger.info("Pretrained encoder brought over to new model!")
checkpoint = torch.empty(0)
new_dict = torch.empty(0)
enc_dict_keys = enc_dict.keys()
enc_dict = torch.empty(0)
logger.info("Memory cleared by emptying out pretrained model")
def freeze_encoder(model):
    for k in enc_dict_keys:
        model.state_dict()[k].requires_grad = False
    logger.info("Pretrained encoder frozen")
# freeze_encoder(model)

SIM_NUMS_TRAIN = ["001", "006", "002", "007", "005", "010"]
SIM_NUMS_VAL = ["003"]
logger.info(f"Training on simulations: {SIM_NUMS_TRAIN}")
logger.info(f"Validating on simulations: {SIM_NUMS_VAL}")
logger.info(f"Number of times each ensemble member is trained on: {args.member_epochs}. Total number of epochs equals {len(SIM_NUMS_TRAIN)}*{args.member_epochs}={len(SIM_NUMS_TRAIN)*args.member_epochs}.")

# outer nested loops: for each simulation 001-008, for each epoch:
for i, sim_num in enumerate(SIM_NUMS_TRAIN): 

    logger.info(f"Loading simulation {sim_num}")
    X = torch.load(DATA_FILEPATH_STEM+sim_num+".pt")
    logger.info("Data loaded!")
    Y = X[1:, 2:] # prognostic + diagnostic
    X = X[:-1, :52] # forcing + prognostic
    tds = TensorDataset(X, Y)
    data_loader = DataLoader(tds, batch_size=ACE_BATCH_SIZE, shuffle=True)
    X = torch.ones(0)
    Y = torch.ones(0)
    logger.info("Data moved into dataloader!")

    # train for this data multiple times (epochs), logging to w&b
    for single_sim_epoch in range(args.member_epochs):
        epoch = single_sim_epoch+(i*args.member_epochs)
        with LaunchLogger("train", epoch=epoch, mini_batch_log_freq=1) as launchlog:
            model.train()
            # freeze_encoder(model)
            for batch in data_loader:
                torch.cuda.empty_cache()
                optimizer.zero_grad()
                pred = model(batch[0].to(DEVICE, non_blocking=True))
                loss = loss_fn(pred, batch[1].to(DEVICE, non_blocking=True))
                loss.backward()
                optimizer.step()
                launchlog.log_minibatch({"Loss": loss.detach().cpu().numpy()})
            launchlog.log_epoch({"Learning Rate": optimizer.param_groups[0]["lr"]})
            scheduler.step()
            logger.info(f"Epoch {epoch} training done")

            # validation for this epoch
            model.eval()
            val_sim = SIM_NUMS_VAL[0]
            logger.info(f"Loading validation data from {val_sim}")
            X = torch.load(DATA_FILEPATH_STEM+val_sim+".pt")
            logger.info("Data loaded!")
            Y = X[1:, 2:] # prognostic + diagnostic
            X = X[:-1, :52] # forcing + prognostic
            val_tds = TensorDataset(X, Y)
            val_loader = DataLoader(val_tds, batch_size=ACE_BATCH_SIZE, shuffle=True)
            X = torch.ones(0)
            Y = torch.ones(0)
            logger.info("Data moved into dataloader!")
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    preds_val = model(batch[0].to(DEVICE, non_blocking=True))
                    loss_val = loss_fn(preds_val, batch[1].to(DEVICE, non_blocking=True))
                    val_loss += loss_val.item()
            avg_val_loss = val_loss / len(val_loader)
            logger.info(f"Validation loss: {avg_val_loss}")
            launchlog.log_epoch({"Validation Loss": avg_val_loss})
            val_tds = torch.ones(0)
            val_dataloader = torch.ones(0)

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
