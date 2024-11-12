# example of how to use modulus for an SFNO on Oscar
# for setup, see: https://docs.google.com/document/d/1EX7XAli5ZD4qx42Z0CDNn4iU1U8jv2HC6fvU87xYbTU/

import torch
import modulus
from modulus.datapipes.benchmarks.darcy import Darcy2D
from modulus.launch.logging import LaunchLogger, PythonLogger, initialize_wandb
from ace_helpers import *

normaliser = {
    "permeability": (1.25, 0.75),
    "darcy": (4.52e-2, 2.79e-2),
}
dataloader = Darcy2D(
    resolution=16, batch_size=ACE_BATCH_SIZE, nr_permeability_freq=5, normaliser=normaliser
)

model = get_ace_sfno(img_shape=(16,16), in_chans=1, out_chans=1)
optimizer = get_ace_optimizer(model)
scheduler = get_ace_lr_scheduler(optimizer)
loss_fn = AceLoss()

logger = PythonLogger("main")
initialize_wandb(
    project="SFNO_test",
    entity="nickmasi",
    name="Train on Darcy2D",
    mode="online",
    resume="never", # use this to separate runs?
)
LaunchLogger.initialize(use_wandb=True)
logger.info("Starting Training!")

for epoch in range(ACE_NUM_EPOCHS):
    with LaunchLogger("train", epoch=epoch, mini_batch_log_freq=1) as launchlog:
        for i in range(5):
            batch = next(iter(dataloader)) # ignore error
            true = batch["darcy"]
            pred = model(batch["permeability"])
            loss = loss_fn(pred, true)
            loss.backward()
            optimizer.step()
            launchlog.log_minibatch({"Loss": loss.detach().cpu().numpy()})
        launchlog.log_epoch({"Learning Rate": optimizer.param_groups[0]["lr"]})
        scheduler.step()

logger.info("Finished Training!")
print("done")
