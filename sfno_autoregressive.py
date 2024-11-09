import torch
import modulus
from modulus.launch.logging import LaunchLogger, PythonLogger, initialize_wandb
from ace_helpers import *

# img_shape=(192,288) [lat x long], time steps (the number of samples) = 35*365=12775
model = get_ace_sfno(img_shape=(192,288), in_chans=59, out_chans=53, device="cpu")
optimizer = get_ace_optimizer(model)
scheduler = get_ace_lr_scheduler(optimizer)
loss_fn = AceLoss()

logger = PythonLogger("main")
initialize_wandb(
    project="SFNO_test",
    entity="nickmasi",
    name="Train on Dummy",
    mode="online",
    resume="never", # use this to separate runs?
)
LaunchLogger.initialize(use_wandb=True)
logger.info("Starting Training!")

# dummy/fake data for 181 days (180 is divisible by 4)
data = torch.randn((181, 59, 192, 288))
train = data[:-1, :, :, :]
test = data[1:, :53, :, :]
dataset = torch.utils.data.TensorDataset(train,test)

for epoch in range(1): #ACE_NUM_EPOCHS):
    with LaunchLogger("train", epoch=epoch, mini_batch_log_freq=1) as launchlog:
        for i in range(45):
            batch = tuple(tensor[i*ACE_BATCH_SIZE:(i+1)*ACE_BATCH_SIZE] for tensor in dataset.tensors)
            pred = model(batch[0])
            true = batch[1]
            loss = loss_fn(pred, true)
            loss.backward()
            optimizer.step()
            launchlog.log_minibatch({"Loss": loss.detach().cpu().numpy()})
        launchlog.log_epoch({"Learning Rate": optimizer.param_groups[0]["lr"]})
        scheduler.step()

logger.info("Finished Training!")
print("done")
