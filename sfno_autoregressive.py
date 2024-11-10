import torch
import modulus
from modulus.launch.logging import LaunchLogger, PythonLogger, initialize_wandb
from ace_helpers import *
from cc2_no_log import *

DEVICE="cuda"

# img_shape=(192,288) [lat x long], time steps (the number of samples) = 35*365=12775
# TODO: should we not model SO2/SO4 vars in output?
model = get_ace_sfno(img_shape=(192,288), in_chans=42, out_chans=42, device=DEVICE)
optimizer = get_ace_optimizer(model)
scheduler = get_ace_lr_scheduler(optimizer)
loss_fn = AceLoss()

logger = PythonLogger("main")
initialize_wandb(
    project="SFNO_test",
    entity="nickmasi",
    name="Train with cc2",
    mode="online",
    resume="never", # use this to separate runs?
)
LaunchLogger.initialize(use_wandb=True)
logger.info("Starting Training!")

# dummy/fake data for 181 days (180 is divisible by 4)
# data = torch.randn((181, 59, 192, 288))
# train = data[:-1, :, :, :]
# test = data[1:, :53, :, :]
# dataset = torch.utils.data.TensorDataset(train,test)

base_dir = "/teamspace/s3_connections/ncar-cesm2-arise-bucket/ARISE-SAI-1.5/b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.001/atm/proc/tseries/day_1"
output_dir = "/teamspace/studios/this_studio/arise_processed"
processor = ARISEProcessor(base_dir, output_dir)
batch_size = ACE_BATCH_SIZE
num_batches = 50 #912
batch_indices = [(i * batch_size, batch_size) for i in range(num_batches)]

for epoch in range(2): #ACE_NUM_EPOCHS):
    with LaunchLogger("train", epoch=epoch, mini_batch_log_freq=1) as launchlog:
        batches = processor.run_batch_processing(batch_indices)
        for batch in batches:
            # batch = tuple(tensor[i*ACE_BATCH_SIZE:(i+1)*ACE_BATCH_SIZE] for tensor in dataset.tensors)
            # pred = model(batch[0])
            # y = batch[1]
            X, y = batch
            pred = model(X.to(DEVICE))
            loss = loss_fn(pred, y.to(DEVICE))
            loss.backward()
            optimizer.step()
            launchlog.log_minibatch({"Loss": loss.detach().cpu().numpy()})
        launchlog.log_epoch({"Learning Rate": optimizer.param_groups[0]["lr"]})
        scheduler.step()

logger.info("Finished Training!")
print("done")
