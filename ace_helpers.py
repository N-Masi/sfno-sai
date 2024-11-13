# paper: https://arxiv.org/pdf/2310.02074
# recreating the ACE model, which is SFNO with specific loss, preprocessing, hyperparams, etc.
# for hyperparams, see Table 4 of the linked paper (ACE, Watt-Meyer et al. 2023)

from typing import Union
import torch
import modulus
from ai2modulus.models.sfno.sfnonet import SphericalFourierNeuralOperatorNet

ACE_NUM_EPOCHS = 30
SINGLE_SIM_EPOCHS = 3
ACE_BATCH_SIZE = 4

def get_ace_sto_sfno(img_shape=(721, 1440), in_chans=2, out_chans=2, scale_factor=1, dropout=0.1, device="cuda") -> torch.nn.Module:
    '''
    Returns the SFNO as used in the linked paper (ACE, Watt-Meyer et al. 2023)
    Non-passable hyperparams are those set in Table 3 of (ACE, Watt-Meyer et al. 2023)
    With dropout=0.1, ACE-STO baseline is implemented
    Defaults refer to the ai2modulus SFNO defaults at:
        https://github.com/ai2cm/modulus/blob/22df4a9427f5f12ff6ac891083220e7f2f54d229/ai2modulus/models/sfno/sfnonet.py
    '''
    model = SphericalFourierNeuralOperatorNet(
        params={},
        filter_type="linear",
        operator_type="dhconv",
        img_shape=img_shape, # (721, 1440) is ai2modulus default
        in_chans=in_chans, # 2 is ai2modulus defualt
        out_chans=out_chans, # 2 is ai2modulus default
        scale_factor=scale_factor, # ACE uses 1, ai2modulus default is 16; lower s_f means higher frequency threshold, so the model can attend to more (& higher) frequencies
        embed_dim=256,
        num_layers=8,
        drop_rate=dropout, # Spherical DYffusion paper uses this for ACE-STO (one SFNO w/ dropout) as benchmark, could be set lower/zero
        spectral_layers=3
    ).to(device)
    averaged_model = torch.optim.swa_utils.AveragedModel(
        model, 
        multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(decay=0.9999)
    )
    return averaged_model

def get_ace_optimizer(model: torch.nn.Module) -> torch.optim.Optimizer:
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
    return optimizer

def get_ace_lr_scheduler(optimizer: torch.optim.Optimizer, num_epochs: int=ACE_NUM_EPOCHS) -> torch.optim.lr_scheduler.LRScheduler:
    '''
    Only step this lr_scheduler once per epoch so that you have the single cycle annealing from the paper,
    this makes it so the lr diminishes towards 0 at the end of training (i.e., by the last epoch)
    '''
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    return lr_scheduler

def ace_data_normalizer():
    pass

class AceLoss(torch.nn.Module):
    def __init__(self):
        super(AceLoss, self).__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculates batch-averaged loss
        See §2.2 of the linked paper (ACE, Watt-Meyer et al. 2023)

        Parameters
        ----------
        pred : Tensor
            Input prediction tensor of shape (batch_size, sample_size)
        target : Tensor
            Target/label tensor of shape (batch_size, sample_size)
        """
        numerator = torch.sqrt(torch.sum((pred-target)**2, dim=1))
        denominator = torch.sqrt(torch.sum((target)**2, dim=1))
        return torch.mean(numerator/denominator)
