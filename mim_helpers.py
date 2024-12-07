import torch
import numpy as np
import random
from torch.nn import L1Loss
import pdb

class MIMLoss(torch.nn.Module):
    def __init__(self):
        super(MIMLoss, self).__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Calculates averaged L1 loss of just the masked regions

        Parameters
        ----------
        pred : Tensor
            Input prediction tensor of shape (batch_size, sample_size)
        target : Tensor
            Target/label tensor of shape (batch_size, sample_size)
        """
        if mask.sum() == 0:
            return torch.Tensor([0])
        l1 = L1Loss()
        loss = l1(pred[mask], target[mask], reduction='mean')
        return loss

def get_mim_mask(batch_size: int, chans: int, same_mask_across_chans: bool, masking_ratio: float, patch_size: int, seed: int, lat: int = 192, lon: int = 288) -> torch.Tensor:
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    assert lat%patch_size==0
    assert lon%patch_size==0

    mask = torch.zeros((batch_size, chans, lat, lon))
    for i in range(batch_size):
        if same_mask_across_chans:
            mask[i, :] = get_mim_mask_one_sample(masking_ratio, patch_size, lat, lon)
        else:
            for j in range(chans):
                mask[i, j] = get_mim_mask_one_sample(masking_ratio, patch_size, lat, lon)
    return mask

def get_mim_mask_one_sample(masking_ratio: float, patch_size: int, lat: int, lon: int) -> torch.Tensor:
    num_lat_patches = lat//patch_size
    num_patches = num_lat_patches * (lon//patch_size)
    num_masked_patches = int(np.ceil(num_patches * masking_ratio))
    mask_idx = np.random.permutation(num_patches)[:num_masked_patches]

    mask = torch.zeros((lat, lon))
    for idx in mask_idx:
        x = (idx%num_lat_patches)*patch_size
        y = (idx//num_lat_patches)*patch_size
        mask[x:(x+patch_size), y:(y+patch_size)] = 1
    return mask
