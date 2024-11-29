import torch

# TODO: have it take in the "image" and the mask mode
def mask():
    pass

class MIMLoss(torch.nn.Module):
    def __init__(self):
        super(MIMLoss, self).__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor, patched_locations) -> torch.Tensor:
        """Calculates batch-averaged loss
        See §2.2 of the linked paper (ACE, Watt-Meyer et al. 2023)

        Parameters
        ----------
        pred : Tensor
            Input prediction tensor of shape (batch_size, sample_size)
        target : Tensor
            Target/label tensor of shape (batch_size, sample_size)
        """
        #TODO: l1 loss on just the patched_locations
        return ...