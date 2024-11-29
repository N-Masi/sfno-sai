import torch
from typing import Dict

class ClimateNormalizer:
    """Climate data normalizer implementing both full-field and residual normalization strategies.
    Implements Appendix H of the ACE paper (Watt-Meyer et al. 2023).

    This normalizer implements the two-stage normalization process:
    1. Full-field normalization: (x - mean) / std
    2. Residual normalization: Rescales full-field normalized data using temporal differences

    Expected data format:
    - Input tensor should be in shape (time, levels, lat, lon) where levels represent vertical levels
    - All statistics are computed across time, lat, and lon dimensions independently for each vertical level
    
    Example shapes:
    - 2D variable (no levels): (420, 192, 288)
    - 3D variable (with levels): (420, 12, 192, 288)
    """

    def __init__(self):
        """Initialize normalizer with empty statistics."""
        # Store normalization parameters for each variable
        self.stats = {}
        
    def fit(self, x: torch.Tensor, variable_name: str):
        """Compute normalization statistics for a variable.
        
        Args:
            x (torch.Tensor): Input tensor of shape (time, levels, lat, lon)
            variable_name (str): Name of variable for storing statistics
            
        The function computes:
        1. Full-field statistics (mean, std) across (time, lat, lon) for each level
        2. Residual scaling based on temporal differences
        """
        # Validate input
        assert len(x.shape) == 4, "Input must have 4 dimensions (time, levels, lat, lon)"

        if len(x.shape) == 4:  # 3D variable with levels
            mu = torch.mean(x, dim=(0, 2, 3), keepdim=True)  # mean across time, lat, lon for each level
            sigma = torch.std(x, dim=(0, 2, 3), keepdim=True)  # std across time, lat, lon for each level
            
            # Apply full-field normalization per level
            x_ff = (x - mu) / sigma

            # Compute forward differences (a(t+1) - a(t)) for each level
            x_ff_diff = x_ff[1:] - x_ff[:-1]

            # Compute standard deviation of differences per level
            sigma_ff_diff = torch.std(x_ff_diff, dim=(0, 2, 3), keepdim=True)
        
        # Store statistics
        self.stats[variable_name] = {
            'mu': mu,
            'sigma': sigma,
            'sigma_ff_diff': sigma_ff_diff
        }
    
    def fit_multiple(self, variables_dict: Dict[str, torch.Tensor]):
        """Fit normalizer on multiple variables and compute reference sigma.
        
        Args:
            variables_dict: Dictionary mapping variable names to their tensors
                          Each tensor should be shape (time, levels, lat, lon) or (time, lat, lon)
        """
        # Fit each variable individually
        for var_name, tensor in variables_dict.items():
            self.fit(tensor, var_name)
        
        # Compute geometric mean of sigma_ff_diff across all variables and levels
        sigmas = torch.cat([
            stats['sigma_ff_diff'].flatten()
            for stats in self.stats.values()
        ])
        sigma_ref = torch.exp(torch.mean(torch.log(sigmas)))
        
        # Compute residual scaling factors
        for var_name in self.stats:
            self.stats[var_name]['sigma_res'] = (
                self.stats[var_name]['sigma_ff_diff'] / sigma_ref
            )
    
    def normalize(self, x: torch.Tensor, variable_name: str, 
                 mode: str = 'residual') -> torch.Tensor:
        """Normalize input data using computed statistics.
        
        Args:
            x (torch.Tensor): Input tensor of shape (time, levels, lat, lon) or (time, lat, lon)
            variable_name (str): Name of variable for loading statistics
            mode (str): Either 'full-field' or 'residual'
            
        Returns:
            torch.Tensor: Normalized tensor of same shape as input
        """
        stats = self.stats[variable_name]
        
        # Full-field normalization
        x_ff = (x - stats['mu']) / stats['sigma']
        
        if mode == 'full-field':
            return x_ff
        elif mode == 'residual':
            # Apply residual scaling
            return x_ff / stats['sigma_res']
        else:
            raise ValueError(f"Unknown normalization mode: {mode}")
    
    def denormalize(self, x: torch.Tensor, variable_name: str, 
                   mode: str = 'residual') -> torch.Tensor:
        """Denormalize data back to original scale.
        
        Args:
            x (torch.Tensor): Normalized tensor of shape (time, levels, lat, lon) or (time, lat, lon)
            variable_name (str): Name of variable for loading statistics
            mode (str): Either 'full-field' or 'residual'
            
        Returns:
            torch.Tensor: Denormalized tensor of same shape as input
        """
        stats = self.stats[variable_name]
        
        if mode == 'residual':
            # First undo residual scaling
            x = x * stats['sigma_res']
        
        # Undo full-field normalization
        return x * stats['sigma'] + stats['mu']
    

if __name__ == "__main__":
    # Example for single variable
    normalizer = ClimateNormalizer()

    # For 2D variable (e.g., surface pressure)
    ps = torch.randn(420, 1, 192, 288)  # (time, levels, lat, lon)
    normalizer.fit(ps, 'PS')

    # For 3D variable (e.g., temperature with levels)
    temp = torch.randn(420, 12, 192, 288)  # (time, levels, lat, lon)
    normalizer.fit(temp, 'T')

    # Fit multiple variables at once to compute reference sigma
    variables = {
        'PS': ps,
        'T': temp,
        # ... other variables
    }
    normalizer.fit_multiple(variables)

    # Apply normalization
    ps_norm = normalizer.normalize(ps, 'PS')
    temp_norm = normalizer.normalize(temp, 'T')

    # Denormalize
    ps_denorm = normalizer.denormalize(ps_norm, 'PS')
    temp_denorm = normalizer.denormalize(temp_norm, 'T')
