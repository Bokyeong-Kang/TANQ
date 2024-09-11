import torch
import torch.nn as nn
import torch.nn.functional as F


class RAIN(nn.Module):
    def __init__(self, dims_in, eps=1e-5):
        '''Compute the instance normalization within only the background region, in which
            the mean and standard variance are measured from the features in background region.
        '''
        super(RAIN, self).__init__()
        self.foreground_gamma = nn.Parameter(torch.zeros(dims_in), requires_grad=True)
        self.foreground_beta = nn.Parameter(torch.zeros(dims_in), requires_grad=True)
        self.cochlea_gamma = nn.Parameter(torch.zeros(dims_in), requires_grad=True)
        self.cochlea_beta = nn.Parameter(torch.zeros(dims_in), requires_grad=True)
        self.background_gamma = nn.Parameter(torch.zeros(dims_in), requires_grad=True)
        self.background_beta = nn.Parameter(torch.zeros(dims_in), requires_grad=True)
        self.eps = eps

    def forward(self, x, mask):
        mask = F.interpolate(mask.detach(), size=x.size()[2:], mode='nearest')

        xor_temp = torch.ones_like(mask)
        back_mask = torch.logical_xor(mask, xor_temp).long()

        mask_vs = torch.where(mask > 1.0, torch.tensor(0, dtype=mask.dtype, device=mask.device), mask)
        mask_cochlea = torch.where(mask < 2.0, torch.tensor(0, dtype=mask.dtype, device=mask.device), mask)

        # background
        mean_back, std_back = self.get_foreground_mean_std(x * back_mask, back_mask)  
        normalized = (x - mean_back) / std_back
        normalized_background = (normalized * (1 + self.background_gamma[None, :, None, None]) + self.background_beta[None, :, None, None]) * back_mask

        # cochlea
        mean_cochlea, std_cochlea = self.get_foreground_mean_std(x * mask_cochlea, mask_cochlea)  
        normalized_cochlea = (x - mean_cochlea) / std_cochlea
        normalized_cochlea = (normalized_cochlea * (1 + self.cochlea_gamma[None, :, None, None]) + self.cochlea_beta[None, :, None, None]) * mask_cochlea

        mean_fore, _ = self.get_foreground_mean_std(x * mask_vs, mask_vs)  
        normalized = (x - mean_fore) + mean_back
        normalized_foreground = (normalized * (1 + self.foreground_gamma[None, :, None, None]) + self.foreground_beta[None, :, None, None]) * mask_vs

        return normalized_foreground + normalized_background + normalized_cochlea

    def get_foreground_mean_std(self, region, mask):
        sum = torch.sum(region, dim=[2, 3])    
        num = torch.sum(mask, dim=[2, 3])       
        mu = sum / (num + self.eps)
        mean = mu[:, :, None, None]
        var = torch.sum((region + (1 - mask)*mean - mean) ** 2, dim=[2, 3]) / (num + self.eps)
        var = var[:, :, None, None]
        return mean, torch.sqrt(var+self.eps)
