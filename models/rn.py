import torch
import torch.nn as nn
import torch.nn.functional as F

class RN_binarylabel(nn.Module):
    def __init__(self, feature_channels):
        super(RN_binarylabel, self).__init__()
        self.bn_norm = nn.InstanceNorm2d(feature_channels, affine=False, track_running_stats=False)

    def forward(self, x, label):

        label = label.detach()
        rn_foreground_region = self.rn(x * label, label)

        rn_background_region = self.rn(x * (1 - label), 1 - label)

        return rn_foreground_region + rn_background_region

    def rn(self, region, mask):

        shape = region.size()

        sum = torch.sum(region, dim=[0,2,3])  
        Sr = torch.sum(mask, dim=[0,2,3])    

        Sr[Sr==0] = 1

        mu = (sum / Sr)     

        return self.bn_norm(region + (1 - mask) * mu[None,:,None,None]) * \
        (torch.sqrt(Sr / (shape[0] * shape[2] * shape[3])))[None,:,None,None]

class RN_B(nn.Module):
    def __init__(self, feature_channels):
        super(RN_B, self).__init__()
        
        self.rn = RN_binarylabel(feature_channels)    

        self.foreground_gamma = nn.Parameter(torch.zeros(feature_channels), requires_grad=True)
        self.foreground_beta = nn.Parameter(torch.zeros(feature_channels), requires_grad=True)
        self.background_gamma = nn.Parameter(torch.zeros(feature_channels), requires_grad=True)
        self.background_beta = nn.Parameter(torch.zeros(feature_channels), requires_grad=True)

    def forward(self, x, mask):
        mask = F.interpolate(mask, size=x.size()[2:], mode='nearest')   

        rn_x = self.rn(x, mask)

        rn_x_foreground = (rn_x * mask) * (1 + self.foreground_gamma[None,:,None,None]) + self.foreground_beta[None,:,None,None]
        rn_x_background = (rn_x * (1 - mask)) * (1 + self.background_gamma[None,:,None,None]) + self.background_beta[None,:,None,None]

        return rn_x_foreground + rn_x_background

