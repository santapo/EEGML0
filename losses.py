import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-8

def entropy_error_function(input, target):
    return -torch.sum(target*torch.log(input/(target + EPS)) \
                      + (1-target)*torch.log((1-input)/(1-target + EPS)))


class EEGMLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        return F.binary_cross_entropy(input, target, reduction="mean")


class EEGMLossWithL0Regularization(nn.Module):
    def __init__(self, model, weight_decay, sigma, weight_decay_function):
        super().__init__()
        self.model = model
        self.sigma = sigma
        self.weight_decay = weight_decay
        self.weight_decay_function = weight_decay_function

    def forward(self, input, target):
        eegm_loss = F.binary_cross_entropy(input, target, reduction="sum")
        l0 = sum([self.weight_decay_function(p, self.sigma).sum() for p in self.model.parameters()])
        eegm_loss_with_l0 = eegm_loss + self.weight_decay*l0
        return eegm_loss_with_l0


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        return torch.sum(torch.sqrt(F.mse_loss(input, target, reduction="none")))