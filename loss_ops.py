import math
import torch.nn as nn
import torch
from pytorch_msssim import MS_SSIM

class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2, target='mse'):
        super().__init__()
        if target == 'mse':
            self.target = nn.MSELoss()
        else:
            self.target = MS_SSIM(data_range=1.0)
        self.type = target
        self.lmbda = lmbda

    def forward(self, output, target, largest_ref=None, size=None):
        if size is not None:
            N, _, _, _ = target.size()
            H, W = size[0], size[1]
        else:
            N, _, H, W = target.size()

        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )

        out["hyper_bpp_loss"] = (torch.log(output["likelihoods"]['z']).sum() / (-math.log(2) * num_pixels))
        out["main_bpp_loss"] = (torch.log(output["likelihoods"]['y']).sum() / (-math.log(2) * num_pixels))
        if self.type == 'mse':
            out["mse_loss"] = self.target(output["x_hat"], target)
            out["loss"] = self.lmbda * 255**2 * out["mse_loss"] + out["bpp_loss"]
        else:
            out["ssim_loss"] = 1.0 - self.target(output["x_hat"], target)
            out["loss"] = self.lmbda * out["ssim_loss"] + out["bpp_loss"]
        return out


class EvaluateLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, target='mse'):
        super().__init__()
        self.target = nn.MSELoss()

    def forward(self, output, target, size=None):
        if size is not None:
            N, _, _, _ = target.size()
            H, W = size[0], size[1]
        else:
            N, _, H, W = target.size()

        out = {}
        num_pixels = N * H * W

        out["hyper_bpp_loss"] = []
        out["main_bpp_loss"] = []
        for likelihoods in output["likelihoods"]['z']:
            out["hyper_bpp_loss"].append(torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))

        for likelihoods in output["likelihoods"]['y']:
            out["main_bpp_loss"].append(torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))

        out["bpp_loss"] = []
        for i in range(len(out["main_bpp_loss"])):
            out["bpp_loss"].append(out["hyper_bpp_loss"][i] + out["main_bpp_loss"][i])

        out["mse_loss"] = []
        for x_hat in output["x_hat"]:
            out["mse_loss"].append(self.target(x_hat, target))
        return out