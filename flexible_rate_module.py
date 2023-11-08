import torch.nn as nn
import torch

class Gain_Module(nn.Module):
    def __init__(self, n=6, N=192, bias=False):
        """
        n: number of scales for quantization levels
        N: number of channels
        """
        super(Gain_Module, self).__init__()
        self.gain_matrix = nn.Parameter(torch.ones(size=[n, N], dtype=torch.float32), requires_grad=True)
        self.bias = bias
        self.ch = N
        self.isFlexibleRate = True if n > 1 else False
        if bias:
            self.bias = nn.Parameter(torch.ones(size=[n, N], dtype=torch.float32), requires_grad=True)

    def forward(self, x, level=None, coeff=None, isInterpolation=False):
        """  level one dim data, coeff two dims datas """
        if isinstance(level, type(x)):
            if isInterpolation:
                coeff = coeff.unsqueeze(1)
                gain1 = self.gain_matrix[level, :]
                gain2 = self.gain_matrix[level + 1, :]
                gain = ((torch.abs(gain1) ** coeff) *
                        (torch.abs(gain2) ** (1 - coeff))).unsqueeze(2).unsqueeze(3)
            else:
                gain = torch.abs(self.gain_matrix[level]).unsqueeze(2).unsqueeze(3)
        else:
            if isInterpolation:
                gain1 = self.gain_matrix[level, :]
                gain2 = self.gain_matrix[level + 1, :]
                gain = ((torch.abs(gain1) ** coeff) *
                        (torch.abs(gain2) ** (1 - coeff))).unsqueeze(0).unsqueeze(2).unsqueeze(3)
            else:
                gain = torch.abs(self.gain_matrix[level, :]).unsqueeze(0).unsqueeze(2).unsqueeze(3)

        if self.isFlexibleRate:
            rescaled_latent = gain * x
            if self.bias:
                rescaled_latent += self.bias[level]
            return rescaled_latent
        else:
            return x
