import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self):
        super(BasicBlock, self).__init__()
        self.active_out_channel = None

    def forward(self, x):
        raise NotImplementedError

    def initialize(self):
        for m in self.modules():
            pass

    @property
    def module_str(self):
        raise NotImplementedError

    @property
    def config(self):
        raise NotImplementedError

    def set_active_channels(self):
        raise NotImplementedError


def set_exist_attr(m, attr, value):
    if hasattr(m, attr):
        setattr(m, attr, value)
