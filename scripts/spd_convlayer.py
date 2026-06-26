# spd.py
import torch
import torch.nn as nn
from ultralytics.nn.modules import Conv

class SPDConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 bias=False, use_modulation=False):
        super().__init__()
        self.use_modulation = use_modulation
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        
        if use_modulation:
            self.scale = nn.Parameter(torch.ones(1, out_channels, 1, 1))
            self.shift = nn.Parameter(torch.zeros(1, out_channels, 1, 1))

    def forward(self, x):
        out = self.conv(x)
        if self.use_modulation:
            out = out * self.scale + self.shift
        return out

def replace_conv_with_spd(module, use_modulation=True):
    """
    Recursively replace nested conv layers with SPDConv.
    """
    for name, child in list(module.named_children()):
        # Check if the child has a 'conv' attribute that is a Conv2d layer
        if hasattr(child, "conv") and isinstance(child.conv, nn.Conv2d):
            old_conv = child.conv
            new_spd = SPDConv(
                old_conv.in_channels,
                old_conv.out_channels,
                old_conv.kernel_size[0],
                old_conv.stride[0],
                old_conv.padding[0],
                bias=(old_conv.bias is not None),
                use_modulation=use_modulation
            )
            # Transfer weights
            new_spd.conv.weight.data.copy_(old_conv.weight.data)
            if old_conv.bias is not None:
                new_spd.conv.bias.data.copy_(old_conv.bias.data)
            
            child.conv = new_spd
        else:
            replace_conv_with_spd(child, use_modulation)

