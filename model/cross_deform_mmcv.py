import torch
import torch.nn as nn

from mmcv.cnn import ConvModule
from mmcv.ops import ModulatedDeformConv2d, modulated_deform_conv2d



class cross_deform(nn.Module):
    def __init__(self, channels = 256, norm =None) -> None:
        super().__init__()
        self.offset = ConvModule(channels, channels, kernel_size=1, stride=1,padding=0,bias=False, norm_cfg = norm)
        self.mdcn = ModulatedDeformConv2dPack(channels, channels, 3, stride=1, padding=1, dilation=1, deform_groups=8)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, feat_u, feat_l):
        # l is lower, u is upper after resizing
        offset_in = self.offset(feat_l)
        feat_align = self.relu(self.mdcn(feat_u, offset_in))
        return feat_align 


class ModulatedDeformConv2dPack(ModulatedDeformConv2d):
    """A ModulatedDeformable Conv Encapsulation that acts as normal Conv
    layers.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int): Same as nn.Conv2d, while tuple is not supported.
        padding (int): Same as nn.Conv2d, while tuple is not supported.
        dilation (int): Same as nn.Conv2d, while tuple is not supported.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
    """

    def __init__(self, *args, **kwargs):
        super(ModulatedDeformConv2dPack, self).__init__(*args, **kwargs)
        self.conv_offset = nn.Conv2d(
            self.in_channels,
            self.deform_groups * 3 * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=True)
        self.init_weights()

    def init_weights(self):
        super(ModulatedDeformConv2dPack, self).init_weights()
        if hasattr(self, 'conv_offset'):
            self.conv_offset.weight.data.zero_()
            self.conv_offset.bias.data.zero_()

    def forward(self, x, offset_f):
        out = self.conv_offset(offset_f)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return modulated_deform_conv2d(x, offset, mask, self.weight, self.bias,
                                       self.stride, self.padding,
                                       self.dilation, self.groups,
                                       self.deform_groups)
