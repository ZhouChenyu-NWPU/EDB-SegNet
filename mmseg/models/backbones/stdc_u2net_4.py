# Copyright (c) OpenMMLab. All rights reserved.
"""Modified from https://github.com/MichaelFan01/STDC-Seg."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule, ModuleList, Sequential

from mmseg.registry import MODELS
# from ..utils import resize
from mmseg.models.utils import resize
# from .bisenetv1 import AttentionRefinementModule
from mmseg.models.backbones.bisenetv1 import AttentionRefinementModule
from mmseg.models.backbones.u2net import RSU6, RSU5, RSU4, RSU4F

class STDCModule(BaseModule):
    """STDCModule.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels before scaling.
        stride (int): The number of stride for the first conv layer.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): The activation config for conv layers.
        num_convs (int): Numbers of conv layers.
        fusion_type (str): Type of fusion operation. Default: 'add'.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 norm_cfg=None,
                 act_cfg=None,
                 num_convs=4,
                 fusion_type='add',
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        assert num_convs > 1
        assert fusion_type in ['add', 'cat']
        self.stride = stride
        self.with_downsample = True if self.stride == 2 else False
        self.fusion_type = fusion_type

        self.layers = ModuleList()
        conv_0 = ConvModule(
            in_channels, out_channels // 2, kernel_size=1, norm_cfg=norm_cfg)

        if self.with_downsample:
            self.downsample = ConvModule(
                out_channels // 2,
                out_channels // 2,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=out_channels // 2,
                norm_cfg=norm_cfg,
                act_cfg=None)

            if self.fusion_type == 'add':
                self.layers.append(nn.Sequential(conv_0, self.downsample))
                self.skip = Sequential(
                    ConvModule(
                        in_channels,
                        in_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        groups=in_channels,
                        norm_cfg=norm_cfg,
                        act_cfg=None),
                    ConvModule(
                        in_channels,
                        out_channels,
                        1,
                        norm_cfg=norm_cfg,
                        act_cfg=None))
            else:
                self.layers.append(conv_0)
                self.skip = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.layers.append(conv_0)

        for i in range(1, num_convs):
            out_factor = 2**(i + 1) if i != num_convs - 1 else 2**i
            self.layers.append(
                ConvModule(
                    out_channels // 2**i,
                    out_channels // out_factor,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

    def forward(self, inputs):
        if self.fusion_type == 'add':
            out = self.forward_add(inputs)
        else:
            out = self.forward_cat(inputs)
        return out

    def forward_add(self, inputs):
        layer_outputs = []
        x = inputs.clone()
        for layer in self.layers:
            x = layer(x)
            layer_outputs.append(x)
        if self.with_downsample:
            inputs = self.skip(inputs)

        return torch.cat(layer_outputs, dim=1) + inputs

    def forward_cat(self, inputs):
        x0 = self.layers[0](inputs)
        layer_outputs = [x0]
        for i, layer in enumerate(self.layers[1:]):
            if i == 0:
                if self.with_downsample:
                    x = layer(self.downsample(x0))
                else:
                    x = layer(x0)
            else:
                x = layer(x)
            layer_outputs.append(x)
        if self.with_downsample:
            layer_outputs[0] = self.skip(x0)
        return torch.cat(layer_outputs, dim=1)


class FeatureFusionModule(BaseModule):
    """Feature Fusion Module. This module is different from FeatureFusionModule
    in BiSeNetV1. It uses two ConvModules in `self.attention` whose inter
    channel number is calculated by given `scale_factor`, while
    FeatureFusionModule in BiSeNetV1 only uses one ConvModule in
    `self.conv_atten`.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        scale_factor (int): The number of channel scale factor.
            Default: 4.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): The activation config for conv layers.
            Default: dict(type='ReLU').
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 scale_factor=4,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        channels = out_channels // scale_factor
        self.conv0 = ConvModule(
            in_channels, out_channels, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            ConvModule(
                out_channels,
                channels,
                1,
                norm_cfg=None,
                bias=False,
                act_cfg=act_cfg),
            ConvModule(
                channels,
                out_channels,
                1,
                norm_cfg=None,
                bias=False,
                act_cfg=None), nn.Sigmoid())

    def forward(self, spatial_inputs, context_inputs):
        inputs = torch.cat([spatial_inputs, context_inputs], dim=1)
        x = self.conv0(inputs)
        attn = self.attention(x)
        x_attn = x * attn
        return x_attn + x


@MODELS.register_module()
class STDCNet(BaseModule):
    """This backbone is the implementation of `Rethinking BiSeNet For Real-time
    Semantic Segmentation <https://arxiv.org/abs/2104.13188>`_.

    Args:
        stdc_type (int): The type of backbone structure,
            `STDCNet1` and`STDCNet2` denotes two main backbones in paper,
            whose FLOPs is 813M and 1446M, respectively.
        in_channels (int): The num of input_channels.
        channels (tuple[int]): The output channels for each stage.
        bottleneck_type (str): The type of STDC Module type, the value must
            be 'add' or 'cat'.
        norm_cfg (dict): Config dict for normalization layer.
        act_cfg (dict): The activation config for conv layers.
        num_convs (int): Numbers of conv layer at each STDC Module.
            Default: 4.
        with_final_conv (bool): Whether add a conv layer at the Module output.
            Default: True.
        pretrained (str, optional): Model pretrained path. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.

    Example:
        >>> import torch
        >>> stdc_type = 'STDCNet1'
        >>> in_channels = 3
        >>> channels = (32, 64, 256, 512, 1024)
        >>> bottleneck_type = 'cat'
        >>> inputs = torch.rand(1, 3, 1024, 2048)
        >>> self = STDCNet(stdc_type, in_channels,
        ...                 channels, bottleneck_type).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 256, 128, 256])
        outputs[1].shape = torch.Size([1, 512, 64, 128])
        outputs[2].shape = torch.Size([1, 1024, 32, 64])
    """

    arch_settings = {
        'STDCNet1': [(2, 1), (2, 1), (2, 1)],
        'STDCNet2': [(2, 1, 1, 1), (2, 1, 1, 1, 1), (2, 1, 1)]
    }


    def __init__(self,
                 stdc_type,
                 in_channels,
                 channels,
                 bottleneck_type,
                 norm_cfg,
                 act_cfg,
                 num_convs=4,
                 in_ch=3,
                 out_ch=1,
                 with_final_conv=False,
                 pretrained=None,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        self.stage2 = RSU6(in_ch, 32, 128)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU5(128,64,256)
        self.pool34 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage4 = RSU4(256,128,256)
        self.pool45 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage5 = RSU4F(256,256,512)
        self.pool56 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage6 = RSU4F(512, 256, 1024)
        self.pool6 = nn.MaxPool2d(2, stride=2, ceil_mode=True)


    def forward(self, x):
        hx = x

        # stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        # stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        # stage 4
        hx4_1 = self.stage4(hx)
        hx4 = self.pool45(hx4_1)

        # stage 5
        hx5_1 = self.stage5(hx4)
        hx5 = self.pool56(hx5_1)

        # stage 6
        hx6_1 = self.stage6(hx5)
        hx6 = self.pool6(hx6_1)

        outs = [F.sigmoid(hx4), F.sigmoid(hx5), F.sigmoid(hx6)]

        return tuple(outs)


@MODELS.register_module()
class STDCContextPathNet(BaseModule):
    """STDCNet with Context Path. The `outs` below is a list of three feature
    maps from deep to shallow, whose height and width is from small to big,
    respectively. The biggest feature map of `outs` is outputted for
    `STDCHead`, where Detail Loss would be calculated by Detail Ground-truth.
    The other two feature maps are used for Attention Refinement Module,
    respectively. Besides, the biggest feature map of `outs` and the last
    output of Attention Refinement Module are concatenated for Feature Fusion
    Module. Then, this fusion feature map `feat_fuse` would be outputted for
    `decode_head`. More details please refer to Figure 4 of original paper.

    Args:
        backbone_cfg (dict): Config dict for stdc backbone.
        last_in_channels (tuple(int)), The number of channels of last
            two feature maps from stdc backbone. Default: (1024, 512).
        out_channels (int): The channels of output feature maps.
            Default: 128.
        ffm_cfg (dict): Config dict for Feature Fusion Module. Default:
            `dict(in_channels=512, out_channels=256, scale_factor=4)`.
        upsample_mode (str): Algorithm used for upsampling:
                ``'nearest'`` | ``'linear'`` | ``'bilinear'`` | ``'bicubic'`` |
                ``'trilinear'``. Default: ``'nearest'``.
        align_corners (str): align_corners argument of F.interpolate. It
            must be `None` if upsample_mode is ``'nearest'``. Default: None.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.

    Return:
        outputs (tuple): The tuple of list of output feature map for
            auxiliary heads and decoder head.
    """

    def __init__(self,
                 backbone_cfg,
                 last_in_channels=(1024, 512),
                 out_channels=128,
                 ffm_cfg=dict(
                     in_channels=512, out_channels=256, scale_factor=4),
                 upsample_mode='nearest',
                 align_corners=None,
                 norm_cfg=dict(type='BN'),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.backbone = MODELS.build(backbone_cfg)
        self.arms = ModuleList()
        self.convs = ModuleList()
        for channels in last_in_channels:
            self.arms.append(AttentionRefinementModule(channels, out_channels))
            self.convs.append(
                ConvModule(
                    out_channels,
                    out_channels,
                    3,
                    padding=1,
                    norm_cfg=norm_cfg))
        self.conv_avg = ConvModule(
            last_in_channels[0], out_channels, 1, norm_cfg=norm_cfg)

        self.ffm = FeatureFusionModule(**ffm_cfg)

        self.upsample_mode = upsample_mode
        self.align_corners = align_corners

    def forward(self, x):
        outs = list(self.backbone(x))
        avg = F.adaptive_avg_pool2d(outs[-1], 1)
        avg_feat = self.conv_avg(avg)

        feature_up = resize(
            avg_feat,
            size=outs[-1].shape[2:],
            mode=self.upsample_mode,
            align_corners=self.align_corners)
        arms_out = []
        for i in range(len(self.arms)):
            x_arm = self.arms[i](outs[len(outs) - 1 - i]) + feature_up
            feature_up = resize(
                x_arm,
                size=outs[len(outs) - 1 - i - 1].shape[2:],
                mode=self.upsample_mode,
                align_corners=self.align_corners)
            feature_up = self.convs[i](feature_up)
            arms_out.append(feature_up)

        feat_fuse = self.ffm(outs[0], arms_out[1])

        # The `outputs` has four feature maps.
        # `outs[0]` is outputted for `STDCHead` auxiliary head.
        # Two feature maps of `arms_out` are outputted for auxiliary head.
        # `feat_fuse` is outputted for decoder head.
        outputs = [outs[0]] + list(arms_out) + [feat_fuse]
        return tuple(outputs)

# net = STDCNet('STDCNet1',3,(32, 64, 256, 512, 1024),'cat',None,None)
# img = torch.rand([2,3,256,256])
# net(img)