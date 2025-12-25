# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Sequence, Tuple

import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmengine.model import BaseModule
from torch import Tensor
from torch.nn.modules.batchnorm import _BatchNorm

from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from ..layers import CSPLayer
from .csp_darknet import SPPBottleneck


from mmseg.models.backbones.model_utils import MSFA, AFF, CSESP, SPASPP
from mmseg.models.nn_layers.eesp import SESP, DSESP
from mmseg.models.utils import DAPPM, BasicBlock, Bottleneck, resize, resize_to_tar
from mmseg.utils import OptConfigType
from mmseg.models.backbones.UNetFormer_GETB import GETBBlock
from mmseg.models.backbones.Laplacian import LaplacianProcessor

from mamba_ultralytics.nn.modules.mamba_yolo import VSSBlock, VisionClueMerge, SimpleStem
from mamba_ultralytics.nn.modules.block import SPPF


@MODELS.register_module()
class CSPNeXt(BaseModule):
    """CSPNeXt backbone used in RTMDet.

    Args:
        arch (str): Architecture of CSPNeXt, from {P5, P6}.
            Defaults to P5.
        expand_ratio (float): Ratio to adjust the number of channels of the
            hidden layer. Defaults to 0.5.
        deepen_factor (float): Depth multiplier, multiply number of
            blocks in CSP layer by this amount. Defaults to 1.0.
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Defaults to 1.0.
        out_indices (Sequence[int]): Output from which stages.
            Defaults to (2, 3, 4).
        frozen_stages (int): Stages to be frozen (stop grad and set eval
            mode). -1 means not freezing any parameters. Defaults to -1.
        use_depthwise (bool): Whether to use depthwise separable convolution.
            Defaults to False.
        arch_ovewrite (list): Overwrite default arch settings.
            Defaults to None.
        spp_kernel_sizes: (tuple[int]): Sequential of kernel sizes of SPP
            layers. Defaults to (5, 9, 13).
        channel_attention (bool): Whether to add channel attention in each
            stage. Defaults to True.
        conv_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            convolution layer. Defaults to None.
        norm_cfg (:obj:`ConfigDict` or dict): Dictionary to construct and
            config norm layer. Defaults to dict(type='BN', requires_grad=True).
        act_cfg (:obj:`ConfigDict` or dict): Config dict for activation layer.
            Defaults to dict(type='SiLU').
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        init_cfg (:obj:`ConfigDict` or dict or list[dict] or
            list[:obj:`ConfigDict`]): Initialization config dict.
    """
    # From left to right:
    # in_channels, out_channels, num_blocks, add_identity, use_spp
    arch_settings = {
        # 'P5': [[64, 128, 3, True, False], [128, 256, 6, True, False], #  ly修改2025-0318，源码
        #        [256, 512, 6, True, False], [512, 1024, 3, False, True]], #  ly修改2025-0318，源码
        # 'P5': [[64, 128, 3, True, False], [128, 256, 6, True, False]],
        'P5': [[64, 128, 3, True, False]],
        'brA': [[256, 256, 3, True, False], [256, 256, 3, False, True], [256, 256, 3, False, True]],  # ly修改2025-0327
        'P6': [[64, 128, 3, True, False], [128, 256, 6, True, False],
               [256, 512, 6, True, False], [512, 768, 3, True, False],
               [768, 1024, 3, False, True]]
    }

    def __init__(
        self,
        arch: str = 'P5',
        arch_brA: str = 'brA',
        deepen_factor: float = 1.0,
        widen_factor: float = 1.0,
        # out_indices: Sequence[int] = (2, 3, 4),   #  ly修改2025-0318，源码
        out_indices: Sequence[int] = (1),
        frozen_stages: int = -1,
        use_depthwise: bool = False,
        expand_ratio: float = 0.5,
        arch_ovewrite: dict = None,
        spp_kernel_sizes: Sequence[int] = (5, 9, 13),
        channel_attention: bool = True,
        conv_cfg: OptConfigType = None,
        norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg: ConfigType = dict(type='SiLU'),
        norm_eval: bool = False,
        planes=96,   # ly修改2025-0318
        align_corners: bool = False,
        init_cfg: OptMultiConfig = dict(
            type='Kaiming',
            layer='Conv2d',
            a=math.sqrt(5),
            distribution='uniform',
            mode='fan_in',
            nonlinearity='leaky_relu')
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        arch_setting = self.arch_settings[arch]
        arch_setting_brA = self.arch_settings[arch_brA]
        if arch_ovewrite:
            arch_setting = arch_ovewrite
        # assert set(out_indices).issubset(   #  ly修改2025-0318，源码
        #     i for i in range(len(arch_setting) + 1))   #  ly修改2025-0318，源码
        if frozen_stages not in range(-1, len(arch_setting) + 1):
            raise ValueError('frozen_stages must be in range(-1, '
                             'len(arch_setting) + 1). But received '
                             f'{frozen_stages}')

        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.use_depthwise = use_depthwise
        self.norm_eval = norm_eval
        self.align_corners = align_corners
        self.eesp_channels = [24, 48, 96, 96, 192, 512, 1024]   # ly修改2025-0318
        self.K = [4, 4, 4, 4, 4]   # ly修改2025-0318
        self.recept_limit = [13, 11, 9, 7, 5]   # ly修改2025-0318
        self.kernel_size = 3
        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule
        self.stem = nn.Sequential(
            ConvModule(
                3,
                int(arch_setting[0][0] * widen_factor // 2),
                3,
                padding=1,
                stride=2,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
            ConvModule(
                int(arch_setting[0][0] * widen_factor // 2),
                int(arch_setting[0][0] * widen_factor // 2),
                3,
                padding=1,
                stride=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
            ConvModule(
                int(arch_setting[0][0] * widen_factor // 2),
                int(arch_setting[0][0] * widen_factor),
                3,
                padding=1,
                stride=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg))
        self.layers = ['stem']

        for i, (in_channels, out_channels, num_blocks, add_identity,
                use_spp) in enumerate(arch_setting):
            in_channels = int(in_channels * widen_factor)
            out_channels = int(out_channels * widen_factor)
            num_blocks = max(round(num_blocks * deepen_factor), 1)
            stage = []
            conv_layer = conv(
                in_channels,
                out_channels,
                3,
                stride=2,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            stage.append(conv_layer)
            if use_spp:
                spp = SPPBottleneck(
                    out_channels,
                    out_channels,
                    kernel_sizes=spp_kernel_sizes,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg)
                stage.append(spp)
            csp_layer = CSPLayer(
                out_channels,
                out_channels,
                num_blocks=num_blocks,
                add_identity=add_identity,
                use_depthwise=use_depthwise,
                use_cspnext_block=True,
                expand_ratio=expand_ratio,
                channel_attention=channel_attention,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            stage.append(csp_layer)
            self.add_module(f'stage{i + 1}', nn.Sequential(*stage))
            self.layers.append(f'stage{i + 1}')

        # self.eesp_channels = [24, 48, 96, 192, 384, 512, 1024]   # ly修改2025-0318

        self.layerA2 = nn.Sequential(
            CSESP(self.eesp_channels[1], self.eesp_channels[1], self.eesp_channels[2], stride_1=2, Ker=[4, 4, 4], recept_limit=[13, 11, 9], Spatial=False),
        )

        self.layerA3 = nn.Sequential(
            CSESP(self.eesp_channels[2], self.eesp_channels[2], self.eesp_channels[2], stride_1=1, Ker=[4, 4, 4], recept_limit=[13, 11, 9], Spatial=False),
        )
        self.layerA4 = nn.Sequential(
            CSESP(self.eesp_channels[2], self.eesp_channels[2], self.eesp_channels[2], stride_1=1, Ker=[4, 4, 4], recept_limit=[13, 11, 9], Spatial=False),
        )
        self.layerA5 = nn.Sequential(
            CSESP(self.eesp_channels[2], self.eesp_channels[2], self.eesp_channels[2], stride_1=1, Ker=[4, 4, 4], recept_limit=[13, 11, 9], Spatial=False),
        )

        self.layerC3 = nn.Sequential(
            # DSESP(self.eesp_channels[2], self.eesp_channels[3], stride=2, k=self.K[2], r_lim=self.recept_limit[2])
            VisionClueMerge(dim=self.eesp_channels[2], out_dim=self.eesp_channels[3]),
            VSSBlock(in_channels=self.eesp_channels[3], hidden_dim=self.eesp_channels[3])
        )

        self.layerC4 = nn.Sequential(
            VisionClueMerge(dim=self.eesp_channels[3], out_dim=self.eesp_channels[4]),
            VSSBlock(in_channels=self.eesp_channels[4], hidden_dim=self.eesp_channels[4]))

        self.layerC5 = nn.Sequential(
            SPPBottleneck(self.eesp_channels[4], self.eesp_channels[4], kernel_sizes=spp_kernel_sizes, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg),
            VSSBlock(in_channels=self.eesp_channels[4], hidden_dim=self.eesp_channels[4]),
        )

        # self.layerA3 = nn.Sequential(
        #     VSSBlock(in_channels=self.eesp_channels[2], hidden_dim=self.eesp_channels[2]),
        # )
        # self.layerA4 = nn.Sequential(
        #     VSSBlock(in_channels=self.eesp_channels[2], hidden_dim=self.eesp_channels[2]),
        # )
        # self.layerA5 = nn.Sequential(
        #     VSSBlock(in_channels=self.eesp_channels[2], hidden_dim=self.eesp_channels[2]),
        # )
        # self.layerC3 = nn.Sequential(
        #     VisionClueMerge(dim=self.eesp_channels[2], out_dim=self.eesp_channels[3]),
        #     CSESP(self.eesp_channels[3], self.eesp_channels[3], self.eesp_channels[3], stride_1=1, Ker=[4, 4, 4], recept_limit=[13, 11, 9], Spatial=False),
        # )
        # self.layerC4 = nn.Sequential(
        #     VisionClueMerge(dim=self.eesp_channels[3], out_dim=self.eesp_channels[4]),
        #     CSESP(self.eesp_channels[4], self.eesp_channels[4], self.eesp_channels[4], stride_1=1, Ker=[4, 4, 4],
        #           recept_limit=[13, 11, 9], Spatial=False),
        # )
        # self.layerC5 = nn.Sequential(
        #     SPPBottleneck(self.eesp_channels[4], self.eesp_channels[4], kernel_sizes=spp_kernel_sizes, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg),
        #     CSESP(self.eesp_channels[4], self.eesp_channels[4], self.eesp_channels[4], stride_1=1, Ker=[4, 4, 4], recept_limit=[13, 11, 9], Spatial=False),
        # )

        # [24, 48, 96, 192, 384, 512, 1024]
        # 第一阶段融合
        # self.compression_CtoAs1 = ConvModule(self.eesp_channels[3], self.eesp_channels[2], kernel_size=1, norm_cfg=norm_cfg, act_cfg=None)
        self.compression_CtoAs1 = SESP(self.eesp_channels[3], self.eesp_channels[2], stride=1, k=self.K[2], r_lim=self.recept_limit[2])
        # self.down_AtoCs1 = ConvModule(self.eesp_channels[2], self.eesp_channels[3], kernel_size=self.kernel_size, stride=2, padding=1, norm_cfg=norm_cfg, act_cfg=None)
        # self.down_AtoCs1 = CSESP(self.eesp_channels[2], self.eesp_channels[2], self.eesp_channels[3], stride_1=2, Ker=[4, 4, 4], recept_limit=[13, 11, 9])
        self.down_AtoCs1 = DSESP(self.eesp_channels[2], self.eesp_channels[3], stride=2, k=self.K[2], r_lim=self.recept_limit[2])
        self.aff_CtoAs1 = MSFA(channels=self.eesp_channels[2])

        # 第二阶段融合
        # self.down_AtoCs2 = ConvModule(self.eesp_channels[2], self.eesp_channels[3], kernel_size=self.kernel_size, stride=2, padding=1, norm_cfg=norm_cfg, act_cfg=None)
        # self.down_AtoCs2 = CSESP(self.eesp_channels[2], self.eesp_channels[2], self.eesp_channels[3], stride_1=2, Ker=[4, 4, 4], recept_limit=[13, 11, 9])
        self.down_AtoCs2 = DSESP(self.eesp_channels[2], self.eesp_channels[3], stride=2, k=self.K[2], r_lim=self.recept_limit[2])
        # self.down_AtoCs3 = ConvModule(self.eesp_channels[3], self.eesp_channels[4], kernel_size=self.kernel_size, stride=2, padding=1, norm_cfg=norm_cfg, act_cfg=None)
        # self.down_AtoCs3 = CSESP(self.eesp_channels[3], self.eesp_channels[3], self.eesp_channels[4], stride_1=2, Ker=[4, 4, 4], recept_limit=[13, 11, 9])
        self.down_AtoCs3 = DSESP(self.eesp_channels[3], self.eesp_channels[4], stride=2, k=self.K[2], r_lim=self.recept_limit[2])
        # self.compression_CtoAs2 = ConvModule(self.eesp_channels[4], self.eesp_channels[3], kernel_size=1, norm_cfg=norm_cfg, act_cfg=None)
        self.compression_CtoAs2 = SESP(self.eesp_channels[4], self.eesp_channels[3], stride=1, k=self.K[2], r_lim=self.recept_limit[2])
        # self.compression_CtoAs3 = ConvModule(self.eesp_channels[3], self.eesp_channels[2], kernel_size=1, norm_cfg=norm_cfg, act_cfg=None)
        self.compression_CtoAs3 = SESP(self.eesp_channels[3], self.eesp_channels[2], stride=1, k=self.K[2], r_lim=self.recept_limit[2])
        self.aff_CtoAs2 = MSFA(channels=self.eesp_channels[2])

        # 第三阶段融合
        # self.down_AtoBs1 = ConvModule(self.eesp_channels[2], self.eesp_channels[3], kernel_size=self.kernel_size, stride=2, padding=1, norm_cfg=norm_cfg, act_cfg=None)
        # self.down_AtoBs1 = CSESP(self.eesp_channels[2], self.eesp_channels[2], self.eesp_channels[3], stride_1=2, Ker=[4, 4, 4], recept_limit=[13, 11, 9])
        self.down_AtoBs1 = DSESP(self.eesp_channels[2], self.eesp_channels[3], stride=2, k=self.K[2], r_lim=self.recept_limit[2])
        # self.compression_CtoBs1 = ConvModule(self.eesp_channels[4], self.eesp_channels[3], kernel_size=1, norm_cfg=norm_cfg, act_cfg=None)
        self.compression_CtoBs1 = SESP(self.eesp_channels[4], self.eesp_channels[3], stride=1, k=self.K[2], r_lim=self.recept_limit[2])
        self.aff_CtoAs3 = MSFA(channels=self.eesp_channels[3], kernel_size=3)

    def _freeze_stages(self) -> None:
        if self.frozen_stages >= 0:
            for i in range(self.frozen_stages + 1):
                m = getattr(self, self.layers[i])
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def train(self, mode=True) -> None:
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

    def forward(self, x: Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
        outs = []
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            # if i in self.out_indices:
            if layer_name == 'stage1':
                break  # 直接跳出循环

        #  此处输出 96*80*80，以下多个路径执行

        brA0 = self.layerA2(x)
        brA1 = self.layerA3(brA0)
        brC1 = self.layerC3(brA0)

        brC1_ = self.down_AtoCs1(brA1) + brC1
        brA1_ = self.aff_CtoAs1(resize_to_tar(self.compression_CtoAs1(brC1), brA1), brA1)

        brA2 = self.layerA4(brA1_)
        brC2 = self.layerC4(brC1_)

        brC2_ = self.down_AtoCs3(self.down_AtoCs2(brA2)) + brC2
        brA2_ = self.aff_CtoAs2(resize_to_tar(self.compression_CtoAs3(resize_to_tar(self.compression_CtoAs2(brC2), brA2)), brA2), brA2)

        brA3 = self.layerA5(brA2_)
        brB3 = self.aff_CtoAs3(self.down_AtoBs1(brA2_), resize_to_tar(self.compression_CtoBs1(brC2_), brC1_))
        brC3 = self.layerC5(brC2_)

        outs.append(brA3)
        outs.append(brB3)
        outs.append(brC3)

        return tuple(outs)
