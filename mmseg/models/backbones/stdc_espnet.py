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

class EESPNet(nn.Module):
    '''
    This class defines the ESPNetv2 architecture for the ImageNet classification
    '''

    def __init__(self, args):
        '''
        :param classes: number of classes in the dataset. Default is 1000 for the ImageNet dataset
        :param s: factor that scales the number of output feature maps
        '''
        super().__init__()

        # ====================
        # Network configuraiton
        # ====================
        try:
            num_classes = args.num_classes
        except:
            # if not specified, default to 1000 for imageNet
            num_classes = 1000  # 1000 for imagenet

        try:
            channels_in = args.channels
        except:
            # if not specified, default to RGB (3)
            channels_in = 3

        s = args.s
        if not s in config_all.sc_ch_dict.keys():
            print_error_message('Model at scale s={} is not suppoerted yet'.format(s))
            exit(-1)

        out_channel_map = config_all.sc_ch_dict[args.s]
        reps_at_each_level = config_all.rep_layers

        recept_limit = config_all.recept_limit  # receptive field at each spatial level
        K = [config_all.branches]*len(recept_limit) # No. of parallel branches at different level

        # True for the shortcut connection with input
        self.input_reinforcement = config_all.input_reinforcement

        assert len(K) == len(recept_limit), 'Length of branching factor array and receptive field array should be the same.'

        self.level1 = CBR(channels_in, out_channel_map[0], 3, 2)  # 112 L1

        self.level2_0 = DownSampler(out_channel_map[0], out_channel_map[1], k=K[0], r_lim=recept_limit[0], reinf=self.input_reinforcement)  # out = 56

        self.level3_0 = DownSampler(out_channel_map[1], out_channel_map[2], k=K[1], r_lim=recept_limit[1], reinf=self.input_reinforcement) # out = 28
        self.level3 = nn.ModuleList()
        for i in range(reps_at_each_level[1]):
            self.level3.append(EESP(out_channel_map[2], out_channel_map[2], stride=1, k=K[2], r_lim=recept_limit[2]))

        self.level4_0 = DownSampler(out_channel_map[2], out_channel_map[3], k=K[2], r_lim=recept_limit[2], reinf=self.input_reinforcement) #out = 14
        self.level4 = nn.ModuleList()
        for i in range(reps_at_each_level[2]):
            self.level4.append(EESP(out_channel_map[3], out_channel_map[3], stride=1, k=K[3], r_lim=recept_limit[3]))

        self.level5_0 = DownSampler(out_channel_map[3], out_channel_map[4], k=K[3], r_lim=recept_limit[3]) #7
        self.level5 = nn.ModuleList()
        for i in range(reps_at_each_level[3]):
            self.level5.append(EESP(out_channel_map[4], out_channel_map[4], stride=1, k=K[4], r_lim=recept_limit[4]))

        # expand the feature maps using depth-wise convolution followed by group point-wise convolution
        self.level5.append(CBR(out_channel_map[4], out_channel_map[4], 3, 1, groups=out_channel_map[4]))
        self.level5.append(CBR(out_channel_map[4], out_channel_map[5], 1, 1, groups=K[4]))

        self.classifier = nn.Linear(out_channel_map[5], num_classes)
        self.config = out_channel_map
        self.init_params()

    def init_params(self):
        '''
        Function to initialze the parameters
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, input, p=0.2):
        '''
        :param input: Receives the input RGB image
        :return: a C-dimensional vector, C=# of classes
        '''
        out_l1 = self.level1(input)  # 112
        if not self.input_reinforcement:
            del input
            input = None

        out_l2 = self.level2_0(out_l1, input)  # 56

        out_l3_0 = self.level3_0(out_l2, input)  # down-sample
        for i, layer in enumerate(self.level3):
            if i == 0:
                out_l3 = layer(out_l3_0)
            else:
                out_l3 = layer(out_l3)

        out_l4_0 = self.level4_0(out_l3, input)  # down-sample
        for i, layer in enumerate(self.level4):
            if i == 0:
                out_l4 = layer(out_l4_0)
            else:
                out_l4 = layer(out_l4)

        out_l5_0 = self.level5_0(out_l4)  # down-sample
        for i, layer in enumerate(self.level5):
            if i == 0:
                out_l5 = layer(out_l5_0)
            else:
                out_l5 = layer(out_l5)

        output_g = F.adaptive_avg_pool2d(out_l5, output_size=1)
        output_g = F.dropout(output_g, p=p, training=self.training)
        output_1x1 = output_g.view(output_g.size(0), -1)

        return self.classifier(output_1x1)

class EfficientPyrPool(nn.Module):
    """Efficient Pyramid Pooling Module"""

    def __init__(self, in_planes, proj_planes, out_planes, scales=[2.0, 1.5, 1.0, 0.5, 0.1], last_layer_br=True):
        super(EfficientPyrPool, self).__init__()
        self.stages = nn.ModuleList()
        scales.sort(reverse=True)

        self.projection_layer = CBR(in_planes, proj_planes, 1, 1)
        for _ in enumerate(scales):
            self.stages.append(nn.Conv2d(proj_planes, proj_planes, kernel_size=3, stride=1, padding=1, bias=False, groups=proj_planes))

        self.merge_layer = nn.Sequential(
            # perform one big batch normalization instead of p small ones
            BR(proj_planes * len(scales)),
            Shuffle(groups=len(scales)),
            CBR(proj_planes * len(scales), proj_planes, 3, 1, groups=proj_planes),
            nn.Conv2d(proj_planes, out_planes, kernel_size=1, stride=1, bias=not last_layer_br),
        )
        if last_layer_br:
            self.br = BR(out_planes)
        self.last_layer_br = last_layer_br
        self.scales = scales

    def forward(self, x):
        hs = []
        x = self.projection_layer(x)
        height, width = x.size()[2:]
        for i, stage in enumerate(self.stages):
            h_s = int(math.ceil(height * self.scales[i]))
            w_s = int(math.ceil(width * self.scales[i]))
            h_s = h_s if h_s > 5 else 5
            w_s = w_s if w_s > 5 else 5
            if self.scales[i] < 1.0:
                h = F.adaptive_avg_pool2d(x, output_size=(h_s, w_s))
                h = stage(h)
                h = F.interpolate(h, (height, width), mode='bilinear', align_corners=True)
            elif self.scales[i] > 1.0:
                h = F.interpolate(x, (h_s, w_s), mode='bilinear', align_corners=True)
                h = stage(h)
                h = F.adaptive_avg_pool2d(h, output_size=(height, width))
            else:
                h = stage(x)
            hs.append(h)

        out = torch.cat(hs, dim=1)
        out = self.merge_layer(out)
        if self.last_layer_br:
            return self.br(out)
        return out

class EfficientPWConv(nn.Module):
    def __init__(self, nin, nout):
        super(EfficientPWConv, self).__init__()
        self.wt_layer = nn.Sequential(
                        nn.AdaptiveAvgPool2d(output_size=1),
                        nn.Conv2d(nin, nout, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
                        nn.Sigmoid()
                    )

        self.groups = math.gcd(nin, nout)
        self.expansion_layer = CBR(nin, nout, kSize=3, stride=1, groups=self.groups)

        self.out_size = nout
        self.in_size = nin

    def forward(self, x):
        wts = self.wt_layer(x)
        x = self.expansion_layer(x)
        x = x * wts
        return x

    def __repr__(self):
        s = '{name}(in_channels={in_size}, out_channels={out_size})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

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
class STDCESPNet(BaseModule):
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
                 args,
                 stdc_type,
                 in_channels,
                 channels,
                 bottleneck_type,
                 norm_cfg,
                 act_cfg,
                 num_convs=4,
                 classes=21,
                 dataset='pascal',
                 with_final_conv=False,
                 pretrained=None,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        config = [32, 128, 256, 512, 1024, 1280]

        dec_feat_dict = {
            'pascal': 16,
            'city': 16,
            'coco': 32
        }
        base_dec_planes = dec_feat_dict[dataset]
        dec_planes = [4 * base_dec_planes, 3 * base_dec_planes, 2 * base_dec_planes, classes]
        pyr_plane_proj = min(classes // 2, base_dec_planes)

        self.base_net = EESPNet(args)

        self.bu_dec_l1 = EfficientPyrPool(in_planes=config[5], proj_planes=pyr_plane_proj,
                                          out_planes=dec_planes[0])
        self.bu_dec_l2 = EfficientPyrPool(in_planes=dec_planes[0], proj_planes=pyr_plane_proj,
                                          out_planes=dec_planes[2])
        self.bu_dec_l3 = EfficientPyrPool(in_planes=dec_planes[0], proj_planes=pyr_plane_proj,
                                          out_planes=dec_planes[2])
        self.bu_dec_l4 = EfficientPyrPool(in_planes=dec_planes[2], proj_planes=pyr_plane_proj,
                                          out_planes=dec_planes[3], last_layer_br=False)

        self.merge_enc_dec_l1 = EfficientPWConv(config[3], dec_planes[0])
        self.merge_enc_dec_l2 = EfficientPWConv(config[2], dec_planes[0])
        self.merge_enc_dec_l3 = EfficientPWConv(dec_planes[2], config[3])  # 32->512
        self.merge_enc_dec_l4 = EfficientPWConv(config[3], config[2])  # 512->256

        self.bu_br_l2 = nn.Sequential(nn.BatchNorm2d(dec_planes[0]),
                                      nn.PReLU(dec_planes[0])
                                      )
        self.bu_br_l3 = nn.Sequential(nn.BatchNorm2d(dec_planes[0]),
                                      nn.PReLU(dec_planes[0])
                                      )
        self.bu_br_l4 = nn.Sequential(nn.BatchNorm2d(dec_planes[2]),
                                      nn.PReLU(dec_planes[2])
                                      )

    def upsample(self, x):
        return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        x_size = (x.size(2), x.size(3))
        enc_out_l1 = self.base_net.level1(x)  # 112
        # if not self.base_net.input_reinforcement:
        #     del x
        #     x = None

        enc_out_l2 = self.base_net.level2_0(enc_out_l1, x)  # 56

        enc_out_l3_0 = self.base_net.level3_0(enc_out_l2, x)  # down-sample
        for i, layer in enumerate(self.base_net.level3):
            if i == 0:
                enc_out_l3 = layer(enc_out_l3_0)
            else:
                enc_out_l3 = layer(enc_out_l3)

        enc_out_l4_0 = self.base_net.level4_0(enc_out_l3, x)  # down-sample
        for i, layer in enumerate(self.base_net.level4):
            if i == 0:
                enc_out_l4 = layer(enc_out_l4_0)
            else:
                enc_out_l4 = layer(enc_out_l4)

        enc_out_l5_0 = self.base_net.level5_0(enc_out_l4, x)  # down-sample
        for i, layer in enumerate(self.base_net.level5):
            if i == 0:
                enc_out_l5 = layer(enc_out_l5_0)
            else:
                enc_out_l5 = layer(enc_out_l5)

        return F.interpolate(enc_out_l5_0, size=x_size, mode='bilinear', align_corners=True)



@MODELS.register_module()
class STDCContextPathNet11(BaseModule):
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

net = STDCESPNet('STDCNet1',3,(32, 64, 256, 512, 1024),'cat', None,None)
# img = torch.rand([2,3,256,256])
img = torch.rand([2,3,512,1024])
net(img)