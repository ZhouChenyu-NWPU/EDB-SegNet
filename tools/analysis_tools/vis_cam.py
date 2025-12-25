from argparse import ArgumentParser
import os

import numpy as np
import torch
import torch.nn.functional as F
from mmengine import Config
from mmengine.model import revert_sync_batchnorm
from PIL import Image
from pytorch_grad_cam import GradCAM, XGradCAM
from pytorch_grad_cam.utils.image import preprocess_image, show_cam_on_image

from mmseg.apis import inference_model, init_model, show_result_pyplot
from mmseg.utils import register_all_modules

import re

def print_named_modules(model, prefix=''):
    for name, module in model.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        print(full_name, ":", module.__class__.__name__)
        print_named_modules(module, prefix=full_name)


def sanitize_layer_name(layer_name):
    # 将 . / [ ] 替换成 _
    safe_name = re.sub(r'[./\[\]]', '_', layer_name)
    return safe_name


class SemanticSegmentationTarget:
    """wrap the model."""
    def __init__(self, category, mask, size):
        self.category = category
        self.mask = torch.from_numpy(mask)
        self.size = size
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()

    def __call__(self, model_output):
        model_output = F.interpolate(
            model_output, size=self.size, mode='bilinear', align_corners=True)
        model_output = torch.squeeze(model_output, dim=0)
        return (model_output[self.category, :, :] * self.mask).sum()

"""
/media/ly/AddDisk/InstanceSeg/ddr_and_pid/mmsegmentation_11g07/configs/ddrnet/ddrnet_23-slim_in1k-pre_2xb6-120k_cityscapes-1024x1024.py
/media/ly/AddDisk/InsSegData/pid_and_ddr_out/ddr_fordata_11g15/iter_80000.pth
"""

def main():
    parser = ArgumentParser()

    # 设置预测文件夹路径
    parser.add_argument('--input_dir', default='/media/zcy/zyy/11_2RELITU/7', help='Input directory containing images')

    # 设置配置文件路径
    parser.add_argument('--config',
                        default='/media/zcy/zyy/yuanma/mmsegmentation_11g07f15_1_nolaplac/configs/ddrnet/ddrnet_23-slim_in1k-pre_2xb6-120k_cityscapes-1024x1024.py',
                        help='Config file')

    # 设置权重文件路径
    parser.add_argument('--checkpoint', default='/media/zcy/zyy/mmsegmentation_11g07/outputsf15_1c/stdc_u2net_4/iter_80000.pth',
                        help='Checkpoint file')

    # 设置保存输出的热力图文件夹路径
    parser.add_argument(
        '--out-dir',
        default='/media/zcy/zyy/语义分割实验/11.2补充热力图两支路区别/7精选',
        help='Directory to save CAM images')

    # 设置需要输出网络特定层的热力图
    parser.add_argument(
        '--target-layers',
        # default='decode_head.head',
        # default='backbone.gltb3.attn',
        # default='backbone.layer5_.module_act',
        # default='backbone.compression_2',
        # default='backbone.gltb3.mlp.drop',
        # default='backbone.conv_2',
        # default='backbone.aff2',
        # default='backbone.compression_2',
        # default='backbone.spatial_branch_layers[1][0]',
        # default='backbone.layer3[0]',
        # default='backbone.gltb1',
        #default='backbone.layer5_',
        #11.2两支路热力图补充层数
        #default='backbone.layer3',
        #default='backbone.layer4[0].conv_list[0].proj_1x1.conv',
        #default='backbone.layerC3[1]',
        #default='backbone.layer3[0].conv_list[0].conv_1x1_exp.conv',
        #default='backbone.layer3[0].conv_list[2].spp_dw_v2[3].conv',
        #default='backbone.layer4[0].conv_list[2].proj_1x1.conv',
        #default='backbone.layer5',
        #default='backbone.layer5.conv_1x1_exp.conv',
        #default='backbone.layer5_',
        #default='backbone.layer5_.conv_1x1_exp.conv',
        #default='backbone.layerC3[1].proj_conv[0]',
        #default='backbone.layerC3[1].lsblock.fc3',
        #default='backbone.spatial_branch_layers[0][0]',
        #default='backbone.spatial_branch_layers[1][0]',
        #default='backbone.spatial_branch_layers[1][0].attention',
        #default='backbone.spatial_branch_layers[1][0].conv_list[2].conv_1x1_exp.conv',
        #default='backbone.spatial_branch_layers[1][0].conv_list[2].spp_dw_v2[3].conv',
        #default='backbone.spatial_branch_layers[1][0].conv_list[0].proj_1x1.conv ',
        #default='backbone.layer5_.proj_1x1.conv',
        #default='backbone.layer5.spp_dw_v2[3].conv',
        #default='backbone.layer5.spp_dw[3].conv',
        #default='backbone.aff2.local_att[0]',
        #default='backbone.aff2.context3[4]',

        default='backbone.aff1',
        #default='backbone.aff1.conv[0]',
        #default='backbone.aff1.local_att[0]',
        #default='backbone.aff2',
        #default='backbone.aff2.conv[0]',
        #default='backbone.aff2.local_att[0]',
        #default='backbone.layer3',
        #default='backbone.layer3[0].conv_list[0].conv_1x1_exp.conv',
        #default='backbone.layer3[0].conv_list[2].spp_dw_v2[3].conv',
        #default='backbone.layer4',
        #default='backbone.layer4[0].conv_list[0].conv_1x1_exp.conv',
        #default='backbone.layer4[0].conv_list[2].spp_dw_v2[3].conv',
        #default='backbone.layer5',
        #default='backbone.layer5.conv_1x1_exp.conv',
        #default='backbone.layer5.proj_1x1.conv',
        #default='backbone.layer5_',
        #default='backbone.layer5_.conv_1x1_exp.conv',
        #default='backbone.layer5_.proj_1x1.conv',
        #default='backbone.spatial_branch_layers[0][0]',
        #default='backbone.spatial_branch_layers[0][0].conv_list[2].spp_dw_v2[3].conv',
        #default='backbone.spatial_branch_layers[0][0].conv_list[0].proj_1x1.conv ',
        #default='backbone.spatial_branch_layers[1][0]',
        #default='backbone.spatial_branch_layers[1][0].conv_list[2].spp_dw_v2[3].conv',
        #default='backbone.spatial_branch_layers[1][0].conv_list[0].proj_1x1.conv ',

        help='Target layers to visualize CAM')

    # Grad-CAM为类激活加权映射，所以需要设置固定类别的索引来绘制，需要与传入网络训练时候一致
    parser.add_argument(
        '--category-index', default='1', help='Category to visualize CAM')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()

    # 创建保存目录
    os.makedirs(args.out_dir, exist_ok=True)

    # build the model from a config file and a checkpoint file
    register_all_modules()
    model = init_model(args.config, args.checkpoint, device=args.device)
    if args.device == 'cpu':
        model = revert_sync_batchnorm(model)

    # 将模型移至指定设备（GPU 或 CPU） 修改 2024-12-01 ly
    device = torch.device(args.device)
    model = model.to(device)
    # print("model:", model)
    #print_named_modules(model)

    # 列出输入目录中的所有图片文件
    input_files = [f for f in os.listdir(args.input_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

    for input_file in input_files:
        input_path = os.path.join(args.input_dir, input_file)

        # test a single image
        result = inference_model(model, input_path)

        # result data conversion
        prediction_data = result.pred_sem_seg.data
        pre_np_data = prediction_data.cpu().numpy().squeeze(0)

        target_layers = args.target_layers
        target_layers = [eval(f'model.{target_layers}')]
        
        category = int(args.category_index)
        mask_float = np.float32(pre_np_data == category)

        # data processing
        image = np.array(Image.open(input_path).convert('RGB'))
        height, width = image.shape[0], image.shape[1]
        rgb_img = np.float32(image) / 255
        config = Config.fromfile(args.config)
        image_mean = config.data_preprocessor['mean']
        image_std = config.data_preprocessor['std']
        
        # 将输入图像转换为张量并移到指定设备
        input_tensor = preprocess_image(
            rgb_img,
            mean=[x / 255 for x in image_mean],
            std=[x / 255 for x in image_std]
        ).to(device)

        # Grad CAM(Class Activation Maps)
        targets = [
            SemanticSegmentationTarget(category, mask_float, (height, width))
        ]
        # cam_file_name = os.path.splitext(input_file)[0] + '_cam.jpg'  # 使用输入图片文件名构造保存文件名
        # 获取 target_layers，替换非法字符用于文件名
        my_target_layers = args.target_layers  # 如 "backbone.spatial_branch_layers[1][0]"
        safe_layer_name = sanitize_layer_name(my_target_layers)
        cam_file_name = safe_layer_name + '.jpg'

        cam_file_path = os.path.join(args.out_dir, cam_file_name)

        # Can also be LayerCAM, XGradCAM, GradCAMPlusPlus, EigenCAM, EigenGradCAM，这里可以根据需要选择不同的特征图生成方式
        with GradCAM(
                model=model,
                target_layers=target_layers,
        ) as cam:
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
            cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

            # 保存 CAM 图片
            Image.fromarray(cam_image).save(cam_file_path)


if __name__ == '__main__':
    main()
