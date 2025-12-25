# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from mmdet.utils import setup_cache_size_limit_of_dynamo

"""
/media/ly/AddDisk/InstanceSeg2025/mmdetection/configs/rtmdet/rtmdet-ins_tiny_8xb32-300e_coco.py
/media/ly/AddDisk/InstanceSeg2025/mmdetection/configs/solov2/solov2_r50_fpn_1x_coco.py
/media/ly/AddDisk/InstanceSeg2025/mmdetection/configs/yolox/yolox_tiny_8xb8-300e_coco.py
/media/ly/AddDisk/InstanceSeg2025/mmdetection/configs/condinst/condinst_r50_fpn_ms-poly-90k_coco_instance.py
/media/ly/AddDisk/InstanceSeg2025/mmdetection/projects/SparseInst/configs/sparseinst_r50_iam_8xb8-ms-270k_coco.py
/media/ly/AddDisk/InstanceSeg2025/mmdetection/configs/convnext/cascade-mask-rcnn_convnext-t-p4-w7_fpn_4conv1fc-giou_amp-ms-crop-3x_coco.py
/media/ly/AddDisk/InstanceSeg2025/mmdetection/configs/boxinst/boxinst_r50_fpn_ms-90k_coco.py
/media/ly/AddDisk/InstanceSeg2025/mmdetection/configs/scnet/scnet_r50_fpn_1x_coco.py
/media/ly/AddDisk/InstanceSeg2025/mmdetection/configs/mask2former/mask2former_r50_8xb2-lsj-50e_coco.py
/media/ly/AddDisk/InstanceSeg2025/mmdetection/configs/point_rend/point-rend_r50-caffe_fpn_ms-3x_coco.py
--work-dir
"/media/ly/AddDisk/PointSegmentation/tree_scan0001/save_dir"
 "/media/dl/add/Ly/InsSeg_apple/InsExper_data/03d00"
"""

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path', default="/home/zcy/yuanma/mmdetection_04d05/configs/rtmdet/rtmdet-ins_tiny_8xb32-300e_coco.py")
    parser.add_argument('--work-dir', default = "/media/zcy/zyy/InsSeg_apple/InsExper_data/04d05", help='the dir to save logs and models')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR.')
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpoint path, resume from it, while if not '
        'specify, try to auto resume from the latest checkpoint '
        'in the work directory.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    # Reduce the number of repeated compilations and improve
    # training speed.
    setup_cache_size_limit_of_dynamo()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # enable automatic-mixed-precision training
    if args.amp is True:
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.loss_scale = 'dynamic'

    # enable automatically scaling LR
    if args.auto_scale_lr:
        if 'auto_scale_lr' in cfg and \
                'enable' in cfg.auto_scale_lr and \
                'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True
        else:
            raise RuntimeError('Can not find "auto_scale_lr" or '
                               '"auto_scale_lr.enable" or '
                               '"auto_scale_lr.base_batch_size" in your'
                               ' configuration file.')

    # resume is determined in this priority: resume from > auto_resume
    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # start training
    runner.train()


if __name__ == '__main__':
    main()
