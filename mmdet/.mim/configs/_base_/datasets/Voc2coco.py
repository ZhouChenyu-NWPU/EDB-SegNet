# import os
# import json
# import numpy as np
# from PIL import Image
# from pycocotools import mask
# import binascii
#
# def voc_to_coco(voc_root, output_json):
#     coco_data = {
#         "images": [],
#         "annotations": [],
#         "categories": []
#     }
#
#     # Assuming you have a classes.txt file containing class names
#     with open(os.path.join(voc_root, "classes.txt"), "r") as f:
#         classes = f.read().strip().split("\n")
#     for i, class_name in enumerate(classes, 1):
#         coco_data["categories"].append({"id": i, "name": class_name})
#
#     image_id = 0
#     annotation_id = 0
#
#     image_list_file = os.path.join(voc_root, "test.txt")
#     with open(image_list_file, "r") as f:
#         image_list = f.read().strip().split("\n")
#
#     for image_name in image_list:
#         image_id += 1
#         image_path = os.path.join(voc_root, image_name.split()[0])
#         segmentation_path = os.path.join(voc_root, image_name.split()[1])
#
#         image = Image.open(image_path)
#         width, height = image.size
#
#         coco_data["images"].append({
#             "id": image_id,
#             "file_name": f"{image_name}.jpg",
#             "width": width,
#             "height": height
#         })
#
#         segmentation = Image.open(segmentation_path)
#         segmentation_array = np.array(segmentation)
#
#         for category_id, class_name in enumerate(classes, 1):
#             mask_array = np.uint8(segmentation_array == category_id)
#             mask_encoded = mask.encode(np.array(mask_array, order="F"))
#             area = mask.area(mask_encoded)
#             bbox = mask.toBbox(mask_encoded)
#             annotation_id += 1
#             # 直接将字典对象转换为 JSON 字符串，然后编码为字节对象
#             segmentation_encoded = json.dumps(mask_encoded).encode('utf-8')
#             encoded = binascii.b2a_base64(segmentation_encoded, newline=False).decode('utf-8')
#             coco_data["annotations"].append({
#                 "id": annotation_id,
#                 "image_id": image_id,
#                 "category_id": category_id,
#                 "iscrowd": 0,
#                 "area": float(area),
#                 "bbox": bbox.tolist(),
#                 "segmentation": encoded  # 使用编码后的字符串
#             })
#
#     with open(output_json, "w") as f:
#         json.dump(coco_data, f)
#
# voc_root = "/media/ly/My Passport/科研重要文件备份/数据集/果树数据集/leaves_branch_data"
# output_json = "/media/ly/My Passport/科研重要文件备份/数据集/果树数据集/leaves_branch_data/coco_dataset.json"
# voc_to_coco(voc_root, output_json)
#

import os
import json
import numpy as np
from PIL import Image
from skimage.measure import label


def image_to_coco_json(image_path, output_json_path):
    # 打开图像文件
    image = Image.open(image_path)

    # 将图像转换为灰度图像并转换为 NumPy 数组
    gray_image = image.convert('L')
    image_array = np.array(gray_image)

    # 进行连通组件标记，将图像分割为不同的区域
    labeled_image = label(image_array)
    num_regions = labeled_image.max()

    # 创建 COCO 数据集字典
    coco_data = {
        "info": {},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }

    # 添加图像信息
    width, height = image.size
    coco_data["images"].append({
        "id": 1,  # 图像ID
        "width": width,  # 图像宽度
        "height": height,  # 图像高度
        "file_name": os.path.basename(image_path)  # 图像文件名
    })

    # 遍历分割后的区域，将每个区域作为一个标注
    annotations = []
    for region_id in range(1, num_regions + 1):
        region_mask = (labeled_image == region_id)
        category_id = int(image_array[region_mask][0])  # 将整数强制转换为Python的基本整数类型

        # 获取区域的边界框信息
        ys, xs = np.where(region_mask)
        min_x, min_y, max_x, max_y = min(xs), min(ys), max(xs), max(ys)
        bbox = [min_x, min_y, max_x - min_x, max_y - min_y]

        # 创建标注
        annotations.append({
            "id": len(annotations) + 1,  # 标注ID
            "image_id": 1,  # 对应图像ID
            "category_id": category_id,  # 类别ID
            "bbox": bbox,  # 边界框信息
            "area": (max_x - min_x) * (max_y - min_y),  # 区域面积
            "iscrowd": 0  # 是否是团队标注
        })

    # 添加标注信息到 COCO 数据集字典
    coco_data["annotations"] = annotations

    # 将 COCO 数据集字典写入 JSON 文件
    with open(output_json_path, 'w') as json_file:
        json.dump(coco_data, json_file, indent=4)

# 例子使用
image_path = "/media/ly/AddDisk/PointSegmentation/tree_scan0001/output/images实例分割输出/00000000.png"  # 图像文件路径
output_json_path = "/media/ly/AddDisk/PointSegmentation/tree_scan0001/output/images实例分割输出/output.json"  # 输出 JSON 文件路径
image_to_coco_json(image_path, output_json_path)
