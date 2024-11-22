import os
import time
import cv2
import numpy as np


import shutil
def get_file_list_extension(path, file_ext):
    names = os.listdir(path)
    namelist = []
    for name in names:
        if name.endswith(file_ext):
            namelist.append(name)
    return namelist
def merge_arrays(array1, array2, weight1=0.9, weight2=0.1):
    """
    根据条件合并两个数组：
    - 如果array1中的像素为[0,0,0]，则取array2的像素；
    - 否则，取array1和array2像素的加权和。

    参数:
    - array1: 第一个形状为(640, 640, 3)的数组。
    - array2: 第二个形状为(640, 640, 3)的数组。
    - weight1: array1的权重，默认为0.7。
    - weight2: array2的权重，默认为0.3。

    返回:
    - 结果数组，形状为(640, 640, 3)。
    """
    # 判断array1中每个像素是否为全黑
    is_black = np.all(array1 == 0, axis=-1, keepdims=True)

    # 根据条件选择或计算结果
    result = np.where(is_black, array2, weight1 * np.array([255, 255, 0]) + weight2 * array2)
    return result

img_path = r"D:\Work\Git-Code\My_MaskRCNN\data\CustomDataset\test\image"
mask_path = r"D:\Work\Git-Code\My_MaskRCNN\data\CustomDataset\test\mask"
merge_path = r"D:\Work\Git-Code\My_MaskRCNN\data\CustomDataset\test\merge"
os.makedirs(merge_path, exist_ok=True)

png_list = get_file_list_extension(img_path, '.png')
for img in png_list:
    img_file = os.path.join(img_path,  img)
    mask_file = os.path.join(mask_path, img)
    img_cv = cv2.imread(img_file)
    # mask_cv = pred.astype(np.uint8)[None].reshape((640, 640, -1))
    # mask_cv = cv2.cvtColor(pred.astype(np.uint8), cv2.COLOR_RGB2BGR)0
    mask_cv = cv2.imread(mask_file)
    # mask = np.full(mask_cv.shape, False, dtype=bool)
    # mask[mask_cv == 255] = True
    result = merge_arrays(mask_cv, img_cv, weight1=0.1, weight2=0.9).astype(np.uint8)
    merge_file = os.path.join(merge_path, img)
    cv2.imwrite(merge_file, result)