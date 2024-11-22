import cv2
import numpy as np
import os
def get_file_list(path, file_ext):
    old_names = os.listdir(path)
    ret = []
    for old_name in old_names:
        if old_name.endswith(file_ext):
            ret.append(old_name)
            # print(old_name)
    return ret

def mask_to_yolo(mask_path, output_path):
    # 读取掩膜图像
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # 找到掩膜中的轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 创建一个空列表来存储YOLO格式的标签
    labels = []

    for contour in contours:
        # 计算每个轮廓的边界框
        x, y, w, h = cv2.boundingRect(contour)

        xmin = x
        xmax = x + w
        ymin = y
        ymax = y + h
        # 将数据添加到标签列表
        labels.append([xmin, ymin, xmax, ymax])

    # 写入YOLO标签文件
    with open(output_path, 'w') as f:
        for label in labels:
            line = " ".join([str(l) for l in label])
            f.write(line + '\n')

if __name__ == '__main__':
    # 使用函数
    # mask_path = r'E:\label\1_20240716_225_QL_618F_ng_20.png'
    # output_path = r'E:\label\1_20240716_225_QL_618F_ng_20.txt'
    # mask_to_yolo(mask_path, output_path)

    path = r'D:\Work\Git-Code\My_MaskRCNN\data\CustomDataset\test\mask'
    file_ext = '.png'
    png_list = get_file_list(path, file_ext)
    for file in png_list:
        mask_path = os.path.join(path, file)
        output_path = mask_path.replace(".png", '.txt')
        mask_to_yolo(mask_path, output_path)
