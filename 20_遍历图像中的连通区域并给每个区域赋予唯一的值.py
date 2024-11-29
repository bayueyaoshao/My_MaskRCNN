import cv2
import numpy as np
import os
def get_file_list_extension(path, file_ext):
    names = os.listdir(path)
    namelist = []
    for name in names:
        if name.endswith(file_ext):
            namelist.append(name)
    return namelist


mask_path = r'.\VOC2007\mask'
mask_split_path = r'.\VOC2007\SegmentationObject'
os.makedirs(mask_split_path, exist_ok=True)
png_files = get_file_list_extension(mask_path, '.png')
for png_file in png_files:
    src_img = os.path.join(mask_path, png_file)
    dst_img = os.path.join(mask_split_path, png_file)
    # 读取图像
    image = cv2.imread(src_img, cv2.IMREAD_GRAYSCALE)


    # 连通成分标记
    num_labels, labels = cv2.connectedComponents(image, connectivity=8)

    # 生成一个新的图像，每个连通区域有不同的像素值
    output_image = np.zeros_like(image)

    # 为每个连通区域分配不同的标签值
    for label in range(1, num_labels):
        output_image[labels == label] = label

    # # 显示结果
    # cv2.imshow('Original Image', image)
    # cv2.imshow('Labeled Image', output_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # 保存结果
    cv2.imwrite(dst_img, output_image)
