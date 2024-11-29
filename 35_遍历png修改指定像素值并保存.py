from PIL import Image
import os
import numpy as np
def process_images(folder_path, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.png'):
            # 打开图像
            image_path = os.path.join(folder_path, filename)
            img = Image.open(image_path)

            # 确保图像为单通道
            if img.mode != 'L':
                raise ValueError(f"Image {filename} is not a single channel image.")

            # 将图像转换为可编辑的数组
            img_array = np.array(img)

            # 将大于0的像素值设置为255
            img_array[img_array > 1] = 255

            # 将数组转换回PIL图像
            processed_img = Image.fromarray(img_array.astype('uint8'), mode='L')

            # 保存处理后的图像
            output_path = os.path.join(output_folder, filename)
            processed_img.save(output_path)

# 指定文件夹路径
input_folder = r'D:\Work\Git-Code\My_MaskRCNN\VOC2007\SegmentationObject'
output_folder = r'D:\Work\Git-Code\My_MaskRCNN\VOC2007\SegmentationObject_mask'

# 处理文件夹中的所有单通道PNG图像
process_images(input_folder, output_folder)
