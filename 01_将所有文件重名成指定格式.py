import os
import glob


def rename_png_images(folder_path):
    # 获取文件夹中所有的 PNG 图像
    png_files = glob.glob(os.path.join(folder_path, "*.png"))

    # 对文件进行排序（可选，但通常是个好习惯）
    png_files.sort()

    # 初始化计数器
    counter = 1

    # 遍历每个 PNG 文件
    for file_path in png_files:
        # 生成新的文件名
        # new_filename = f"ISIC_{counter:06d}.png"
        new_filename = f"{counter:04d}.png"
        new_file_path = os.path.join(folder_path, new_filename)

        # 重命名文件
        os.rename(file_path, new_file_path)

        # 增加计数器
        counter += 1


# 使用示例
folder_path = r"D:\Work\Git-Code\My_MaskRCNN\data\CustomDataset\test\mask"  # 替换为你的文件夹路径
rename_png_images(folder_path)
