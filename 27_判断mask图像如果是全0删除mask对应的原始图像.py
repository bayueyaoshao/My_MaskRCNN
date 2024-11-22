import os
from PIL import Image
import numpy as np
image_path = r"D:\Work\Git-Code\My_MaskRCNN\data\CustomDataset\train\image"
mask_path = r"D:\Work\Git-Code\My_MaskRCNN\data\CustomDataset\train\mask"

if __name__ == '__main__':
    mask_png_list = os.listdir(mask_path)
    print(mask_png_list)
    for mask_img in mask_png_list:
        mask_full_path = os.path.join(mask_path, mask_img)
        img = Image.open(mask_full_path).convert("L")
        np_img = np.array(img)

        if np_img.max() == 0:
            print(np_img.max())
            print(mask_img)
            os.remove(mask_full_path)
            src_full_path = os.path.join(image_path, mask_img)
            os.remove(src_full_path)





