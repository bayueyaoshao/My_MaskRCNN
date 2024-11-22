import os
import cv2
def resize_image(image_path, output_path, width, height, keep_shape=True):
    os.makedirs(output_path, exist_ok=True)
    fullpaths = [os.path.join(image_path, name) for name in os.listdir(image_path) if name.endswith('.png')]
    print(fullpaths)
    print(len(fullpaths))
    for src_img in fullpaths:
        im = cv2.imread(src_img, cv2.IMREAD_UNCHANGED)
        # im = cv2.imread(src_img)
        resize_image = cv2.resize(im, (width, height), interpolation=cv2.INTER_AREA)
        new_full_path = src_img.replace(image_path, output_path)
        cv2.imwrite(new_full_path, resize_image)



if __name__ == "__main__":
    # resize_image(r'.\project_data\ap2358-wheeldectct\2-29-v1\dataset\skf-7\image_label_data_ori', r'.\project_data\ap2358-wheeldectct\2-29-v1\dataset\skf-7\image_label_data_ori_resize', 480, 360)
    resize_image(r'D:\Work\Git-Code\My_MaskRCNN\data\CustomDataset\train\mask',
                 r'D:\Work\Git-Code\My_MaskRCNN\data\CustomDataset\train\mask', 900, 900)

    