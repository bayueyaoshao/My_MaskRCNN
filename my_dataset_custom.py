import os
import json


import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import glob

# import transforms
# data_transform1 = {
#         "train": transforms.Compose([transforms.ToTensor(),
#                                      transforms.RandomHorizontalFlip(0.5)]),
#         "val": transforms.Compose([transforms.ToTensor()])
#     }


def get_file_list_extension(path, file_ext):
    names = os.listdir(path)
    namelist = []
    for name in names:
        if name.endswith(file_ext):
            namelist.append(name)
    return namelist
class CustomDataset(Dataset):
    def __init__(self, data_root, name: str = "train", transforms=None):
        super().__init__()
        self.transforms = transforms
        image_dir = os.path.join(data_root, name, 'image')
        boxes_dir = os.path.join(data_root, name, 'boxes')
        mask_dir = os.path.join(data_root, name, 'mask')
        self.images_path = []     # 存储图片路径
        self.masks_path = []      # 存储SegmentationObject图片路径
        self.bboxes_path = []  # 存储解析的目标boxes等信息
        self.image_id_list = []
        self.boxes_list = []
        self.labels_list = []
        self.iscrowd_list = []
        self.area_list = []
        png_files = get_file_list_extension(image_dir, '.png')
        # 对文件进行排序（可选，但通常是个好习惯）
        png_files.sort()
        for idx, png_file in enumerate(png_files):
            self.images_path.append(os.path.join(image_dir, png_file))
            self.masks_path.append(os.path.join(mask_dir, png_file))
            bboxes_path_tmp = os.path.join(boxes_dir, png_file).replace(".png", '.txt')
            self.bboxes_path.append(bboxes_path_tmp)
            self.image_id_list.append(idx)
            with open(bboxes_path_tmp) as f:
                ret = []
                label = []
                crowd = []
                area = []
                for x in f.read().strip().splitlines():
                    if len(x):
                        lb = x.split()   
                        lb = np.array(lb, dtype=np.float32)
                        area.append((lb[3]-lb[1])*(lb[2]-lb[0]))
                        ret.append(lb)
                        label.append(1)
                        crowd.append(0)
                self.boxes_list.append(ret)
                self.labels_list.append(label)
                self.iscrowd_list.append(crowd)
                self.area_list.append(area)

    def parse_mask(self, idx: int):
        mask_path = self.masks_path[idx]
        mask = Image.open(mask_path)
        mask = np.array(mask)
        mask[mask == 255] = 0  # 255为背景或者忽略掉的地方，这里为了方便直接设置为背景(0)

        # mask = self.masks[idx]
        c = mask.max()  # 有几个目标最大索引就等于几
        masks = []
        # 对每个目标的mask单独使用一个channel存放
        for i in range(1, c + 1):
            masks.append(mask == i)
        masks = np.stack(masks, axis=0)
        return torch.as_tensor(masks, dtype=torch.uint8)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images_path[idx]).convert('RGB')
        target = {"boxes": torch.tensor(self.boxes_list[idx]),
            "labels": torch.tensor(self.labels_list[idx]),
            "iscrowd": torch.tensor(self.iscrowd_list[idx]),
            "image_id": torch.tensor(self.image_id_list[idx]),
            "area": torch.tensor(self.area_list[idx])}
        masks = self.parse_mask(idx)
        target["masks"] = masks

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        # image->[3, 640, 640] (0-1)
        # boxes->[n, 4](0-255)
        # labels->[n]
        # iscrowd->[n]
        # image_id->[1]
        # area->[n]
        # mask->[n, 640,640] torch.uint8

        return img, target

    def __len__(self):
        return len(self.images_path)


    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))


if __name__ == '__main__':
    data_root = r'D:\Work\Git-Code\My_MaskRCNN\data\CustomDataset'
    train_dataset = CustomDataset(data_root, name="train", transforms=None)
    print(train_dataset[2])

