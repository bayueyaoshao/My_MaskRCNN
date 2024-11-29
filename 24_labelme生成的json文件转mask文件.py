import os
import json

import cv2
import numpy as np
def get_file_list(path, file_ext):
    file_list = os.listdir(path)
    ret = []
    for old_name in file_list:
        if old_name.endswith(file_ext):
            ret.append(f'{path}/{old_name}')
    return ret


if __name__ == "__main__":
    path = r'D:\Work\Git-Code\My_MaskRCNN\VOC2007\mask'
    json_list = get_file_list(path, '.json')
    for json_file in json_list:
        print(json_file)
        with open(json_file, 'r') as f:
            # data = f.read()
            # convert fp to json
            data = json.load(f)
            shapes = data["shapes"]
            if len(shapes) == 0:
                print(json_file)
            w = data["imageWidth"]
            h = data["imageHeight"]
            mask = np.zeros((h, w), dtype=np.uint8)
            for i in range(len(shapes)):
                points = shapes[i]["points"]
                points = np.array(points, dtype=np.int32)
                cv2.fillPoly(mask, [points], (255, 255, 255))
            mask_file = json_file.replace(".json", '.png')
            cv2.imwrite(mask_file, mask)






