from PIL import Image
import numpy as np
img_path = r'D:\Work\Git-Code\My_MaskRCNN\VOC2007\SegmentationObject\ISIC_000106.png'
img = Image.open(img_path).convert('RGB')
img2 = np.array(img)
print()