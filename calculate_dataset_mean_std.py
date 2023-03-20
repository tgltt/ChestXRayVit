import numpy as np
import cv2
import random

import os
from tqdm.notebook import tqdm_notebook

img_root = "D:\\workspace\\ai_study\\dataset\\ChestX-Ray\\ChestXRay\\train"
sub_dirs = [os.path.join(img_root, sub_dir) for sub_dir in os.listdir(path=img_root)]

CNum = 6000  # 挑选多少图片进行计算

img_h, img_w = 256, 256
imgs = np.zeros([img_w, img_h, 1, 1])
means, stdevs = [], []

data_files = []
for sub_dir in sub_dirs:
    data_files.extend([os.path.join(sub_dir, data_type) for data_type in os.listdir(path=sub_dir)])

random.shuffle(data_files)  # shuffle, 随机挑选图片
for index in tqdm_notebook(range(CNum)):
    if index >= len(data_files):
        break
    data_file = data_files[index]
    img = cv2.imread(data_file)
    img = cv2.resize(img, (img_h, img_w))
    img = img.transpose(2, 0, 1)
    img = img.mean(axis=0)
    img = img[:, :, np.newaxis, np.newaxis]

    imgs = np.concatenate((imgs, img), axis=3)

imgs = imgs.astype(np.float32) / 255.

for i in tqdm_notebook(range(1)):
    pixels = imgs[:, :, i, :].ravel()  # 拉成一行
    means.append(np.mean(pixels))
    stdevs.append(np.std(pixels))

# cv2 读取的图像格式为BGR，PIL/Skimage读取到的都是RGB不用转
means.reverse()  # BGR --> RGB
stdevs.reverse()

print("normMean = {}".format(means))
print("normStd = {}".format(stdevs))
print('transforms.Normalize(normMean = {}, normStd = {})'.format(means, stdevs))

# normMean = [0.48698336]
# normStd = [0.23673548]
# transforms.Normalize(normMean = [0.48698336], normStd = [0.23673548])