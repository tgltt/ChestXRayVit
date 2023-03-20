import random

import torchvision.transforms as t
from torchvision.transforms import functional as F

from data import TARGET_FIELD_HEIGHT_WIDTH

class Compose(object):
    """组合多个transform函数"""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target=None):
        for trans in self.transforms:
            image, target = trans(image, target)
        return image, target


class ToTensor(object):
    """将PIL图像转为Tensor"""
    def __call__(self, image, target):
        image = F.to_tensor(image).contiguous()
        channel_count = len(image)
        if channel_count > 1:
            image = image.mean(dim=0).unsqueeze(dim=0)
        return image, target


class RandomHorizontalFlip(object):
    """随机水平翻转图像以及bboxes,该方法应放在ToTensor后"""
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = image.flip(-1)  # 水平翻转图片
        return image, target


class SSDCropping(object):
    """
    对图像进行裁剪,该方法应放在ToTensor前
    """
    def __init__(self):
        super(SSDCropping, self).__init__()
        self.sample_options = (None, (0.3, 1.0))

    def __call__(self, image, target):
        # 死循环，确保一定会返回结果
        while True:
            mode = random.choice(self.sample_options)
            if mode is None:  # 不做随机裁剪处理
                return image, target

            # Implementation use 5 iteration to find possible candidate
            for _ in range(5):
                # 0.3*0.3 approx. 0.1
                w = random.uniform(0.3, 1.0)
                h = random.uniform(0.3, 1.0)
                if w/h < 0.5 or w/h > 2:  # 保证宽高比例在0.5-2之间
                    continue

                # left 0 ~ wtot - w, top 0 ~ htot - h
                left = random.uniform(0, 1.0 - w)
                top = random.uniform(0, 1.0 - h)

                right = left + w
                bottom = top + h

                htot = target[TARGET_FIELD_HEIGHT_WIDTH][0]
                wtot = target[TARGET_FIELD_HEIGHT_WIDTH][1]

                # 裁剪 patch
                left_idx = int(left * wtot)
                top_idx = int(top * htot)
                right_idx = int(right * wtot)
                bottom_idx = int(bottom * htot)
                image = image.crop((left_idx, top_idx, right_idx, bottom_idx))
                # image.save(f"./output/crop/crop_image{self.count}.jpg")
                # self.count += 1
                return image, target


class Resize(object):
    """对图像进行resize处理,该方法应放在ToTensor前"""
    def __init__(self, size=(256, 256)):
        self.resize = t.Resize(size)
        # self.count = 0

    def __call__(self, image, target):
        image = self.resize(image)
        # image.save(f"./output/resize/resize{self.count}.jpg")
        # self.count += 1
        return image, target


class ColorJitter(object):
    """对图像颜色信息进行随机调整,该方法应放在ToTensor前"""
    def __init__(self, brightness=0.125, contrast=0.5, saturation=0.5, hue=0.05):
        self.trans = t.ColorJitter(brightness, contrast, saturation, hue)
        self.count = 0

    def __call__(self, image, target):
        image = self.trans(image)
        # image.save(f"./output/color_jitter/color_jitter{self.count}.jpg")
        # self.count += 1
        return image, target


class Normalization(object):
    """对图像标准化处理,该方法应放在ToTensor后"""
    def __init__(self, mean=None, std=None):
        if mean is None:
            mean = [0.48698336]
        if std is None:
            std = [0.23673548]
        self.normalize = t.Normalize(mean=mean, std=std)

    def __call__(self, image, target):
        image = self.normalize(image)
        return image, target