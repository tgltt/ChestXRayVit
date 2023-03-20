import os
import torch
import json

from PIL import Image

TRAIN_DATA_PATH = "train"
VAL_DATA_PATH = "val"
TEST_DATA_PATH = "test"

CLASS_NORMAL_KEY = "NORMAL"
CLASS_PNEUMONIA_KEY = "PNEUMONIA"
CLASS_COVID19_KEY = "COVID19"
CLASS_TUBERCULOSIS_KEY = "TURBERCULOSIS"

CLASS_COUNT = 4

TARGET_FIELD_LABELS = "labels"
TARGET_FIELD_IMAGE_ID = "image_id"
TARGET_FIELD_HEIGHT_WIDTH = "height_width"
TARGET_FIELD_IMAGE_DATA = "image_data"

# 定义数据集
class ChestXRayDataset(torch.utils.data.Dataset):
    _SUPPORT_FILE_TYPES = (".jpg", ".jpeg", ".mpo", ".png")

    def __init__(self, data_path, transforms):
        super(ChestXRayDataset, self).__init__()
        self.data_path = data_path if len(data_path) > 0 else "."
        self.sub_dirs = [os.path.join(self.data_path, sub_dir) for sub_dir in os.listdir(path=data_path)]
        # read class_indict
        json_file = "./chest_x_ray_classes.json"
        assert os.path.exists(json_file), "{} file not exist.".format(json_file)

        with open(json_file, 'r') as f:
            self.class_dict = json.load(f)

        self.data_filenames = []
        self.label_type = []
        for sub_dir in self.sub_dirs:
            label_type = os.path.basename(sub_dir)
            data_file_names = [os.path.join(data_path, sub_dir, data_file_name) for data_file_name in os.listdir(path=os.path.join(data_path, sub_dir))
                                if self._is_support_file_type(filename=data_file_name)]
            self.data_filenames.extend(data_file_names)
            self.label_type.extend([self.class_dict[label_type] for _ in range(len(data_file_names))])

        self.transforms = transforms

    def __len__(self):
        return len(self.data_filenames)

    def __getitem__(self, i):
        assert i < len(self.data_filenames), f"Index {i} out of bounds."
        return self._get_data(data_file_name=self.data_filenames[i], label_type=self.label_type[i])

    def _get_data(self, data_file_name, label_type):
        try:
            # data_file_name = 'D:\\workspace\\ai_study\\dataset\\ChestX-Ray\\ChestXRay\\train\\COVID19\\COVID19(436).jpg'
            raw_image_data = Image.open(fp=data_file_name, mode="r")
            if not self._is_file_ext_supported("." + raw_image_data.format):
                raise ValueError(f"Image '{data_file_name}'s' format not JPEG")

            target = {}
            target[TARGET_FIELD_IMAGE_DATA] = raw_image_data
            target[TARGET_FIELD_IMAGE_ID] = data_file_name
            target[TARGET_FIELD_LABELS] = torch.as_tensor([label_type], dtype=torch.int64)
            target[TARGET_FIELD_HEIGHT_WIDTH] = [raw_image_data.height, raw_image_data.width]

            if self.transforms is not None:
                image, target = self.transforms(raw_image_data, target)
            return image, target[TARGET_FIELD_LABELS]
        except:
            print(f"Open {data_file_name} failed.")
            return (None, None)

    @staticmethod
    def collate_fn(batch):
        images, targets = tuple(zip(*batch))

        filter_images = []
        filter_targets = []
        for index in range(len(images)):
            if images[index] is not None:
                filter_images.append(images[index])
                filter_targets.append(targets[index])

        return tuple(filter_images), tuple(filter_targets)

    def _is_file_ext_supported(self, file_ext):
        return file_ext.lower() in self._SUPPORT_FILE_TYPES

    def _is_support_file_type(self, filename):
        return os.path.splitext(p=(filename.lower()))[-1] in self._SUPPORT_FILE_TYPES

def get_chest_xray_dataloader(data_root_path,
                              data_use_type,
                              transforms,
                              batch_size,
                              drop_last,
                              shuffle,
                              collate_fn):
    dataset = ChestXRayDataset(os.path.join(data_root_path, data_use_type), transforms)
    # 数据加载器
    return torch.utils.data.DataLoader(dataset=dataset,
                                       batch_size=batch_size,
                                       drop_last=drop_last,
                                       shuffle=shuffle,
                                       collate_fn=collate_fn if collate_fn is not None else dataset.collate_fn)