from PIL import Image
import os
import json

import torch

import transforms

from model import Vit

# 预测函数
def predict(img_path_list, label_list):
    if len(img_path_list) <= 0:
        print("img_path_list is null or empty")
        return

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    save_weights = "./save_weights/ChestXRay-80.pth"

    # read class_indict
    json_file = "./chest_x_ray_classes.json"
    assert os.path.exists(json_file), "{} file not exist.".format(json_file)
    with open(json_file, 'r') as f:
        class_dict = json.load(f)
    reverse_class_dict = {v : k for k, v in class_dict.items()}

    model = Vit()
    model.load_state_dict(state_dict=torch.load(f=save_weights, map_location="cpu")["model"])
    model.to(device=device)

    model.eval()

    predict_result = []
    for index in range(len(img_path_list)):
        img_path = img_path_list[index]
        original_img = Image.open(img_path)
        data_transform = transforms.Compose([transforms.Resize(),
                                             transforms.ToTensor(),
                                             transforms.Normalization()])
        img, _ = data_transform(original_img)
        # 改为批量预测
        x = torch.unsqueeze(img, dim=0)
        pred = model(x).argmax(dim=-1).item()

        print(f"{img_path}: predict result: " + reverse_class_dict[pred] + ", real result: " + label_list[index])


if __name__ == '__main__':
    img_path_list = (
    "D:\\workspace\\ai_study\\dataset\\ChestX-Ray\\ChestXRay\\test\\COVID19\\COVID19(497).jpg",
    "D:\\workspace\\ai_study\\dataset\\ChestX-Ray\\ChestXRay\\test\\COVID19\\COVID19(520).jpg",
    "D:\\workspace\\ai_study\\dataset\\ChestX-Ray\\ChestXRay\\test\\NORMAL\\IM-0025-0001.jpeg",
    "D:\\workspace\\ai_study\\dataset\\ChestX-Ray\\ChestXRay\\test\\NORMAL\\NORMAL2-IM-0345-0001.jpeg",
    "D:\\workspace\\ai_study\\dataset\\ChestX-Ray\\ChestXRay\\test\\PNEUMONIA\\person14_virus_44.jpeg",
    "D:\\workspace\\ai_study\\dataset\\ChestX-Ray\\ChestXRay\\test\\PNEUMONIA\\person34_virus_76.jpeg",
    "D:\\workspace\\ai_study\\dataset\\ChestX-Ray\\ChestXRay\\test\\TURBERCULOSIS\\Tuberculosis-684.png",
    "D:\\workspace\\ai_study\\dataset\\ChestX-Ray\\ChestXRay\\test\\TURBERCULOSIS\\Tuberculosis-699.png",
    ".\\test_res\\Tuberculosis-web-01.jpg",
    ".\\test_res\\Tuberculosis-web-02.jpg")

    label_list = ("COVID19", "COVID19", "NORMAL", "NORMAL", "PNEUMONIA", "PNEUMONIA", "TURBERCULOSIS", "TURBERCULOSIS",
                  "TURBERCULOSIS", "TURBERCULOSIS")

    predict(img_path_list, label_list)