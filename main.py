import torch

import os
import datetime

from data import get_chest_xray_dataloader

from data import TRAIN_DATA_PATH
from data import VAL_DATA_PATH
from data import TEST_DATA_PATH

import transforms

from model import Vit

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 1，数据读取后的处理工作
    #     - 类型转换
    #     - 数据增强
    data_transform = {
        TRAIN_DATA_PATH: transforms.Compose([transforms.SSDCropping(),  # 图像切割
                                             transforms.Resize(),  # 统一大小
                                             transforms.ColorJitter(),  # 颜色抖动
                                             transforms.ToTensor(),  # 转张量
                                             transforms.RandomHorizontalFlip(),  # 水平翻转
                                             transforms.Normalization()]),  # 标准化

        VAL_DATA_PATH: transforms.Compose([transforms.Resize(),
                                           transforms.ToTensor(),
                                           transforms.Normalization()]),

        TEST_DATA_PATH: transforms.Compose([transforms.Resize(),
                                            transforms.ToTensor(),
                                            transforms.Normalization()])
    }

    # 构建训练数据集
    train_loader = get_chest_xray_dataloader(data_root_path=args.data_path,
                                             data_use_type=TRAIN_DATA_PATH,
                                             transforms=data_transform[TRAIN_DATA_PATH],
                                             batch_size=32,
                                             drop_last=False,
                                             shuffle=True,
                                             collate_fn=None)

    # 构建验证数据集
    val_loader = get_chest_xray_dataloader(data_root_path=args.data_path,
                                           data_use_type=VAL_DATA_PATH,
                                           transforms=data_transform[VAL_DATA_PATH],
                                           batch_size=5,
                                           drop_last=False,
                                           shuffle=True,
                                           collate_fn=None)

    # 构建测试数据集
    test_loader = get_chest_xray_dataloader(data_root_path=args.data_path,
                                            data_use_type=TEST_DATA_PATH,
                                            transforms=data_transform[TEST_DATA_PATH],
                                            batch_size=32,
                                            drop_last=False,
                                            shuffle=True,
                                            collate_fn=None)

    # 构建模型
    model = Vit()
    model.to(device=device)
    model.train()
    # 定义损失函数
    loss_func = torch.nn.CrossEntropyLoss()
    # 定义优化器
    optim = torch.optim.Adam(params=model.parameters(), lr=0.00001)
    if len(args.resume) > 0:
        checkpoint = torch.load(f=args.resume, map_location="cpu")
        model.load_state_dict(state_dict=checkpoint["model"])
        optim.load_state_dict(state_dict=checkpoint["optimizer"])
        args.start_epoch = checkpoint["last_epoch"] + 1
        print("the training process from epoch{}...".format(args.start_epoch))

    weighted_losses = torch.zeros(1).to(device=device)
    for epoch in range(args.start_epoch, args.start_epoch + args.epochs):
        train_acc_list = []
        val_acc_list = []
        test_acc_list = []
        for i, (x, y) in enumerate(train_loader):
            if len(x) <= 0:
                continue

            x = torch.stack(x, dim=0)
            y = torch.stack(y, dim=0)
            # x = [B, 256 + 1, 768]
            x = x.to(device=device)
            # y = [B, 1]
            y = y.to(device=device)
            # pred = [B, 1]
            pred = model(x)

            loss = loss_func(pred, y.reshape(-1))
            weighted_losses = (i * weighted_losses + loss ) / (i + 1)

            optim.zero_grad()
            loss.backward()
            optim.step()

            if i % 20 == 0:
                lr = optim.param_groups[0]['lr']
                print("{} Epoch{}/{}: lr={}, cur_loss={:.4}, weighted_loss={:.4}".format(
                      datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
                      epoch,
                      i,
                      lr,
                      loss.item(),
                      weighted_losses.item()))

            # [B, 1] -> [B]
            pred = pred.argmax(dim=-1).reshape(-1)
            # [B, 1] -> [B]
            y = y.reshape(-1)

            train_acc_list.extend((pred == y).to(dtype=torch.float32).tolist())

            # save weights
        save_files = {
            'model': model.state_dict(),
            'optimizer': optim.state_dict(),
            'last_epoch': epoch}
        torch.save(save_files, "./save_weights/ChestXRay-{}.pth".format(epoch))

        # 验证
        for i, (x, y) in enumerate(val_loader):
            if len(x) <= 0:
                continue

            x = torch.stack(x, dim=0)
            y = torch.stack(y, dim=0)

            x = x.to(device=device)
            y = y.to(device=device)

            pred_val = model(x).argmax(dim=-1).reshape(-1)
            y = y.reshape(-1)
            val_acc_list.extend((pred_val == y).to(dtype=torch.float32).tolist())

        print("Epoch {}: train_acc={:4}/{}, val_acc={:4}/{}, lr={:6}".format(
            epoch,
            sum(train_acc_list) / len(train_acc_list),
            len(train_acc_list),
            sum(val_acc_list) / len(val_acc_list),
            len(val_acc_list),
            optim.param_groups[0]["lr"]))

    # 测试
    for i, (x, y) in enumerate(test_loader):
        if len(x) <= 0:
            continue

        x = torch.stack(x, dim=0)
        y = torch.stack(y, dim=0)

        x = x.to(device=device)
        y = y.to(device=device)

        pred_test = model(x).argmax(dim=-1).reshape(-1)
        y = y.reshape(-1)
        test_acc_list.extend((pred_test == y).to(dtype=torch.float32).tolist())

    print("Epoch {}-{}: test_acc={:4}/{}".format(
          args.start_epoch,
          args.start_epoch + args.epochs,
          sum(test_acc_list) / len(test_acc_list),
          len(test_acc_list)))

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description=__doc__)

    # 检测的目标类别个数，不包括背景(替换：自己的检测类别)
    parser.add_argument('--num_classes', default=4, type=int, help='num_classes')
    # 训练数据集的根目录
    parser.add_argument('--data_path', default='D:\\workspace\\ai_study\\dataset\\ChestX-Ray\\ChestXRay', help='dataset')
    # 文件保存地址
    parser.add_argument('--output-dir', default='./save_weights', help='path where to save')
    # 若需要接着上次训练，则指定上次训练保存权重文件地址
    parser.add_argument('--resume', default='', type=str, help='resume from checkpoint')
    # 指定接着从哪个epoch数开始训练
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    # 训练的总epoch数
    parser.add_argument('--epochs', default=15, type=int, metavar='N',
                        help='number of total epochs to run')
    # 训练的batch size
    parser.add_argument('--batch_size', default=32, type=int, metavar='N',
                        help='batch size when training.')

    args = parser.parse_args()

    # 检查保存权重文件夹是否存在，不存在则创建
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    main(args)