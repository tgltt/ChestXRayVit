import torch
from torch import nn

from util import MultiHead
from util import PatchEmbedding
from util import ClassificationHead
from util import FullyConnectedOutput

# 编码器层
class EncoderLayer(nn.Module):
    def __init__(self, max_len, emb_size):
        super().__init__()
        # 多头注意力
        self.mh = MultiHead(max_len, emb_size)
        # 全连接输出
        self.fc = FullyConnectedOutput()

    def forward(self, x):
        # 计算自注意力,维度不变
        # [b, 256 + 1, 768] -> [b, 256 + 1, 768]
        score = self.mh(x, x, x, mask=None)

        # 全连接输出,维度不变
        # [b, 256 + 1, 768] -> [b, 256 + 1, 768]
        out = self.fc(score)

        return out


class Encoder(torch.nn.Module):
    def __init__(self, max_len, emb_size):
        super().__init__()
        self.layer_1 = EncoderLayer(max_len, emb_size)
        self.layer_2 = EncoderLayer(max_len, emb_size)
        self.layer_3 = EncoderLayer(max_len, emb_size)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        return x

# 主模型
class TransformerEncoder(torch.nn.Module):
    def __init__(self, max_len, emb_size):
        super().__init__()
        # 位置编码和词嵌入层
        self.encoder = Encoder(max_len, emb_size)

    def forward(self, x):
        # 编码层计算
        # [b, 256 + 1, 768] -> [b, 256 + 1, 768]
        return self.encoder(x)


class Vit(nn.Sequential):
    def __init__(self):
        super(Vit, self).__init__(
            PatchEmbedding(in_channels=1, patch_size=16, emb_size=768, img_size=256),
            TransformerEncoder(max_len=(256 // 16) ** 2 + 1, emb_size=768),
            ClassificationHead(emb_size=768, n_classes=4)
        )