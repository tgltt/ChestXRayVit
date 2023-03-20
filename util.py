from collections import defaultdict, deque

import torch
from torch import nn
from torch import Tensor

from data import CLASS_COUNT

DEFAULT_HIDDEN_SIZE = 768
IMAGE_DEFAULT_INNER_SIZE = 256
PATCH_DEFAULT_SIZE = 16
IMAGE_DEFAULT_CHANNELS = 3

# 注意力计算函数
# [BatchSize, Head_i, SeqLen, Emb_Size]
def attention(Q, K, V, mask, max_len, emb_size):
    # b句话,每句话50个词,每个词编码成32维向量,4个头,每个头分到8维向量
    # Q,K,V = [b, 4, 50, 8]

    # [b, 4, 256 + 1, 192] * [b, 4, 192, 256 + 1] -> [b, 4, 256 + 1, 256 + 1]
    # Q,K矩阵相乘,求每个词相对其他所有词的注意力
    score = torch.matmul(Q, K.permute(0, 1, 3, 2))

    # 除以每个头维数的平方根,做数值缩放
    score /= 8 ** 0.5

    # mask 遮盖,mask是true的地方都被替换成-inf,这样在计算softmax的时候,-inf会被压缩到0
    # mask = [b, 1, 256 + 1, 256 + 1]
    if mask is not None:
        score = score.masked_fill_(mask, -float('inf'))

    score = torch.softmax(score, dim=-1)

    # 以注意力分数乘以V,得到最终的注意力结果
    # [b, 4, 256 + 1, 256 + 1] * [b, 4, 256 + 1, 192] -> [b, 4, 256 + 1, 192]
    score = torch.matmul(score, V)

    # 每个头计算的结果合一
    # [b, 4, 256 + 1, 192] -> [b, 256 + 1, 768]
    score = score.permute(0, 2, 1, 3).reshape(-1, max_len, emb_size)

    return score


# 多头注意力计算层
class MultiHead(nn.Module):
    def __init__(self, max_len, emb_size):
        super().__init__()
        self.max_len = max_len
        self.emb_size = emb_size
        # Q 矩阵
        self.fc_Q = nn.Linear(emb_size, emb_size)
        # K 矩阵
        self.fc_K = nn.Linear(emb_size, emb_size)
        # V 矩阵
        self.fc_V = nn.Linear(emb_size, emb_size)

        self.out_fc = nn.Linear(emb_size, emb_size)
        #
        self.norm = nn.LayerNorm(normalized_shape=emb_size, elementwise_affine=True)

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, Q, K, V, mask):
        # Q, K, V 指的是 embedding + pe 之后的结果
        # b句话,每句话50个词,每个词编码成32维向量
        # Q,K,V = [b, 256 + 1, 768]

        # 批量
        b = Q.shape[0]

        # 保留下原始的Q,后面要做短接用
        clone_Q = Q.clone()

        # 规范化
        Q = self.norm(Q)
        K = self.norm(K)
        V = self.norm(V)

        # 线性运算,维度不变
        # [b, 256 + 1, 768] -> [b, 256 + 1, 768]
        K = self.fc_K(K)
        V = self.fc_V(V)
        Q = self.fc_Q(Q)

        # 拆分成多个头
        # b句话,每句话50个词,每个词编码成32维向量,4个头,每个头分到8维向量
        # [b, 256 + 1, 768] -> [b, 4, 256 + 1, 192]
        Q = Q.reshape(b, self.max_len, 4, self.emb_size // 4).permute(0, 2, 1, 3)
        K = K.reshape(b, self.max_len, 4, self.emb_size // 4).permute(0, 2, 1, 3)
        V = V.reshape(b, self.max_len, 4, self.emb_size // 4).permute(0, 2, 1, 3)

        # 计算注意力
        # [b, 4, 256 + 1, 192] -> [b, 256 + 1, 768]
        score = attention(Q, K, V, mask, self.max_len, self.emb_size)

        # 计算输出,维度不变
        # [b, 256 + 1, 768] -> [b, 256 + 1, 768]
        score = self.dropout(self.out_fc(score))

        # 短接
        score = clone_Q + score
        return score

# 位置编码层
class PatchEmbedding(nn.Module):
    def __init__(self,
                 in_channels=IMAGE_DEFAULT_CHANNELS,
                 patch_size=PATCH_DEFAULT_SIZE,
                 emb_size=DEFAULT_HIDDEN_SIZE,
                 img_size=IMAGE_DEFAULT_INNER_SIZE):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=emb_size,
                      kernel_size=patch_size,
                      stride=patch_size)
        )
        self.max_seq_len = (img_size // patch_size) ** 2 + 1
        self.emb_size = emb_size
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.positions = nn.Parameter(torch.randn(self.max_seq_len, emb_size))

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        # [B, C, H, W] -> [B, C', H', W']
        x = self.projection(x)
        # [B, C', H', W'] -> [B, H', W', C']
        x = x.permute(0, 2, 3, 1)
        # [B, H', W', C'] -> [B, PATCHES_COUNT, C']
        x = x.view(b, -1, self.emb_size)
        # [1, 1, C'] -> [B, 1, C']
        cls_tokens = self.cls_token.repeat(b, 1, 1)
        # [B, PATCHES_COUNT, C'] -> [B, PATCHES_COUNT + 1, C']
        x = torch.cat([cls_tokens, x], dim=1)
        # 融入位置编码信息
        x += self.positions

        return x

class ClassificationHead(nn.Module):
    def __init__(self, emb_size= DEFAULT_HIDDEN_SIZE, n_classes=CLASS_COUNT):
        super(ClassificationHead, self).__init__()
        self.classification_head = nn.Sequential(
            nn.LayerNorm(normalized_shape=emb_size),
            nn.Linear(in_features=emb_size, out_features=n_classes)
        )

    def forward(self, x: Tensor) -> Tensor:
        # [B, PATCHES_COUNT, C'] -> [B, C']
        x = x.mean(dim=1)
        return self.classification_head(x)

# 全连接输出层
class FullyConnectedOutput(nn.Module):
    def __init__(self, hidden_size=DEFAULT_HIDDEN_SIZE):
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=hidden_size, out_features=2 * hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=2 * hidden_size, out_features=hidden_size),
            torch.nn.Dropout(p=0.1)
        )

        self.norm = torch.nn.LayerNorm(normalized_shape=hidden_size,
                                       elementwise_affine=True)

    def forward(self, x):
        # 保留下原始的x,后面要做短接用
        clone_x = x.clone()

        # 规范化
        x = self.norm(x)

        # 线性全连接运算
        # [b, 256 + 1, 768] -> [b, 256 + 1, 768]
        out = self.fc(x)

        # 做短接
        out = clone_x + out

        return out