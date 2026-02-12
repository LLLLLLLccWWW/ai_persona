# 建立模型
import torch
import torch.nn as nn

class MiniGPT(nn.Module):
    def __init__(self,vocab_size,embed=256):
        super().__init__()
        # 嵌入層
        self.embed = nn.Embedding(vocab_size,embed)
        # 線性變換
        self.fc = nn.Linear(embed,vocab_size)

    def forward(self,x):
        x = self.embed(x)
        return self.fc(x)