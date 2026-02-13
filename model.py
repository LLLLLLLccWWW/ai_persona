# 建立模型
import torch
import torch.nn as nn, math

class SelfAttention(nn.Module):
    """
    自注意力機制 (Self-Attention)
    讓模型能夠關注輸入序列中的不同位置
    """
    def __init__(self, d):
        """
        初始化自注意力層
        參數:
            d: 嵌入維度 (embedding dimension)，決定每個詞的向量大小
        """
        super().__init__()  # 呼叫父類別 nn.Module 的初始化
        
        # nn.Linear(in_features, out_features): 線性轉換層 y = xW^T + b
        # 這裡創建三個線性層，將輸入轉換成 Query, Key, Value
        self.q = nn.Linear(d, d)  # Query 投影層：輸入 d 維，輸出 d 維
        self.k = nn.Linear(d, d)  # Key 投影層：輸入 d 維，輸出 d 維
        self.v = nn.Linear(d, d)  # Value 投影層：輸入 d 維，輸出 d 維

    def forward(self, x):
        """
        前向傳播
        參數:
            x: 輸入張量，形狀為 (sequence_length, d)
        """
        # 將輸入 x 分別通過三個線性層得到 Q, K, V
        q = self.q(x)  # Query: 查詢向量
        k = self.k(x)  # Key: 鍵向量
        v = self.v(x)  # Value: 值向量

        # 計算注意力分數
        # @: 矩陣乘法運算符
        # k.transpose(-2, -1): 轉置 K 矩陣的最後兩個維度
        # math.sqrt(q.size(-1)): 縮放因子，防止點積過大
        scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
        
        # torch.softmax(input, dim): 計算 softmax，將分數轉換為機率分布
        # dim=-1: 在最後一個維度上進行 softmax
        attn = torch.softmax(scores, dim=-1)
        
        # 用注意力權重對 Value 加權求和
        out = attn @ v
        return out


class TransformerBlock(nn.Module):
    """
    Transformer 區塊
    包含自注意力層和前饋神經網路，以及殘差連接和層歸一化
    """
    def __init__(self, d):
        """
        初始化 Transformer 區塊
        參數:
            d: 嵌入維度
        """
        super().__init__()
        
        self.attn = SelfAttention(d)  # 自注意力層
        
        # nn.Sequential: 按順序執行多個層
        # 前饋神經網路 (Feed-Forward Network)
        self.ff = nn.Sequential(
            nn.Linear(d, d*4),    # 第一層：d → 4d (擴展)
            nn.ReLU(),            # 激活函數：ReLU(x) = max(0, x)
            nn.Linear(d*4, d)     # 第二層：4d → d (壓縮回原始維度)
        )
        
        # nn.LayerNorm(normalized_shape): 層歸一化
        # 對每個樣本的特徵進行標準化，使訓練更穩定
        self.norm1 = nn.LayerNorm(d)  # 第一個歸一化層 (用於注意力後)
        self.norm2 = nn.LayerNorm(d)  # 第二個歸一化層 (用於前饋網路後)

    def forward(self, x):
        """
        前向傳播
        使用殘差連接 (Residual Connection): x + layer(x)
        """
        # 第一個子層：自注意力 + 殘差連接 + 層歸一化
        x = self.norm1(x + self.attn(x))
        
        # 第二個子層：前饋網路 + 殘差連接 + 層歸一化
        x = self.norm2(x + self.ff(x))
        
        return x

    
class MiniGPT(nn.Module):
    """
    迷你 GPT 模型
    包含嵌入層、Transformer 區塊和輸出層
    """
    def __init__(self, vocab_size, embed=512):
        """
        初始化模型
        參數:
            vocab_size: 詞彙表大小（有多少個不同的詞）
            embed: 嵌入維度，預設 512
        """
        super().__init__()
        
        # nn.Embedding(num_embeddings, embedding_dim): 嵌入層
        # 將詞的索引（整數）轉換為稠密向量
        # num_embeddings: 詞彙表大小
        # embedding_dim: 每個詞的向量維度
        self.embed = nn.Embedding(vocab_size, embed)
        
        # Transformer 區塊
        self.block = TransformerBlock(embed)
        
        # nn.Linear(in_features, out_features): 線性輸出層
        # 將 Transformer 的輸出轉換回詞彙表大小的 logits
        self.fc = nn.Linear(embed, vocab_size)

    def forward(self, x):
        """
        前向傳播
        參數:
            x: 輸入的詞索引序列，形狀為 (sequence_length,)
        返回:
            輸出 logits，形狀為 (sequence_length, vocab_size)
        """
        # 1. 嵌入：將詞索引轉換為向量
        x = self.embed(x)  # (sequence_length,) → (sequence_length, embed)
        
        # 2. Transformer 處理
        x = self.block(x)  # (sequence_length, embed) → (sequence_length, embed)
        
        # 3. 輸出投影：轉換為詞彙表大小的分數
        return self.fc(x)  # (sequence_length, embed) → (sequence_length, vocab_size)