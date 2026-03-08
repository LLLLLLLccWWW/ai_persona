"""
MiniGPT 模型架構
===============
這是一個簡化版的 GPT (Generative Pre-trained Transformer) 模型
主要包含三個部分：
1. SelfAttention: 自注意力機制
2. TransformerBlock: Transformer 區塊
3. MiniGPT: 完整的 GPT 模型
"""

import torch
import torch.nn as nn
import math


class SelfAttention(nn.Module):
    """
    自注意力機制 (Self-Attention)
    
    功能：讓模型能夠「注意」輸入序列中的不同位置
    例如：在翻譯 "The cat sat on the mat" 時，
         翻譯 "sat" 需要注意 "cat"（誰坐）和 "mat"（坐在哪）
    
    原理：使用 Query、Key、Value 三個矩陣計算注意力權重
    """
    
    def __init__(self, d, num_heads=2, dropout=0.3):
        """
        初始化自注意力層
        
        參數：
            d (int): 嵌入維度，決定每個詞用多少維度的向量來表示
            num_heads (int): 注意力頭數
            dropout (float): 注意力權重的 dropout 比例
        """
        super().__init__()

        # 確保可以整除
        assert d % num_heads == 0, f"d_model ({d}) 必須能被 num_heads ({num_heads}) 整除"
        
        self.d = d
        self.num_heads = num_heads
        self.d_k = d // num_heads  # 每個頭的維度

        # 線性層生成 Query, Key, Value
        self.q = nn.Linear(d, d)  
        self.k = nn.Linear(d, d)  
        self.v = nn.Linear(d, d)  

        self.out_proj = nn.Linear(d, d)

        # Dropout for attention
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        前向傳播：計算自注意力
        
        參數：
            x: 輸入張量，形狀為 (B, T, C)
               B = batch size (批次大小，一次處理幾個序列)
               T = sequence length (序列長度，一個序列有幾個詞)
               C = embedding dimension (嵌入維度，每個詞的向量大小)
        
        返回：
            out: 注意力加權後的輸出，形狀為 (B, T, C)
        """
        B, T, C = x.shape
        
        # 步驟 1: 通過線性層生成 Q, K, V
        q = self.q(x)  # Query: (B, T, C) → (B, T, C)
        k = self.k(x)  # Key:   (B, T, C) → (B, T, C)
        v = self.v(x)  # Value: (B, T, C) → (B, T, C)

        q = q.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        # 步驟 2: 計算注意力分數 (Attention Scores)
        # scores[i,j] 表示位置 i 對位置 j 的注意力程度
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)
        # @ 是矩陣乘法
        # k.transpose(-2, -1) 將最後兩個維度轉置: (B, T, C) → (B, C, T)
        # 結果形狀: (B, T, T)
        # 除以 sqrt(C) 是為了防止分數過大導致 softmax 飽和
        
        # 步驟 3: 因果遮罩 (Causal Mask)
        # 目的：防止模型"偷看"未來的詞
        # 例如：預測第 3 個詞時，只能看到第 1、2 個詞，不能看到第 4、5... 個詞
        # 使用 bool mask 並使用 -inf 以避開 half precision overflow
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool)).view(1, 1, T, T)
        # torch.tril() 生成下三角矩陣:
        # [[1, 0, 0],
        #  [1, 1, 0],
        #  [1, 1, 1]]
        scores = scores.masked_fill(~mask, float('-inf'))
        # 將上三角部分設為 -∞，這樣 softmax 後會變成 0

        # 步驟 4: 應用 Softmax 得到注意力權重
        attn = torch.softmax(scores, dim=-1)
        # softmax 將分數轉換為機率分布（總和為 1）
        # 形狀: (B, T, T)
        attn = self.attn_dropout(attn)
        
        # 步驟 5: 用注意力權重對 Value 加權求和
        out = attn @ v
        # 形狀: (B, T, T) @ (B, T, C) → (B, T, C)
        # 每個位置都是其他位置 Value 的加權平均

        # 步驟 6: 合併所有頭
        # (B, num_heads, T, d_k) → (B, T, num_heads, d_k) → (B, T, d)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d)
        # .contiguous() 確保記憶體連續，view() 才能正確運作

        # 步驟 7: 輸出投影（讓模型學習整合多頭資訊）
        out = self.out_proj(out)
        # 形狀維持: (B, T, d)
        
        return out


class TransformerBlock(nn.Module):
    """
    Transformer 區塊
    
    結構：
    1. 自注意力層 (Self-Attention)
    2. 前饋神經網路 (Feed-Forward Network)
    3. 殘差連接 (Residual Connection)
    4. 層歸一化 (Layer Normalization)
    
    這是 Transformer 的核心組件，可以堆疊多個以增加模型深度
    """
    
    def __init__(self, d, num_heads=2, dropout=0.3):
        """
        初始化 Transformer 區塊
        
        參數：
            d (int): 嵌入維度
            num_heads (int): 注意力頭數
            dropout (float): dropout 比例
        """
        super().__init__()
        
        # 子層 1: 自注意力層
        self.attn = SelfAttention(d, num_heads=num_heads, dropout=dropout)
        
        # 子層 2: 前饋神經網路 (Feed-Forward Network, FFN)
        self.ff = nn.Sequential(
            nn.Linear(d, d * 4),
            nn.ReLU(),
            nn.Linear(d * 4, d),
            nn.Dropout(dropout)
        )
        
        # 層歸一化
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)

        # Dropout after FFN
        self.ff_dropout = nn.Dropout(dropout)
    def forward(self, x):
        """
        前向傳播
        
        架構 (Post-LN):
        x → [Attention] → (+x) → [Norm] → [FFN] → (+x) → [Norm] → out
            ↑___________↓           ↑_______↓
           殘差連接               殘差連接
        
        參數：
            x: 輸入張量，形狀為 (B, T, C)
        
        返回：
            輸出張量，形狀為 (B, T, C)
        """
        # 第一個子層: 自注意力 + 殘差連接 + 層歸一化
        x = self.norm1(x + self.attn(x))
        # x + self.attn(x) 是殘差連接，幫助梯度流動
        
        # 第二個子層: 前饋網路 + 殘差連接 + 層歸一化
        x = self.norm2(x + self.ff_dropout(self.ff(x)))
        
        return x


class MiniGPT(nn.Module):
    """
    MiniGPT 主模型
    
    完整架構:
    輸入 (詞索引) 
      ↓
    詞嵌入 + 位置嵌入
      ↓
    Transformer Block × 4
      ↓
    層歸一化
      ↓
    線性輸出層
      ↓
    輸出 (每個位置的詞機率分布)
    """
    
    def __init__(self, vocab_size, embed=128, block_size=256, num_layers=4, num_heads=2, dropout=0.2):
        """
        初始化 MiniGPT 模型
        
        參數：
            vocab_size (int): 詞彙表大小，有多少個不同的詞
            embed (int): 嵌入維度，預設 128
            block_size (int): 最大序列長度，預設 256
            num_layers (int): Transformer block 層數
            num_heads  (int): 自注意力頭數
            dropout    (float): dropout 比例，應用於多個子層
        """
        super().__init__()
        self.block_size = block_size
        
        # 1. 詞嵌入層 (Token Embedding)
        self.embed = nn.Embedding(vocab_size, embed)

        # 2. 位置嵌入層 (Position Embedding)
        self.pos = nn.Embedding(block_size, embed)

        self.drop = nn.Dropout(dropout)  # 整體 dropout，防止過擬合
        
        # 3. Transformer 區塊
        self.blocks = nn.Sequential(*[TransformerBlock(embed, num_heads=num_heads, dropout=dropout) for _ in range(num_layers)])
        
        # 4. 最終層歸一化
        self.ln = nn.LayerNorm(embed)
        
        # 5. 輸出層 (Language Model Head)
        self.fc = nn.Linear(embed, vocab_size)

    def forward(self, x):
        """
        前向傳播：從輸入詞索引到輸出詞機率
        
        參數：
            x: 輸入的詞索引序列
               形狀: (B, T)
               B = batch_size (批次大小)
               T = sequence_length (序列長度)
               
               例如：x = [[5, 12, 3],    ← 第 1 個序列
                         [8, 2, 15]]   ← 第 2 個序列
        
        返回：
            logits: 每個位置對每個詞的分數
                   形狀: (B, T, vocab_size)
                   
                   例如：logits[0, 2, :] 是第 1 個序列第 3 個位置
                        對所有詞的分數分布
        """
        B, T = x.shape  # 取得批次大小和序列長度
        
        # 生成位置索引: [0, 1, 2, ..., T-1]
        pos = torch.arange(T, device=x.device)

        # 步驟 1: 嵌入
        # 詞嵌入 + 位置嵌入（廣播相加）
        tok_emb = self.embed(x)      # (B, T) → (B, T, embed)
        pos_emb = self.pos(pos)      # (T,) → (T, embed)
        x = tok_emb + pos_emb        # (B, T, embed) + (T, embed) → (B, T, embed)
        # 廣播機制會自動將 pos_emb 複製 B 次
        
        # 步驟 2: 通過 4 層 Transformer 區塊
        x = self.blocks(x)           # (B, T, embed) → (B, T, embed)
        
        # 步驟 3: 最終層歸一化
        x = self.ln(x)               # (B, T, embed) → (B, T, embed)
        
        # 步驟 4: 投影到詞彙表空間
        logits = self.fc(x)          # (B, T, embed) → (B, T, vocab_size)
        
        # logits[b, t, v] 表示：
        # 在批次 b 的序列中，位置 t 預測詞 v 的分數
        
        return logits