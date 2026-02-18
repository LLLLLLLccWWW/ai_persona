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
    
    def __init__(self, d, num_heads=8):
        """
        初始化自注意力層
        
        參數：
            d (int): 嵌入維度，決定每個詞用多少維度的向量來表示
                    例如 d=512 表示每個詞是一個 512 維的向量
        """
        super().__init__()

        # 確保可以整除
        assert d % num_heads == 0, f"d_model ({d}) 必須能被 num_heads ({num_heads}) 整除"
        
        self.d = d
        self.num_heads = num_heads
        self.d_k = d // num_heads  # 每個頭的維度

        # 創建三個線性轉換層，用於生成 Query, Key, Value
        # nn.Linear(in_features, out_features): 執行線性變換 y = xW^T + b
        self.q = nn.Linear(d, d)  # Query (查詢): "我要找什麼？"
        self.k = nn.Linear(d, d)  # Key (鍵): "我是什麼？"
        self.v = nn.Linear(d, d)  # Value (值): "我的內容是什麼？"

        self.out_proj = nn.Linear(d, d)  # 最終輸出投影層

        # Dropout：訓練時隨機關閉 10% 的神經元，防止過擬合
        # model.eval() 後會自動關閉，不影響生成
        self.attn_dropout = nn.Dropout(0.1)  # 注意力權重的 dropout，防止過擬合

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
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        # torch.tril() 生成下三角矩陣:
        # [[1, 0, 0],
        #  [1, 1, 0],
        #  [1, 1, 1]]
        scores = scores.masked_fill(mask == 0, -1e9)
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
    
    def __init__(self, d):
        """
        初始化 Transformer 區塊
        
        參數：
            d (int): 嵌入維度
        """
        super().__init__()
        
        # 子層 1: 自注意力層
        self.attn = SelfAttention(d)
        
        # 子層 2: 前饋神經網路 (Feed-Forward Network, FFN)
        # 結構: Linear → ReLU → Linear
        # 維度變化: d → 4d → d (先擴展再壓縮)
        self.ff = nn.Sequential(
            nn.Linear(d, d * 4),    # 擴展層: d → 4d
            nn.ReLU(),              # 激活函數: ReLU(x) = max(0, x)
            nn.Linear(d * 4, d)     # 壓縮層: 4d → d
        )
        
        # 層歸一化 (Layer Normalization)
        # 作用：標準化每個樣本的特徵，使訓練更穩定
        # 公式: y = (x - mean(x)) / std(x) * gamma + beta
        self.norm1 = nn.LayerNorm(d)  # 用於自注意力後
        self.norm2 = nn.LayerNorm(d)  # 用於前饋網路後

        # Dropout：前饋網路輸出後也加 Dropout
        self.ff_dropout = nn.Dropout(0.1)  # 前饋網路的 dropout，防止過擬合
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
    
    def __init__(self, vocab_size, embed=512, block_size=256):
        """
        初始化 MiniGPT 模型
        
        參數：
            vocab_size (int): 詞彙表大小，有多少個不同的詞
                             例如: vocab_size=300 表示模型認識 300 個詞
            embed (int): 嵌入維度，預設 512
                        每個詞會被轉換成 512 維的向量
            block_size (int): 最大序列長度，預設 256
                             模型一次最多能處理 256 個詞
        """
        super().__init__()
        self.block_size = block_size
        
        # 1. 詞嵌入層 (Token Embedding)
        # 作用：將詞的索引（整數）轉換為稠密向量
        # 例如：詞 "apple" (索引 5) → 512 維向量 [0.1, -0.3, 0.7, ...]
        self.embed = nn.Embedding(vocab_size, embed)
        # 這會創建一個大小為 (vocab_size, embed) 的查找表

        # 2. 位置嵌入層 (Position Embedding)
        # 作用：為每個位置添加位置信息
        # 為什麼需要：自注意力本身不知道詞的順序
        # 例如：第 1 個位置 → 位置向量 1，第 2 個位置 → 位置向量 2
        self.pos = nn.Embedding(block_size, embed)
        
        # 3. Transformer 區塊
        # nn.Sequential(*[...]) 將多個層串聯起來
        # 這裡堆疊了 4 個 TransformerBlock
        self.blocks = nn.Sequential(*[TransformerBlock(embed) for _ in range(4)])
        
        # 4. 最終層歸一化
        # 在輸出前再做一次歸一化，提高穩定性
        self.ln = nn.LayerNorm(embed)
        
        # 5. 輸出層 (Language Model Head)
        # 作用：將 Transformer 的輸出 (embed 維) 轉換為詞彙表大小的分數
        # 每個詞會得到一個分數，分數越高表示該詞出現的機率越大
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