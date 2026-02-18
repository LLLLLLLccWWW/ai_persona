"""
MiniGPT 訓練腳本
==============
這個腳本負責訓練 MiniGPT 模型
主要流程：
1. 載入並處理訓練資料
2. 創建模型和優化器
3. 訓練循環（前向傳播、計算損失、反向傳播、更新參數）
4. 儲存訓練好的模型
"""

import torch
from model import MiniGPT


# ============================================================================
# 第一部分：資料準備
# ============================================================================

print("=== 載入訓練資料 ===")

# 讀取訓練文本檔案
text = open('data/input.txt').read()
# 例如：text = "I love AI. AI is amazing. I want to learn more about AI."

# 將文本分割成詞（以空格為分隔符）
words = text.split()
# 例如：words = ["I", "love", "AI.", "AI", "is", "amazing.", ...]

# 建立詞彙表：取得所有不重複的詞，並按字母順序排序
vocab = sorted(set(words))
# set(words) 去除重複的詞
# sorted() 按字母順序排序
# 例如：vocab = ["AI", "AI.", "I", "about", "amazing.", "is", "learn", "love", "more", "to", "want"]

print(f"詞彙表大小: {len(vocab)} 個詞")
print(f"訓練資料總詞數: {len(words)} 個詞")

# 建立詞到索引的映射字典 (string to index)
# 用途：將詞轉換為數字，因為模型只能處理數字
stoi = {w: i for i, w in enumerate(vocab)}
# enumerate(vocab) 產生 (索引, 詞) 對
# 例如：stoi = {"AI": 0, "AI.": 1, "I": 2, "about": 3, ...}

# 建立索引到詞的映射字典 (index to string)
# 用途：將模型輸出的數字轉換回詞
itos = {i: w for w, i in stoi.items()}
# 例如：itos = {0: "AI", 1: "AI.", 2: "I", 3: "about", ...}

# 將所有詞轉換成索引序列
# torch.tensor(): 創建張量（多維數組）
# dtype=torch.long: 使用 64 位元整數類型
data = torch.tensor([stoi[w] for w in words], dtype=torch.long)
# 例如：data = tensor([2, 7, 1, 0, 5, 4, 2, 10, 9, 6, 8, 3, 0])
#                     I  love AI. AI is amazing. I want to learn more about AI

print(f"資料張量形狀: {data.shape}")


# ============================================================================
# 第二部分：超參數設定
# ============================================================================

print("\n=== 設定超參數 ===")

# block_size: 模型能處理的最大序列長度
# 太小：模型看不到足夠的上下文
# 太大：訓練速度慢，記憶體消耗大
block_size = 256

# batch_size: 每次訓練使用的序列數量
# 太小：訓練不穩定，收斂慢
# 太大：記憶體消耗大，可能裝不下
batch_size = 128

# device: 使用 GPU 或 CPU
# GPU (cuda) 比 CPU 快 10-100 倍
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"使用設備: {device}")

if device == 'cuda':
    print(f"GPU 名稱: {torch.cuda.get_device_name(0)}")
    print(f"GPU 記憶體: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")


# ============================================================================
# 第三部分：模型初始化
# ============================================================================

print("\n=== 創建模型 ===")

# 創建 MiniGPT 模型
model = MiniGPT(
    vocab_size=len(vocab),    # 詞彙表大小
    embed=512,                # 嵌入維度（每個詞用 512 維向量表示）
    block_size=block_size     # 最大序列長度
).to(device)  # 將模型移動到 GPU（如果可用）

# 計算模型參數量
total_params = sum(p.numel() for p in model.parameters())
print(f"模型參數量: {total_params:,} ({total_params/1e6:.2f}M)")

# 優化器：AdamW (Adam with Weight Decay)
# AdamW 是 Adam 的改進版本，加入了權重衰減正則化
# lr (learning rate): 學習率，控制參數更新的步長
#   - 太大：訓練不穩定，可能發散
#   - 太小：訓練太慢，可能卡在局部最優
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

# 損失函數：交叉熵損失 (Cross Entropy Loss)
# 用於多分類問題，衡量預測分布和真實分布的差異
# 公式: Loss = -log(P(正確詞))
loss_fn = torch.nn.CrossEntropyLoss()


# ============================================================================
# 第四部分：批次資料生成函數
# ============================================================================

def get_batch():
    """
    從訓練資料中隨機抽取一個批次
    
    返回：
        x: 輸入序列，形狀為 (batch_size, block_size)
        y: 目標序列，形狀為 (batch_size, block_size)
    
    例如：
        如果 data = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        block_size = 3, batch_size = 2
        
        可能返回：
        x = [[2, 3, 4],    ← 從位置 1 開始的 3 個詞
             [5, 6, 7]]    ← 從位置 4 開始的 3 個詞
        
        y = [[3, 4, 5],    ← x 的下一個詞（目標）
             [6, 7, 8]]
    """
    # 隨機選擇 batch_size 個起始位置
    # randint(low, high, size) 產生 [low, high) 範圍內的隨機整數
    # len(data) - block_size 確保不會超出資料範圍
    ix = torch.randint(len(data) - block_size, (batch_size,))
    
    # 從每個起始位置切出 block_size 長度的序列
    # 使用 torch.stack 將多個序列堆疊成一個批次
    x = torch.stack([data[i:i+block_size] for i in ix]).to(device)
    # x 的形狀: (batch_size, block_size)
    
    # 目標序列是輸入序列的下一個詞
    # 例如：輸入 [1, 2, 3]，目標 [2, 3, 4]
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]).to(device)
    # y 的形狀: (batch_size, block_size)
    
    return x, y


# ============================================================================
# 第五部分：訓練循環
# ============================================================================

print("\n=== 開始訓練 ===")

num_epochs = 50000  # 訓練輪數
print_interval = 1000  # 每多少輪印出一次損失

for epoch in range(num_epochs):
    
    # --- 步驟 1: 取得一個批次的訓練資料 ---
    x, y = get_batch()
    # x: 輸入序列 (batch_size, block_size)
    # y: 目標序列 (batch_size, block_size)
    
    # --- 步驟 2: 前向傳播 (Forward Pass) ---
    # 將輸入送進模型，得到預測
    out = model(x)
    # out 的形狀: (batch_size, block_size, vocab_size)
    # out[i, j, k] 表示：第 i 個序列，第 j 個位置，預測詞 k 的分數
    
    # --- 步驟 3: 計算損失 (Loss) ---
    # 需要重塑張量以符合 CrossEntropyLoss 的要求
    # CrossEntropyLoss 期望:
    #   - 輸入: (N, C) 其中 N 是樣本數，C 是類別數
    #   - 目標: (N,) 其中每個元素是類別索引
    
    # 重塑輸出: (batch_size, block_size, vocab_size) → (batch_size*block_size, vocab_size)
    out_reshaped = out.view(-1, len(vocab))
    # -1 表示自動計算該維度 = batch_size * block_size
    
    # 重塑目標: (batch_size, block_size) → (batch_size*block_size,)
    y_reshaped = y.view(-1)
    
    # 計算交叉熵損失
    loss = loss_fn(out_reshaped, y_reshaped)
    # 損失值越小，表示模型預測越準確
    
    # --- 步驟 4: 反向傳播 (Backward Pass) ---
    # 4.1 清空之前的梯度
    # PyTorch 會累積梯度，所以每次迭代前要清零
    opt.zero_grad()
    
    # 4.2 計算梯度
    # 自動計算所有參數對損失的梯度（偏導數）
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # 梯度裁剪：防止梯度爆炸，將梯度的范數限制在 1.0 以內
    
    # 4.3 更新參數
    # 使用計算出的梯度更新模型參數
    # 新參數 = 舊參數 - 學習率 × 梯度
    opt.step()
    
    # --- 步驟 5: 定期印出訓練進度 ---
    if epoch % print_interval == 0:
        # .item() 將 PyTorch 張量轉換為 Python 數值
        print(f"Epoch {epoch:5d} / {num_epochs} | Loss: {loss.item():.4f}")


# ============================================================================
# 第六部分：儲存模型
# ============================================================================

print("\n=== 儲存模型 ===")

# 儲存模型參數到檔案
# model.state_dict() 返回一個字典，包含所有可學習的參數
# 例如: {'embed.weight': tensor([...]), 'blocks.0.attn.q.weight': tensor([...]), ...}
torch.save(model.state_dict(), "persona.pt")
# .pt 是 PyTorch 模型的慣用副檔名

print("✓ 模型已儲存到 persona.pt")
print("訓練完成！")


