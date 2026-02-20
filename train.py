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

# ⭐ 新增：分割訓練集和驗證集（90% 訓練，10% 驗證）
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

print(f"資料張量形狀: {data.shape}")
print(f"訓練集大小: {len(train_data)} 個詞 ({len(train_data)/len(data)*100:.1f}%)")
print(f"驗證集大小: {len(val_data)} 個詞 ({len(val_data)/len(data)*100:.1f}%)")


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
    embed=128,                # 嵌入維度（每個詞用 128 維向量表示）
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
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=5000)

# 損失函數：交叉熵損失 (Cross Entropy Loss)
# 用於多分類問題，衡量預測分布和真實分布的差異
# 公式: Loss = -log(P(正確詞))
loss_fn = torch.nn.CrossEntropyLoss()


# ============================================================================
# 第四部分：批次資料生成函數
# ============================================================================

def get_batch(split='train'):
    """
    從訓練資料或驗證資料中隨機抽取一個批次
    
    參數：
        split (str): 'train' 或 'val'，決定從哪個資料集取樣
    
    返回：
        x: 輸入序列，形狀為 (batch_size, block_size)
        y: 目標序列，形狀為 (batch_size, block_size)
    """
    # 根據 split 選擇資料集
    data = train_data if split == 'train' else val_data
    
    # 隨機選擇 batch_size 個起始位置
    ix = torch.randint(len(data) - block_size, (batch_size,))
    
    # 從每個起始位置切出 block_size 長度的序列
    x = torch.stack([data[i:i+block_size] for i in ix]).to(device)
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]).to(device)
    
    return x, y


# ============================================================================
# 第五部分：訓練循環
# ============================================================================

print("\n=== 開始訓練 ===")

num_epochs = 50000
print_interval = 500   # 每 500 輪印出一次（增加頻率）
eval_interval = 1000   # 每 1000 輪評估（降低頻率，給模型更多時間）
eval_iters = 50

# ⭐ 放寬早停機制
best_val_loss = float('inf')
patience = 20  # 驗證損失未改善多少次後觸發早停
patience_counter = 0
min_epochs = 6000  # 至少訓練 6000 輪才能早停

def estimate_loss():
    """估計訓練集和驗證集的平均損失"""
    model.eval()  # 切換到評估模式（關閉 dropout）
    losses = {'train': 0.0, 'val': 0.0}
    
    for split in ['train', 'val']:
        total_loss = 0.0
        for _ in range(eval_iters):
            x, y = get_batch(split)
            with torch.no_grad():
                out = model(x)
                loss = loss_fn(out.view(-1, len(vocab)), y.view(-1))
                total_loss += loss.item()
        losses[split] = total_loss / eval_iters
    
    model.train()  # 切換回訓練模式
    return losses

for epoch in range(num_epochs):
    
    # --- 訓練步驟 ---
    x, y = get_batch('train')  # 只從訓練集取樣
    
    out = model(x)
    out_reshaped = out.view(-1, len(vocab))
    y_reshaped = y.view(-1)
    loss = loss_fn(out_reshaped, y_reshaped)
    
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    scheduler.step()
    
    # --- 定期評估 ---
    if epoch % eval_interval == 0:
        losses = estimate_loss()
        print(f"Epoch {epoch:5d} | Train Loss: {losses['train']:.4f} | Val Loss: {losses['val']:.4f}")
        
        # ⭐ 早停檢查（但要達到最小訓練輪數）
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            patience_counter = 0
            torch.save(model.state_dict(), "persona_best.pt")
            print(f"  ✓ 新的最佳模型！驗證損失: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            # 只有在達到最小訓練輪數後才允許早停
            if patience_counter >= patience and epoch >= min_epochs:
                print(f"\n早停觸發！驗證損失已 {patience} 次未改善")
                print(f"最佳驗證損失: {best_val_loss:.4f}")
                break
            elif patience_counter >= patience:
                print(f"  (驗證損失未改善 {patience_counter} 次，但未達最小訓練輪數 {min_epochs})")
    
    elif epoch % print_interval == 0:
        print(f"Epoch {epoch:5d} | Loss: {loss.item():.4f}")

print(f"\n訓練結束！最佳驗證損失: {best_val_loss:.4f}")


# ============================================================================
# 第六部分：儲存模型
# ============================================================================

print("\n=== 儲存模型 ===")

# 載入最佳模型並儲存為最終版本
model.load_state_dict(torch.load("persona_best.pt"))
torch.save(model.state_dict(), "persona.pt")

print("✓ 最佳模型已儲存到 persona.pt")
print(f"✓ 最佳驗證損失: {best_val_loss:.4f}")
print("訓練完成！")