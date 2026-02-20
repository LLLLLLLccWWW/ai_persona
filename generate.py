"""
MiniGPT 文字生成腳本
==================
這個腳本負責載入訓練好的模型並生成文字
主要流程：
1. 載入詞彙表（必須與訓練時相同）
2. 載入訓練好的模型
3. 根據用戶輸入生成文字
"""

import torch
from model import MiniGPT


# ============================================================================
# 第一部分：載入詞彙表
# ============================================================================

print("=== 載入詞彙表 ===")

# 重要：詞彙表必須與訓練時完全相同！
# 因為模型是基於特定的詞彙表訓練的

# 讀取訓練時使用的相同文本檔案
text = open('data/input.txt').read()

# 分割成詞
words = text.split()

# 建立詞彙表（必須與訓練時的順序完全一致）
vocab = sorted(set(words))

# 建立詞 ↔ 索引的映射
stoi = {w: i for i, w in enumerate(vocab)}  # 詞 → 索引
itos = {i: w for w, i in stoi.items()}      # 索引 → 詞

print(f"詞彙表大小: {len(vocab)} 個詞")


# ============================================================================
# 第二部分：設定與載入模型
# ============================================================================

print("\n=== 載入模型 ===")

# 超參數（必須與訓練時相同）
vocab_size = len(vocab)
block_size = 256
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"使用設備: {device}")

# 創建模型架構（必須與訓練時完全相同）
model = MiniGPT(
    vocab_size=vocab_size,
    embed=128,           # 嵌入維度
    block_size=block_size
).to(device)

# 載入訓練好的模型參數
# torch.load() 載入參數檔案
# model.load_state_dict() 將參數載入到模型中
model.load_state_dict(torch.load("persona.pt"))

# 設定為評估模式
# 這會關閉 dropout 等訓練專用的功能
model.eval()

print("✓ 模型載入成功")


# ============================================================================
# 第三部分：文字生成函數
# ============================================================================

def generate(start="I", length=200):
    """
    根據起始文字生成新文字
    
    參數：
        start (str): 起始提示詞，可以是一個或多個詞
                    例如: "I like", "AI is", "Hello"
        length (int): 要生成的詞數量
    
    返回：
        生成的完整文字（包含起始詞）
    
    生成流程：
        1. 將起始文字轉換為索引序列
        2. 循環預測下一個詞
        3. 將預測的詞加入序列
        4. 重複直到生成足夠的詞
        5. 將索引序列轉換回文字
    """
    
    # --- 步驟 1: 處理起始文字 ---
    
    # 將起始文字分割成詞
    start_words = start.split()
    # 例如: "I like AI" → ["I", "like", "AI"]
    
    # 過濾掉不在詞彙表中的詞
    # 如果用戶輸入了訓練資料中沒有的詞，就跳過
    start_words = [w for w in start_words if w in stoi]
    
    # 如果所有詞都不在詞彙表中，使用詞彙表的第一個詞
    if not start_words:
        start_words = [vocab[0]]
        print(f"輸入的詞不在詞彙表中，使用預設起始詞: {vocab[0]}")
    
    # 將詞轉換為索引序列
    # 例如: ["I", "like"] → [2, 7]
    indices = [stoi[w] for w in start_words]
    
    # 創建張量並增加批次維度
    # (T,) → (1, T) 其中 T 是起始序列長度
    x = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)
    # unsqueeze(0) 在第 0 維增加一個維度
    # 例如: tensor([2, 7]) → tensor([[2, 7]])
    
    
    # --- 步驟 2: 逐詞生成循環 ---
    
    for _ in range(length):
        
        # 2.1 將當前序列送入模型
        with torch.no_grad():  # 不計算梯度，節省記憶體和計算
            out = model(x)
        # out 的形狀: (1, T, vocab_size)
        # out[0, i, :] 是位置 i 對所有詞的預測分數
        
        # 2.2 只取最後一個位置的輸出（預測下一個詞）
        logits = out[:, -1, :]
        # 形狀: (1, vocab_size)
        # logits[0, i] 是預測詞 i 的分數
        
        
        # --- 2.3 溫度採樣 (Temperature Sampling) ---
        # 溫度控制生成的隨機性
        
        temperature = 0.8
        # temperature < 1: 更保守，偏向選擇高機率的詞
        #                 生成的文字更連貫但可能重複
        # temperature = 1: 標準機率分布
        # temperature > 1: 更隨機，給低機率詞更多機會
        #                 生成的文字更有創意但可能不連貫
        
        # 將 logits 除以溫度
        scaled_logits = logits / temperature
        
        # 轉換為機率分布（softmax）
        # softmax(x_i) = exp(x_i) / sum(exp(x_j))
        probs = torch.softmax(scaled_logits, dim=-1)
        # 形狀: (1, vocab_size)
        # probs[0, i] 是選擇詞 i 的機率
        
        
        # --- 2.4 Top-k 採樣 ---
        # 只從機率最高的 k 個詞中選擇
        # 好處：避免選到機率極低的奇怪詞
        
        k = 50  # 只從前 50 個詞中選擇
        # 取出機率最高的 k 個詞
        # torch.topk(input, k) 返回最大的 k 個值及其索引
        top_probs, top_indices = torch.topk(probs, k, dim=-1)
        # top_probs 形狀: (1, k)
        # top_indices 形狀: (1, k)
        
        # 重新歸一化：讓前 k 個詞的機率和為 1
        top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True)
        
        
        # --- 2.5 採樣下一個詞 ---
        
        # 根據機率分布隨機選擇一個詞
        # torch.multinomial(probs, num_samples) 按照機率分布採樣
        sampled_idx = torch.multinomial(top_probs, 1)
        # sampled_idx 形狀: (1, 1)
        # 這是在 top_k 中的相對位置
        
        # 取得實際的詞索引
        next_token = top_indices[0, sampled_idx].item()
        # .item() 將張量轉換為 Python 整數
        
        
        # --- 2.6 將新詞加入序列 ---
        
        # 創建新詞的張量
        next_token_tensor = torch.tensor([[next_token]], device=device)
        # 形狀: (1, 1)
        
        # 連接到現有序列
        # torch.cat([tensor1, tensor2], dim) 在指定維度連接張量
        x = torch.cat([x, next_token_tensor], dim=-1)
        # 例如: [[2, 7]] + [[15]] → [[2, 7, 15]]
    
    
    # --- 步驟 3: 將索引序列轉換回文字 ---
    
    # x[0] 取出批次維度，得到 1D 張量
    # 例如: tensor([[2, 7, 15, 3]]) → tensor([2, 7, 15, 3])
    
    # 將每個索引轉換回詞
    generated_words = [itos[i.item()] for i in x[0]]
    # 例如: [2, 7, 15, 3] → ["I", "like", "AI", "and"]
    
    # 用空格連接所有詞
    result = ' '.join(generated_words)
    # 例如: "I like AI and"
    
    return result


# ============================================================================
# 第四部分：互動式生成
# ============================================================================

print("\n=== 開始生成 ===")
print("輸入提示詞來生成文字（輸入 'quit' 退出）")
print("-" * 50)

while True:
    # 等待用戶輸入
    prompt = input("\nYou: ")
    
    # 退出條件
    if prompt.lower() in ['quit', 'exit', 'q']:
        print("再見！")
        break
    
    # 生成文字
    try:
        generated_text = generate(prompt, length=200)
        print(f"\nAI: {generated_text}")
    except Exception as e:
        print(f" 生成時發生錯誤: {e}")


"""
使用技巧：

1. 調整生成長度：
   generate(prompt, length=100)  # 生成 100 個詞

2. 調整溫度（在 generate 函數內）：
   temperature = 0.5   # 更保守、更連貫
   temperature = 1.0   # 平衡
   temperature = 1.5   # 更隨機、更有創意

3. 調整 top-k：
   k = 10   # 只從前 10 個詞中選（更保守）
   k = 100  # 從前 100 個詞中選（更多樣）

4. 使用更好的起始提示：
   - 好: "I think AI is"
   - 差: "xyz" (詞彙表中沒有)
"""