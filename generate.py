import torch
from model import MiniGPT

# ====== 1. 載入詞彙表 ======
# 必須與訓練時使用相同的詞彙表！
text = open('data/input.txt').read()
words = text.split()  # 分割成詞
vocab = sorted(set(words))  # 建立詞彙表
stoi = {w: i for i, w in enumerate(vocab)}  # 詞 → 索引
itos = {i: w for w, i in stoi.items()}      # 索引 → 詞


# ====== 2. 載入訓練好的模型 ======
model = MiniGPT(len(vocab))  # 創建模型（必須與訓練時相同架構）

# torch.load(): 載入已儲存的模型參數
# model.load_state_dict(): 將參數載入到模型中
model.load_state_dict(torch.load("persona.pt"))


# ====== 3. 文字生成函數 ======
def generate(start="I", length=200):
    """
    生成文字的函數
    參數:
        start: 起始提示詞（字串）
        length: 要生成的詞數量
    """
    # 將輸入字串分割成詞
    start_words = start.split()
    
    # 過濾掉不在詞彙表中的詞
    # 如果用戶輸入的詞不在訓練資料中，就跳過
    start_words = [w for w in start_words if w in stoi]
    
    # 如果沒有有效的詞，使用詞彙表中的第一個詞
    if not start_words:
        start_words = [vocab[0]]
    
    # torch.tensor(): 將詞列表轉換為索引張量
    # dtype=torch.long: 使用長整數型態
    x = torch.tensor([stoi[w] for w in start_words], dtype=torch.long)

    # 生成循環：逐詞生成
    for _ in range(length):
        # 將當前序列送入模型
        out = model(x)  # 輸出形狀: (sequence_length, vocab_size)
        
        # 只取最後一個位置的輸出（預測下一個詞）
        out = out[-1]  # 形狀: (vocab_size,)

        # === 溫度採樣 (Temperature Sampling) ===
        # 溫度參數控制生成的隨機性
        temperature = 0.9
        # temperature < 1: 更保守，偏好高機率的詞
        # temperature = 1: 標準分布
        # temperature > 1: 更隨機，給低機率詞更多機會
        
        # torch.softmax(input, dim): 將 logits 轉換為機率分布
        # dim=0: 在第 0 維（vocab_size 維度）計算 softmax
        p = torch.softmax(out / temperature, 0)

        # === Top-k 採樣 ===
        # 只從機率最高的 k 個詞中選擇
        k = 50
        # torch.topk(input, k): 返回最大的 k 個值及其索引
        values, indices = torch.topk(p, k)
        # values: 前 k 大的機率值
        # indices: 對應的詞索引
        
        # 重新歸一化：讓前 k 個詞的機率和為 1
        p = values / values.sum()
        
        # torch.multinomial(input, num_samples): 根據機率分布採樣
        # 從前 k 個詞中，根據機率隨機選一個
        c = indices[torch.multinomial(p, 1)].item()
        # .item(): 將單元素張量轉換為 Python 整數
        
        # 將選中的詞加入序列
        # torch.cat([tensor1, tensor2]): 連接張量
        x = torch.cat([x, torch.tensor([c])])
    
    # 將索引序列轉換回文字
    # ' '.join(): 用空格連接所有詞
    return ' '.join([itos[i.item()] for i in x])


# ====== 4. 互動式生成 ======
while True:
    # input(): 等待用戶輸入
    prompt = input("You: ")
    
    # 生成並印出 AI 的回應
    print("AI:", generate(prompt))