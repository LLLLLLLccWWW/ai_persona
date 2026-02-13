import torch
from model import MiniGPT

# ====== 1. 資料準備 ======
# 讀取訓練資料
text = open('data/input.txt').read()  # 讀取文字檔

# 分詞：將文字分割成詞
words = text.split()  # split(): 按空格分割字串，返回詞的列表

# 建立詞彙表：取得所有不重複的詞，並排序
vocab = sorted(set(words))  # set(): 去除重複，sorted(): 排序

# 建立詞到索引的映射（字典）
# stoi = string to index，將詞轉換為數字
stoi = {w: i for i, w in enumerate(vocab)}
# enumerate(vocab) 會產生 (0, 'word1'), (1, 'word2'), ...

# 建立索引到詞的映射（字典）
# itos = index to string，將數字轉換回詞
itos = {i: w for w, i in stoi.items()}

# 將所有詞轉換成索引序列
# torch.tensor(): 創建張量
# dtype=torch.long: 資料型態為長整數（64位元整數）
data = torch.tensor([stoi[c] for c in words], dtype=torch.long)


# ====== 2. 模型初始化 ======
# 創建模型實例
model = MiniGPT(len(vocab))  # len(vocab): 詞彙表大小

# torch.optim.Adam: Adam 優化器
# model.parameters(): 取得模型所有可訓練的參數
# 1e-3: 學習率 (learning rate) = 0.001
opt = torch.optim.Adam(model.parameters(), 1e-3)

# torch.nn.CrossEntropyLoss: 交叉熵損失函數
# 用於多分類問題，結合了 softmax 和負對數似然損失
loss_fn = torch.nn.CrossEntropyLoss()


# ====== 3. 訓練循環 ======
for epoch in range(3000):  # 訓練 3000 個 epoch（訓練輪次）
    
    # 準備輸入和目標
    # x: 輸入序列（除了最後一個詞）
    # y: 目標序列（除了第一個詞）
    # 這樣每個位置的目標就是預測下一個詞
    x = data[:-1]  # [:-1]: 從開始到倒數第二個
    y = data[1:]   # [1:]: 從第二個到最後
    
    # 前向傳播：將輸入送進模型
    out = model(x)  # 輸出形狀: (sequence_length, vocab_size)
    
    # 計算損失
    # out.view(-1, len(vocab)): 重塑輸出為 (batch_size, vocab_size)
    #   -1 表示自動計算該維度大小
    # y: 目標索引
    loss = loss_fn(out.view(-1, len(vocab)), y)

    # 反向傳播三步驟：
    
    # 1. 梯度清零
    # 因為 PyTorch 會累積梯度，所以每次迭代前要清零
    opt.zero_grad()

    # 2. 反向傳播：計算梯度
    # 自動計算所有參數對損失的梯度
    loss.backward()
    
    # 3. 更新參數
    # 使用計算出的梯度更新模型參數
    opt.step()

    # 每 500 個 epoch 印出一次損失值
    if epoch % 500 == 0:
        # loss.item(): 將張量轉換為 Python 數值
        print('loss : ', loss.item())

# ====== 4. 儲存模型 ======
# torch.save(): 儲存模型
# model.state_dict(): 取得模型的所有參數（權重和偏差）
# "persona.pt": 儲存的檔案名稱（.pt 是 PyTorch 的慣用副檔名）
torch.save(model.state_dict(), "persona.pt")
print("Training finished")