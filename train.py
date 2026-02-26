"""
MiniGPT 訓練腳本（過濾低頻詞版本）
"""

import torch
from model import MiniGPT
from collections import Counter

print("=== 載入訓練資料 ===")
text = open('data/input.txt',encoding='utf-8').read()
words = text.split()

print(f"原始詞彙表大小: {len(set(words))} 個詞")
print(f"訓練資料總詞數: {len(words)} 個詞")

# ⭐ 過濾低頻詞
word_counts = Counter(words)
min_count = 2 # 最小詞頻，出現次數少於這個值的詞會被過濾掉
vocab = sorted([w for w, count in word_counts.items() if count >= min_count])

# 用 <UNK> 替換低頻詞
UNK = '<UNK>'
if UNK not in vocab:
    vocab = [UNK] + vocab

filtered_words = [w if word_counts[w] >= min_count else UNK for w in words]

print(f"過濾後詞彙表大小: {len(vocab)} 個詞")
print(f"過濾掉的低頻詞: {len(set(words)) - len(vocab)} 個")
print(f"<UNK> 出現次數: {filtered_words.count(UNK)}")

# 建立映射
stoi = {w: i for i, w in enumerate(vocab)}
itos = {i: w for w, i in stoi.items()}

# 轉換為張量
data = torch.tensor([stoi[w] for w in filtered_words], dtype=torch.long)

# 分割訓練集和驗證集
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

print(f"訓練集大小: {len(train_data)} 個詞 ({len(train_data)/len(data)*100:.1f}%)")
print(f"驗證集大小: {len(val_data)} 個詞 ({len(val_data)/len(data)*100:.1f}%)")

# 計算詞頻統計
print(f"\n詞頻統計:")
print(f"平均每詞出現次數: {len(filtered_words) / len(vocab):.1f} 次")

print("\n=== 設定超參數 ===")
block_size = 256
batch_size = 128
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"使用設備: {device}")

if device == 'cuda':
    print(f"GPU 名稱: {torch.cuda.get_device_name(0)}")

print("\n=== 創建模型 ===")
model = MiniGPT(
    vocab_size=len(vocab),
    embed=128,
    block_size=block_size
).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"模型參數量: {total_params:,} ({total_params/1e6:.2f}M)")

opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50000)
loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

def get_batch(split='train'):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix]).to(device)
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]).to(device)
    return x, y

print("\n=== 開始訓練 ===")
num_epochs = 100000
print_interval = 500
eval_interval = 1000
eval_iters = 50

best_val_loss = float('inf')
patience = 30
patience_counter = 0
min_epochs = 8000

def estimate_loss():
    model.eval()
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
    model.train()
    return losses

for epoch in range(num_epochs):
    x, y = get_batch('train')
    out = model(x)
    out_reshaped = out.view(-1, len(vocab))
    y_reshaped = y.view(-1)
    loss = loss_fn(out_reshaped, y_reshaped)
    
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    # scheduler.step()
    
    if epoch % eval_interval == 0:
        losses = estimate_loss()
        print(f"Epoch {epoch:5d} | Train Loss: {losses['train']:.4f} | Val Loss: {losses['val']:.4f}")
        
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            patience_counter = 0
            torch.save(model.state_dict(), "persona_best.pt")
            print(f"  ✓ 新的最佳模型！驗證損失: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience and epoch >= min_epochs:
                print(f"\n早停觸發！驗證損失已 {patience} 次未改善")
                print(f"最佳驗證損失: {best_val_loss:.4f}")
                break
            elif patience_counter >= patience:
                print(f"  (驗證損失未改善 {patience_counter} 次，但未達最小訓練輪數 {min_epochs})")
    
    elif epoch % print_interval == 0:
        print(f"Epoch {epoch:5d} | Loss: {loss.item():.4f}")

print(f"\n訓練結束！最佳驗證損失: {best_val_loss:.4f}")

print("\n=== 儲存模型 ===")
model.load_state_dict(torch.load("persona_best.pt"))
torch.save(model.state_dict(), "persona.pt")

# 儲存詞彙表
import pickle
with open('vocab.pkl', 'wb') as f:
    pickle.dump({'vocab': vocab, 'stoi': stoi, 'itos': itos}, f)

print("✓ 最佳模型已儲存到 persona.pt")
print("✓ 詞彙表已儲存到 vocab.pkl")
print(f"✓ 最佳驗證損失: {best_val_loss:.4f}")
print("訓練完成！")