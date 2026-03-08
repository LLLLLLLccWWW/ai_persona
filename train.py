"""
MiniGPT 訓練腳本（過濾低頻詞版本）
"""

import os
import torch
from model import MiniGPT
from collections import Counter

import argparse

# --- 命令列參數設定（方便靈活調整） ---
# 新增 argparse 允許透過 CLI 更改訓練參數，方便快速實驗不同設定
parser = argparse.ArgumentParser(description="Train the MiniGPT model with configurable hyperparameters")
parser.add_argument('--input', type=str, default='data/input.txt', help='主要訓練文本路徑 (檔案或資料夾)')
parser.add_argument('--extra_data', type=str, nargs='*', default=None, help='額外的文本檔案或資料夾，可提供多個')
parser.add_argument('--init_from', type=str, default=None, help='如有預訓練權重 (checkpoint) 可指定路徑')
parser.add_argument('--min_count', type=int, default=2, help='保留詞彙的最低頻率')
parser.add_argument('--embed', type=int, default=256, help='嵌入維度')
parser.add_argument('--block_size', type=int, default=64, help='最大序列長度')
parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
parser.add_argument('--num_layers', type=int, default=6, help='Transformer 區塊數')
parser.add_argument('--num_heads', type=int, default=4, help='注意力頭數')
parser.add_argument('--dropout', type=float, default=0.4, help='dropout 比例')
parser.add_argument('--lr', type=float, default=1e-4, help='學習率')
parser.add_argument('--weight_decay', type=float, default=0.01, help='權重衰減')
parser.add_argument('--epochs', type=int, default=100000, help='最大訓練迴圈數')
parser.add_argument('--patience', type=int, default=50, help='早停容忍次數')
parser.add_argument('--min_epochs', type=int, default=8000, help='早停前最少訓練迴圈')
parser.add_argument('--no_early_stop', action='store_true', help='完全禁用早停')
parser.add_argument('--device', type=str, default=None, help='使用設備 (cuda 或 cpu)')
parser.add_argument('--log_csv', type=str, default='training_log.csv', help='損失記錄 CSV 檔路徑')
parser.add_argument('--use_tokenizer', action='store_true', help='使用 BPE 分詞器代替簡單分詞')
parser.add_argument('--vocab_size', type=int, default=2000, help='tokenizer 詞彙表大小 (BPE)')
parser.add_argument('--tokenizer_path', type=str, default='tokenizer.json', help='tokenizer 儲存/載入路徑')
parser.add_argument('--augment_factor', type=int, default=1, help='資料擴充倍數 (整數)')
parser.add_argument('--augment_swap', action='store_true', help='若擴充則隨機交換相鄰詞以添加噪音')
args = parser.parse_args()

print("=== 載入訓練資料 ===")

# 支援多個檔案或資料夾
input_paths = [args.input]
if args.extra_data:
    input_paths += args.extra_data

def load_texts(paths):
    pieces = []
    for p in paths:
        if os.path.isdir(p):
            for fname in os.listdir(p):
                if fname.endswith('.txt'):
                    fp = os.path.join(p, fname)
                    print(f"載入 {fp}")
                    pieces.append(open(fp, encoding='utf-8').read())
        elif os.path.isfile(p):
            print(f"載入 {p}")
            pieces.append(open(p, encoding='utf-8').read())
        else:
            print(f"警告：找不到路徑 {p}")
    return "\n".join(pieces)

text = load_texts(input_paths)

# 簡單資料擴充：如果指定了 augment_factor>1，就重複資料並視情況加入少量噪音
if args.augment_factor > 1:
    print(f"進行簡單資料擴充: factor={args.augment_factor}, swap_noise={args.augment_swap}")
    originals = text.split()
    aug_pieces = []
    for i in range(args.augment_factor - 1):
        if args.augment_swap:
            # 將相鄰兩詞隨機交換幾次
            words = originals.copy()
            import random
            for _ in range(len(words)//10):
                j = random.randint(0, len(words)-2)
                words[j], words[j+1] = words[j+1], words[j]
            aug_pieces.append(' '.join(words))
        else:
            aug_pieces.append(' '.join(originals))
    text = text + '\n' + '\n'.join(aug_pieces)
    print(f"擴充後總詞數: {len(text.split())}")


if args.use_tokenizer:
    # 訓練或載入 HuggingFace tokenizers 的 BPE 分詞器
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace

    if os.path.exists(args.tokenizer_path):
        print(f"載入現有 tokenizer: {args.tokenizer_path}")
        tokenizer = Tokenizer.from_file(args.tokenizer_path)
    else:
        print(f"訓練新 tokenizer（vocab_size={args.vocab_size}）…")
        tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(vocab_size=args.vocab_size, min_frequency=args.min_count, special_tokens=["<UNK>"])
        tokenizer.train([args.input], trainer)
        tokenizer.save(args.tokenizer_path)
        print(f"Tokenizer 保存到 {args.tokenizer_path}")

    # 把整個文字轉成 token id 序列
    encoded = tokenizer.encode(text)
    data_ids = encoded.ids
    stoi = tokenizer.get_vocab()
    # 依照 id 構建 itos 和詞彙表列表
    itos = {i: w for w, i in stoi.items()}
    vocab = [itos[i] for i in range(len(itos))]
    print(f"使用 tokenizer, 詞彙表大小: {len(vocab)}")
    # 可選：顯示 token 統計資料等
else:
    words = text.split()
    print(f"原始詞彙表大小: {len(set(words))} 個詞")
    print(f"訓練資料總詞數: {len(words)} 個詞")

    # ⭐ 過濾低頻詞
    # 改為使用命令列參數 min_count，可在訓練時調整過濾閾值
    word_counts = Counter(words)
    min_count = args.min_count  # 使用命令列參數

    # 如果 min_count <= 1，就不做 <UNK> 替換，保留所有詞
    UNK = '<UNK>'
    if min_count <= 1:
        vocab = sorted(set(words))
        filtered_words = words.copy()
        unk_count = 0
    else:
        vocab = sorted([w for w, count in word_counts.items() if count >= min_count])
        # 用 <UNK> 替換低頻詞
        if UNK not in vocab:
            vocab = [UNK] + vocab
        filtered_words = [w if word_counts[w] >= min_count else UNK for w in words]
        unk_count = filtered_words.count(UNK)

    print(f"過濾後詞彙表大小: {len(vocab)} 個詞")
    if min_count > 1:
        print(f"過濾掉的低頻詞: {len(set(words)) - len(vocab)} 個")
        print(f"<UNK> 出現次數: {unk_count} ({unk_count/len(filtered_words)*100:.1f}% of tokens)")
    else:
        print("min_count <= 1，未使用 <UNK>，保留全部詞彙。")

# 建立映射
if args.use_tokenizer:
    # 使用 tokenizer 時前面已經建立了 stoi/itos
    # data_ids 來自 encoder.ids
    data = torch.tensor(data_ids, dtype=torch.long)
else:
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
if args.use_tokenizer:
    avg = len(data) / len(vocab) if len(vocab) > 0 else 0
    print(f"平均每 token 出現次數: {avg:.1f} 次")
else:
    print(f"平均每詞出現次數: {len(filtered_words) / len(vocab):.1f} 次")


# 如果有指定預訓練權重，之後會在建立模型後載入

print("\n=== 設定超參數 ===")
# 把硬編碼參數改為使用 args，由 CLI 控制
block_size = args.block_size
batch_size = args.batch_size
if args.device is not None:
    device = args.device
else:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"使用設備: {device}")

if device == 'cuda':
    print(f"GPU 名稱: {torch.cuda.get_device_name(0)}")

print("\n=== 創建模型 ===")
model = MiniGPT(
    vocab_size=len(vocab),
    embed=args.embed,
    block_size=block_size,
    num_layers=args.num_layers,
    num_heads=args.num_heads,
    dropout=args.dropout
).to(device)  # 模型結構也可藉由 CLI 參數調整

# 如果使用者提供了預訓練 checkpoint，就載入
if args.init_from:
    if os.path.exists(args.init_from):
        print(f"從預訓練權重 {args.init_from} 初始化模型…")
        model.load_state_dict(torch.load(args.init_from))
    else:
        print(f"警告：指定的預訓練檔案 {args.init_from} 不存在，將從頭開始訓練")

total_params = sum(p.numel() for p in model.parameters())
print(f"模型參數量: {total_params:,} ({total_params/1e6:.2f}M)")

opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
# warmup 加上餘弦學習率調度器
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

# 自動混合精度，加速 GPU 訓練
# 只在 CUDA 上啟用 AMP，可以提升速度和降低顯存
use_amp = device.startswith('cuda')
# 更新為新的 AMP API
scaler = torch.amp.GradScaler(enabled=use_amp)

def get_batch(split='train'):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix]).to(device)
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]).to(device)
    return x, y

print("\n=== 開始訓練 ===")
num_epochs = args.epochs
print_interval = 500
eval_interval = 1000
eval_iters = 50

best_val_loss = float('inf')
patience = args.patience
# 若有預訓練模型，可能已經少訓練幾個 epoch，patience 計算不變
if args.no_early_stop:
    patience = float('inf')  # 永遠不觸發
patience_counter = 0
min_epochs = args.min_epochs

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

import csv

# 開啟 CSV 日誌記錄器
# 訓練過程會儲存到 CSV 檔，方便後續繪圖或分析
csv_file = open(args.log_csv, 'w', newline='', encoding='utf-8')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['epoch','train_loss','val_loss'])

for epoch in range(num_epochs):
    x, y = get_batch('train')
    opt.zero_grad()
    # 使用新的 autocast API；啟用時需指定 device_type
    with torch.amp.autocast(device_type='cuda' if use_amp else 'cpu', enabled=use_amp):
        out = model(x)
        out_reshaped = out.view(-1, len(vocab))
        y_reshaped = y.view(-1)
        loss = loss_fn(out_reshaped, y_reshaped)
    scaler.scale(loss).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt)
    scaler.update()
    scheduler.step()
    
    if epoch % eval_interval == 0:
        losses = estimate_loss()
        print(f"Epoch {epoch:5d} | Train Loss: {losses['train']:.4f} | Val Loss: {losses['val']:.4f}")
        
        csv_writer.writerow([epoch, losses['train'], losses['val']])
        csv_file.flush()
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            patience_counter = 0
            torch.save(model.state_dict(), "persona_best.pt")
            print(f"  ✓ 新的最佳模型！驗證損失: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            if not args.no_early_stop:
                if patience_counter >= patience and epoch >= min_epochs:
                    print(f"\n早停觸發！驗證損失已 {patience} 次未改善")
                    print(f"最佳驗證損失: {best_val_loss:.4f}")
                    break
                elif patience_counter >= patience:
                    print(f"  (驗證損失未改善 {patience_counter} 次，但未達最小訓練輪數 {min_epochs})")
            # 若已禁用早停，就繼續訓練到指定 epoch
    
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

csv_file.close()

print("✓ 最佳模型已儲存到 persona.pt")
print("✓ 詞彙表已儲存到 vocab.pkl")
print(f"✓ 最佳驗證損失: {best_val_loss:.4f}")
print("訓練完成！")