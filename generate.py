"""
MiniGPT 文字生成腳本（過濾詞彙表版本）
"""

import torch
from model import MiniGPT
import pickle
import os

import argparse

# 生成時的命令列選項
# 加入 argparse 讓生成時也能調整長度、溫度和 top-k
parser = argparse.ArgumentParser(description='Generate text with a trained MiniGPT model')
parser.add_argument('--model', type=str, default='persona.pt', help='模型權重檔路徑')
parser.add_argument('--vocab', type=str, default='vocab.pkl', help='詞彙表 pickle 檔路徑')
parser.add_argument('--length', type=int, default=200, help='最大生成 token 數')
parser.add_argument('--temperature', type=float, default=0.8, help='採樣溫度')
parser.add_argument('--top_k', type=int, default=50, help='採樣時保留前 k 個候選')
parser.add_argument('--repeat_penalty', type=float, default=1.3, help='重複詞懲罰強度（>1 懲罰越重，建議 1.1~1.5）')
parser.add_argument('--no_repeat_ngram', type=int, default=4, help='禁止重複出現的 n-gram 長度（0 為關閉）')
parser.add_argument('--use_tokenizer', action='store_true', help='使用 tokenizer 進行編碼/解碼')
parser.add_argument('--tokenizer_path', type=str, default='tokenizer.json', help='tokenizer 檔案路徑')
args = parser.parse_args()

# block_size 在訓練時固定為 64，同樣使用全域變數
block_size = 64

def load_resources():
    print("=== 載入詞彙表 ===")
    # 載入訓練時儲存的詞彙表
    with open(args.vocab, 'rb') as f:
        vocab_data = pickle.load(f)
        vocab = vocab_data['vocab']
        stoi = vocab_data['stoi']
        itos = vocab_data['itos']

    print(f"詞彙表大小: {len(vocab)} 個詞")

    # 如果使用 tokenizer，就載入它
    if args.use_tokenizer:
        from tokenizers import Tokenizer
        if not os.path.exists(args.tokenizer_path):
            print(f"錯誤：找不到 tokenizer 檔案 {args.tokenizer_path}。\n" \
                  "請先用 train.py 生成或提供有效 tokenizer，或移除 --use_tokenizer。")
            exit(1)
        tokenizer = Tokenizer.from_file(args.tokenizer_path)
        # 確保 stoi/itos 與 tokenizer 一致
        stoi = tokenizer.get_vocab()
        itos = {i: w for w, i in stoi.items()}
        vocab = list(stoi.keys())
        print(f"使用 tokenizer, vocab_size={len(vocab)}")
    else:
        tokenizer = None

    print("\n=== 載入模型 ===")
    vocab_size = len(vocab)
    # block_size 必須與訓練時相同（checkpoint 使用 64）
    block_size = 64
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"使用設備: {device}")

    model = MiniGPT(
        vocab_size=vocab_size,
        embed=256,
        block_size=block_size,
        num_layers=6,
        num_heads=4,
        dropout=0.4
    ).to(device)

    model.load_state_dict(torch.load("persona.pt"))
    model.eval()

    print("✓ 模型載入成功")
    return vocab, stoi, itos, tokenizer, model, device

# 執行初始化動作，設定全域變數
vocab, stoi, itos, tokenizer, model, device = load_resources()

def apply_repeat_penalty(logits, generated_ids, penalty):
    """對已出現過的 token 降低機率"""
    for token_id in set(generated_ids):
        if logits[0, token_id] > 0:
            logits[0, token_id] /= penalty
        else:
            logits[0, token_id] *= penalty
    return logits

def apply_no_repeat_ngram(logits, generated_ids, ngram_size):
    """禁止已出現過的 n-gram 再次出現"""
    if ngram_size <= 0 or len(generated_ids) < ngram_size - 1:
        return logits
    # 取最近 ngram_size-1 個 token 作為前綴
    prefix = tuple(generated_ids[-(ngram_size - 1):])
    # 找出訓練序列中這個前綴後面接過哪些 token
    banned = set()
    for i in range(len(generated_ids) - ngram_size + 1):
        if tuple(generated_ids[i:i + ngram_size - 1]) == prefix:
            banned.add(generated_ids[i + ngram_size - 1])
    for token_id in banned:
        logits[0, token_id] = -float('inf')
    return logits

def generate(start="I", length=None, temperature=None, top_k=None):
    """生成文字
    這個函式現在會使用 CLI 參數作為預設值，方便在 prompt 時修改行為
    """
    if length is None:
        length = args.length
    if temperature is None:
        temperature = args.temperature
    if top_k is None:
        top_k = args.top_k

    # 將提示編碼為索引
    # encode prompt 及時記錄 prompt 長度，稍後生成時用於保留
    if args.use_tokenizer:
        encoded = tokenizer.encode(start)
        indices = encoded.ids
        if len(indices) == 0:
            indices = [stoi['<UNK>']] if '<UNK>' in stoi else [0]
        prompt_len = len(indices)
        x = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)
    else:
        start_words = start.split()
        # 如果詞彙表有 <UNK>，把不認識的詞都映射成它，否則丟棄
        if '<UNK>' in stoi:
            indices = [stoi.get(w, stoi['<UNK>']) for w in start_words]
        else:
            filtered = [w for w in start_words if w in stoi]
            if not filtered:
                filtered = [vocab[0]]
                print(f"輸入的詞不在詞彙表中，使用預設起始詞: {vocab[0]}")
            indices = [stoi[w] for w in filtered]
        prompt_len = len(indices)
        x = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)
    
    for _ in range(length):
        # 只保留最近 block_size 個 token 以避免溢位
        if x.size(1) > block_size:
            x = x[:, -block_size:]

        with torch.no_grad():
            out = model(x)
        
        logits = out[:, -1, :].clone()

        # 套用重複懲罰
        generated_ids = x[0].tolist()
        if args.repeat_penalty != 1.0:
            logits = apply_repeat_penalty(logits, generated_ids, args.repeat_penalty)
        if args.no_repeat_ngram > 0:
            logits = apply_no_repeat_ngram(logits, generated_ids, args.no_repeat_ngram)

        scaled_logits = logits / temperature
        probs = torch.softmax(scaled_logits, dim=-1)
        
        if top_k is not None and top_k > 0:
            k = top_k
            top_probs, top_indices = torch.topk(probs, k, dim=-1)
            top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True)
            sampled_idx = torch.multinomial(top_probs, 1)
            next_token = top_indices[0, sampled_idx].item()
        else:
            next_token = torch.multinomial(probs, 1).item()

        next_token_tensor = torch.tensor([[next_token]], device=device)
        x = torch.cat([x, next_token_tensor], dim=-1)
    
    if args.use_tokenizer:
        # 使用 tokenizer 將整個序列解碼為字串；prompt 已包含其中
        result = tokenizer.decode(x[0].tolist())
        return result
    else:
        # 直接用原始 prompt 當開頭，避免 block_size 截斷造成 index 跑掉
        all_words = [itos[i.item()] for i in x[0]]
        continuation_words = all_words[prompt_len:]
        continuation_words = [w for w in continuation_words if w != '<UNK>']
        result = (start + ' ' + ' '.join(continuation_words)).strip()
        return result

print("\n=== 開始生成 ===")
print("輸入提示詞來生成文字（輸入 'quit' 退出）")
print("-" * 50)

while True:
    prompt = input("\nYou: ")
    
    if prompt.lower() in ['quit', 'exit', 'q']:
        print("再見！")
        break
    
    try:
        generated_text = generate(prompt, length=200)
        print(f"\nAI: {generated_text}")
    except Exception as e:
        print(f"生成時發生錯誤: {e}")