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
        
        logits = out[:, -1, :]
        
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
        generated_words = [itos[i.item()] for i in x[0]]
        # 將 prompt 詞與續寫詞分離
        prefix = generated_words[:prompt_len]
        continuation_words = generated_words[prompt_len:]
        # 只在續寫部分移除 <UNK>
        continuation_words = [w for w in continuation_words if w != '<UNK>']
        if continuation_words:
            result = ' '.join(prefix + continuation_words)
        else:
            result = ' '.join(prefix)
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