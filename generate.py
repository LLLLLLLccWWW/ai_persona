"""
MiniGPT 文字生成腳本（過濾詞彙表版本）
"""

import torch
from model import MiniGPT
import pickle

print("=== 載入詞彙表 ===")

# 載入訓練時儲存的詞彙表
with open('vocab.pkl', 'rb') as f:
    vocab_data = pickle.load(f)
    vocab = vocab_data['vocab']
    stoi = vocab_data['stoi']
    itos = vocab_data['itos']

print(f"詞彙表大小: {len(vocab)} 個詞")

print("\n=== 載入模型 ===")

vocab_size = len(vocab)
block_size = 256
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"使用設備: {device}")

model = MiniGPT(
    vocab_size=vocab_size,
    embed=128,
    block_size=block_size
).to(device)

model.load_state_dict(torch.load("persona.pt"))
model.eval()

print("✓ 模型載入成功")

def generate(start="I", length=200):
    """生成文字"""
    start_words = start.split()
    
    # 過濾掉不在詞彙表中的詞
    start_words = [w for w in start_words if w in stoi]
    
    # 如果所有詞都不在詞彙表中，使用詞彙表的第一個詞
    if not start_words:
        start_words = [vocab[0]]
        print(f"輸入的詞不在詞彙表中，使用預設起始詞: {vocab[0]}")
    
    indices = [stoi[w] for w in start_words]
    x = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)
    
    for _ in range(length):
        with torch.no_grad():
            out = model(x)
        
        logits = out[:, -1, :]
        
        temperature = 0.8
        scaled_logits = logits / temperature
        probs = torch.softmax(scaled_logits, dim=-1)
        
        k = 50
        top_probs, top_indices = torch.topk(probs, k, dim=-1)
        top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True)
        
        sampled_idx = torch.multinomial(top_probs, 1)
        next_token = top_indices[0, sampled_idx].item()
        
        next_token_tensor = torch.tensor([[next_token]], device=device)
        x = torch.cat([x, next_token_tensor], dim=-1)
    
    generated_words = [itos[i.item()] for i in x[0]]
    result = ' '.join(generated_words)
    
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