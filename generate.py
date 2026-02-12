import torch
from model import MiniGPT

# 讀取資料
text = open('data/input.txt').read()
chars = sorted(set(text))
stoi = {c:i for i,c in enumerate(chars)}
itos = {i:c for c,i in stoi.items()}

model = MiniGPT(len(chars))
model.load_state_dict(torch.load("persona.pt"))

def generate(start="I",length=200):
    x = torch.tensor([stoi[c] for c in start],dtype=torch.long)
    for _ in range(length):
        out = model(x)[-1]
        p = torch.softmax(out,0)
        c = torch.multinomial(p,1).item()
        x = torch.cat([x,torch.tensor([c])])
    return ''.join([itos[i.item()] for i in x])

while True:
    prompt = input("You: ")
    print("AI:",generate(prompt))