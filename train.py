import torch
from model import MiniGPT

# 讀取資料
text = open('data/input.txt').read()
chars = sorted(set(text))
stoi = {c:i for i,c in enumerate(chars)}
itos = {i:c for c,i in stoi.items()}

data = torch.tensor([stoi[c] for c in text],dtype=torch.long)

model = MiniGPT(len(chars))
# Adam 算法
opt = torch.optim.Adam(model.parameters(), 1e-3)
# 損失函數
loss_fn = torch.nn.CrossEntropyLoss()

for epoch in range(30000):
    x = data [:-1]
    y = data [1:]

    out = model(x)
    loss = loss_fn(out.view(-1,len(chars)),y)

    # 梯度清零
    opt.zero_grad()

    loss.backward()
    opt.step()

    if epoch % 500 == 0:
        print('loss : ',loss.item())

torch.save(model.state_dict(),"persona.pt")
print("Training finished")