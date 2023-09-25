import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # 必須在`import torch`語句之前設置才能生效
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm  
from model import Net
from data import train_dataset

device = torch.device('cuda')
batch_size = 64

# 初始化DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 初始化
model = Net()
model = model.to(device)  # 使用第一個GPU
optimizer = optim.SGD(model.parameters(), lr=0.1)


for i, (inputs, labels) in tqdm(enumerate(train_loader), total=len(train_loader), desc="Training"):
    # forward
    inputs = inputs.to(device)
    labels = labels.to(device)
    outputs = model(inputs)
    if isinstance(outputs, tuple):
        outputs = outputs[0]
    loss = nn.CrossEntropyLoss()(outputs, labels)
    
    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if i % 10 == 0:
        print(f"Step {i}, Loss: {loss.item()}")
