import os
import argparse
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from model import Net
from data import train_dataset, test_dataset

# 從環境變數獲取 LOCAL_RANK
local_rank = int(os.environ['LOCAL_RANK'])

# 初始化分散式環境
torch.cuda.set_device(local_rank)
device = torch.device('cuda', local_rank)
torch.distributed.init_process_group(backend='nccl')

# 固定隨機種子
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

batch_size = 64
NUM_EPOCHS = 5

# 初始化模型，損失函數，和優化器
model = Net()
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 配置分散式數據採樣器和數據加載器
train_sampler = DistributedSampler(train_dataset)
train_loader = torch.utils.data.DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)

# 設置 DistributedDataParallel
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

# 訓練模型
for epoch in range(NUM_EPOCHS):  # 添加了 NUM_EPOCHS，表示訓練的輪數
    running_loss = 0.0
    for i, (inputs, labels) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Training Epoch {epoch + 1}"):
        # forward
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        if isinstance(outputs, tuple):
            outputs = outputs[0]  
        loss = criterion(outputs, labels)
        
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # loss
        running_loss += loss.item()
        if local_rank == 0 and i % 5 == 4:  # 每5個印一次
            print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 5:.3f}")
            running_loss = 0.0
