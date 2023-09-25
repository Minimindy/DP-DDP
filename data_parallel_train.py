import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import Net
from data import train_dataset

device = torch.device('cuda')
batch_size = 64
NUM_EPOCHS = 10

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.1)
model = nn.DataParallel(model)

# Loss function
loss_function = torch.nn.CrossEntropyLoss()

# tqdm 
for epoch in range(NUM_EPOCHS):  
    running_loss = 0.0
    for i, (inputs, labels) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Training Epoch {epoch + 1}"):
        # forward
        inputs, labels = inputs.to(device), labels.to(device)

        # Predict and compute the loss
        outputs = model(inputs)
        if isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs

        loss = loss_function(logits, labels)

        # backward and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print or log the loss
        running_loss += loss.item()
        if i % 10 == 9:  # print every 10 mini-batches
            print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 10:.3f}")
            running_loss = 0.0
