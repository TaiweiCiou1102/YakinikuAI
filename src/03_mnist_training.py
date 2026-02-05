import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用裝置 : {device}")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081))
])

train_dataset = datasets.MNIST('../data',train = True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1,32,kernel_size=3)
        self.conv2 = nn.Conv2d(32,64, kernel_size=3)
        self.dropout = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(9216,128)
        self.fc2 = nn.Linear(128,10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x,2)
        x = self.dropout(x)
        x = torch.flatten(x,1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x 

model = SimpleCNN().to(device)

print("模型架構建立完成!")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

num_epochs = 1

print("開始訓練")
model.train()

total_step = len(train_loader)
# for i, (images, labels) in enumerate(train_loader):
#     images = images.to(device)
#     labels = labels.to(device)

#     outputs = model(images)
#     loss = criterion(outputs, labels)

#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

#     if (i+1) % 100 == 0: 
#         print(f'Step [{i+1}/{total_step}], Loss: {loss.item():.4f}')

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # A. Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # B. Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}')

print("訓練完成！模型已學會看數字了。")

# 6. 保存模型 (存成 .pth 檔)
torch.save(model.state_dict(), "mnist_cnn.pth")
print("模型已儲存為 mnist_cnn.pth")