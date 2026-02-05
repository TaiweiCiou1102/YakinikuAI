import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import random

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.dropout = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(9216,128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x,2)
        x = torch.flatten(x,1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SimpleCNN().to(device)

model.load_state_dict(torch.load("mnist_cnn.pth", map_location=device, weights_only=True))

model.eval()
print("模型載入成功!")

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28,28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

from PIL import Image

PATH = './source/hand_writing.png'
img = Image.open(PATH)
img = img.convert("RGB")
img.save("./source/hand_writing.jpg", "JPEG")

image = Image.open("./source/hand_writing.jpg")
image = transform(image)
label = 5

input_tensor = image.unsqueeze(0).to(device)

with torch.no_grad():
    output = model(input_tensor)
    prediction = output.argmax(dim=1).item()

# 6. 顯示結果
plt.imshow(image.squeeze(), cmap='gray') # squeeze: 把 (1, 28, 28) 變成 (28, 28) 以便畫圖
plt.title(f"True: {label} | AI Predict: {prediction}")
plt.axis('off')
plt.show()

print(f"正確答案是: {label}")
print(f"AI 猜測是: {prediction}")

if label == prediction:
    print("✅ 答對了！")
else:
    print("❌ 答錯了...看來還需要多讀點書 (Train more epochs)")
# %%
