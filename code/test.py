from torchvision import transforms as tf
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import torch
from torch import load
from train import CNN
from warnings import filterwarnings

filterwarnings('ignore')

if torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f'Using {device} device')

classes = ['cherry', 'strawberry', 'tomato']

DIM = 256
BCH = 16

test_tf = tf.Compose([ tf.Resize((DIM, DIM)),
    tf.ToTensor(),
    tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

test_data = ImageFolder('testdata', transform=test_tf)
test_loader = DataLoader(test_data, batch_size=BCH, shuffle=False)

model = load('model.pth')
model.eval()
model.to(device)

correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the test images: {100 * correct / total}%')