from torchvision import transforms as tf
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import random
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch import save
from torch.utils.data import random_split
import torch.nn.functional as func
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time

classes = ['cherry', 'strawberry', 'tomato']
DIM = 256

if torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f'Using {device} device')

def main(seed):
    random.seed(seed)
    #load data from train_data
    train_tf = tf.Compose([ tf.Resize((DIM, DIM)), 
        tf.RandomHorizontalFlip(),
        tf.RandomVerticalFlip(),
        tf.RandomRotation(15),
        tf.ToTensor(), 
        tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]) #resnet normalisation
    
    test_tf = tf.Compose([ tf.Resize((DIM, DIM)), 
        tf.ToTensor(), 
        tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]) 

    #loads base data with no transformations
    train_data = ImageFolder('train_data')
    #perform tts of 0.8/0.2
    train_size = int(0.8*len(train_data))
    test_size = len(train_data) - train_size
    train_data, test_data = random_split(train_data,[train_size, test_size], generator=torch.Generator().manual_seed(seed))

    #apply transformations
    train_data.dataset.transform = train_tf
    test_data.dataset.transform = test_tf

    #put both sets into loaders
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=True)

    #load model
    model = CNN().to(device)

    #define loss function
    criterion = nn.CrossEntropyLoss()

    #define optimiser
    optimiser = Adam(model.parameters(), lr=0.001)

    #define scheduler
    scheduler = ReduceLROnPlateau(optimiser, 'min')

    #train model
    epochs = 10
    start = time.time()

    for epoch in range(epochs):
        loss = 0
        acc = 0
        total = 0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimiser.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimiser.step()

            loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            acc += (predicted == labels).sum().item()
        test_acc, val_loss = test(model, test_loader)
        scheduler.step(val_loss)

        print(f'Epoch {epoch + 1}/{epochs} - Loss: {loss:.4f} - Accuracy: {acc / total:.4f} - Test Accuracy: {test_acc:.4f}')

    #save model
    save(model, 'model.pth')
    print("Computation time: ", time.time() - start)

def test(model, test_loader, criterion=nn.CrossEntropyLoss()):
    model.eval()

    correct = 0
    total = 0
    val_loss = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            val_loss += criterion(outputs, labels)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    model.train()
    return correct / total, val_loss / len(test_loader)

class CNN(nn.Module):
    def __init__(self, num_classes=3):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(64 * (DIM // 4) * (DIM // 4), 128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.pool(func.relu(self.conv1(x)))  # Conv layer followed by ReLU and pooling
        x = self.dropout(x)
        x = self.pool(func.relu(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(-1, 64 * (DIM // 4) * (DIM // 4))  # Flattening for FC layer
        x = func.relu(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == '__main__':
    main(0)
