from torchvision import transforms as tf
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import random
import numpy as np
import torch
import copy
from torch import nn
from torch.optim import Adam, Adagrad, SGD, RMSprop
from torchvision.models import resnet18
from torch import save
from torch.utils.data import random_split
import torch.nn.functional as func
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time

classes = ['cherry', 'strawberry', 'tomato']
DIM = 256
BCH = 16

if torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f'Using {device} device')

def main(seed):
    losscount = 0
    prev = 0
    best_model = None
    best = 0
    random.seed(seed)
    torch.manual_seed(seed)
    #load data from train_data
    train_tf = tf.Compose([ tf.Resize((DIM, DIM)), 
        tf.RandomHorizontalFlip(),
        tf.RandomVerticalFlip(),
        tf.RandomRotation(30),
        tf.ToTensor(), 
        tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    test_tf = tf.Compose([ tf.Resize((DIM, DIM)), 
        tf.ToTensor(), 
        tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    train_data = ImageFolder('traindata', transform=train_tf)
    test_data = ImageFolder('testdata', transform=test_tf)

    #put both sets into loaders
    train_loader = DataLoader(train_data, batch_size=BCH, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BCH, shuffle=False)

    #load model
    #model = CNN().to(device)
    model = resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 3)
    model.to(device)

    #define loss function
    criterion = nn.CrossEntropyLoss()

    #define optimiser
    #optimiser = Adam(model.parameters(), lr=0.001)
    #optimiser = SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimiser = Adagrad(model.parameters(), lr=0.001)
    #optimiser = RMSprop(model.parameters(), lr=0.001)

    #define scheduler
    scheduler = ReduceLROnPlateau(optimiser, 'min')

    #store accuracies
    train_accs = []
    test_accs = []

    #train model
    epochs = 15
    start = time.time()

    losses = []
    accuracies = []

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
        test_acc, val_loss = test(model, test_loader, criterion)

        scheduler.step(val_loss)

        losses.append(float(loss / len(train_loader)))
        accuracies.append(float(acc / total))

        print(f'Epoch {epoch + 1}/{epochs} - Loss: {loss:.4f} - Accuracy: {acc / total:.4f} - Test Loss: {val_loss:.4f} - Test Accuracy: {test_acc:.4f}')
        #store accuracies
        train_accs.append(acc / total)
        test_accs.append(test_acc)
        if test_acc > best:
            best = test_acc
            best_model = copy.deepcopy(model)
        else:
            if test_acc < prev:
                losscount += 1
            else:
                losscount = 0
            if losscount == 3:
                break
        prev = test_acc

    #save model
    save(best_model, 'model.pth')
    print("Computation time: ", time.time() - start)

    #plot accuracies
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(test_accs, label='Test Accuracy')
    plt.legend()
    plt.show()

    plot_results(losses, accuracies)

def test(model, test_loader, criterion):
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

def plot_results(losses, accuracies):
    plt.plot(losses, label='Loss')
    plt.plot(accuracies, label='Accuracy')
    plt.legend()
    plt.show()

class CNN(nn.Module):
    def __init__(self, num_classes=3):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32) 
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)  
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)  
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 32 * 32, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(func.relu(self.bn1(self.conv1(x)))) 
        x = self.pool(func.relu(self.bn2(self.conv2(x))))
        x = self.pool(func.relu(self.bn3(self.conv3(x)))) 
        x = x.view(-1, 128 * 32 * 32)  
        x = func.relu(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == '__main__':
    main(0)
