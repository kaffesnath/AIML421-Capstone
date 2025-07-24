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
import time

classes = ['cherry', 'strawberry', 'tomato']
DIM = 128

if torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f'Using {device} device')


#original had 1,495 each
#after is 1418 cherry, 1298 strawberry, 1333 tomato

#display distribution of classes EDA 
def display_distribution(data):
    count = [0, 0, 0]
    for _, label in data:
        count[label] += 1
    plt.bar(classes, count)
    plt.show()
    print(count)

def display_first(data):
    imag = data[0][0].numpy()
    imag = np.transpose(imag, (1, 2, 0))
    print(imag.shape)
    plt.imshow(imag)
    plt.show()

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

    #display_distribution(train_data)
    #display_first(train_data)

    #initialise MLP
    mlp = nn.Sequential(
        nn.Linear(DIM*DIM*3, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 3)
    )

    #move to gpu
    mlp.to(device)

    #initialise loss function
    loss_fn = nn.CrossEntropyLoss()

    #initialise optimiser
    optimiser = Adam(mlp.parameters(), lr=0.001)

    #storage for results
    losses = []
    accuracies = []

    #train MLP
    for epoch in range(15):
        loss = 0
        acc = 0
        total = 0
        for data, label in train_loader:
            #move to gpu
            data, label = data.to(device), label.to(device)

            #flatten data
            data = data.view(-1, DIM*DIM*3)
            optimiser.zero_grad()
            output = mlp(data)
            loss = loss_fn(output, label)
            loss.backward()
            optimiser.step()
            #add to total loss and accuracy
            loss += loss.item()

            _, predicted = torch.max(output, 1)
            acc += (predicted == label).sum().item()
            #increment total
            total += label.size(0)
        loss_val = loss/len(train_loader)
        acc_val = acc/total
        print(f'Epoch {epoch+1}, Loss: {loss_val}, Accuracy: {acc_val}')
        losses.append(float(loss_val))
        accuracies.append(float(acc_val))

        
    #save MLP
    save(mlp, 'mlp.pth')

    #test MLP
    test(mlp, train_loader)
    test(mlp, test_loader)
    plot_results(losses, accuracies)

def test(model, test_loader):
    model.eval()

    correct = 0
    total = 0
    for data, label in test_loader:
        data, label = data.to(device), label.to(device)
        data = data.view(-1, DIM*DIM*3)
        output = model(data)
        _, predicted = torch.max(output, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()
    print(f'Accuracy: {correct/total}')
    print(f'Correct: {correct}, Total: {total}')

#plots training results
def plot_results(losses, accuracies):
    plt.plot(losses, label='Loss')
    plt.plot(accuracies, label='Accuracy')
    plt.legend()
    plt.show()
    
if __name__ == '__main__':
    start = time.time()
    main(0)
    end = time.time()
    print(f'Time taken: {end-start}')








