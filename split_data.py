import random
import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms as tf
from torch.utils.data import random_split
import os
seed = 0
DIM = 300
test_dir = 'testdata'
train_dir = 'traindata'
classes = ['cherry', 'strawberry', 'tomato']

def store_data(data, path):
    # Loop through the test dataset and save each image
    for idx, (image, label) in enumerate(data):
        label = classes[label]
        
        # Define the path where the image will be saved
        label_dir = os.path.join(path, str(label))
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)
        sub_dir = label_dir.split('/')[-1]
        image_path = os.path.join(label_dir, f'{sub_dir}_{idx}.png')
        
        # Save the image
        image.save(image_path)

random.seed(seed)
torch.manual_seed(seed)

#loads base data with no transformations
train_data = ImageFolder('train_data')
#perform tts of 0.8/0.2
train_size = int(0.8*len(train_data))
test_size = len(train_data) - train_size
train_data, test_data = random_split(train_data,[train_size, test_size], generator=torch.Generator().manual_seed(seed))
store_data(test_data, test_dir)
store_data(train_data, train_dir)