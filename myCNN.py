import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torchvision
from torchvision import datasets, models, transforms
import torch.optim as optim
from torch.optim import lr_scheduler

import numpy as np

import matplotlib.pyplot as plt

import time
import copy

# Device configuration
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# Hyper parameters
num_epochs = 30
num_classes = 10
batch_size = 10
learning_rate = 0.001

data_dir = './bulloying_dataset/'
train_data_dir = './bulloying_dataset/train_data'
val_data_dir = './bulloying_dataset/val_data'
# data_tranforms = {
#     'train_data'    : transforms.Compose([
#         transforms.RandomResizedCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]),
#     'val_data'      : transforms.Compose([
#         transforms.Resize(256),
#         transforms.RandomResizedCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])
# }
# my_image_datasets = {
#     x: datasets.ImageFolder(os.path.join(data_dir, x), data_tranforms[x])
#     for x in ['train_data', 'val_data']
# }

# my_dataloaders = {
#     x: torch.utils.data.DataLoader(my_image_datasets[x], batch_size=4, shuffle=True, num_workers=4) 
#     for x in ['train_data', 'val_data']
# }

# dataset_size = {x: len(my_image_datasets[x]) for x in ['train_data', 'val_data']}

# Traning Data
train_data_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

train_image_dataset = datasets.ImageFolder(train_data_dir, train_data_transforms)

train_dataloaders = torch.utils.data.DataLoader(train_image_dataset, batch_size=batch_size, shuffle=True)

train_datasize = len(train_image_dataset)

# Evaluation Data
val_data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
val_image_dataset = datasets.ImageFolder(val_data_dir, val_data_transforms)

val_dataloaders = torch.utils.data.DataLoader(val_image_dataset, batch_size=batch_size, shuffle=False)

val_datasize = len(val_image_dataset)

# Building Convolutional Neural Network
class MyCNN(nn.Module):
    """Some Information about MyCNN"""
    #regard input image size as 224X224
    def __init__(self, num_classes = 10):
        super(MyCNN, self).__init__()
        # Conv1
        self.layer1 = nn.Sequential(
            # if stride size =1 , padding size = (kernel_size -1)/2
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.dropout = nn.Dropout(p=0.5)
        # Conv2
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.dropout = nn.Dropout(p=0.5)
        # Full Connected layer
        self.fc1 = nn.Linear(32*56*56, 10)
        self.fc_bn = nn.BatchNorm1d(10)
        #self.fc2 = nn.Linear(1024, 10)
        self.initialize_weights()

    def initialize_weights(self):
        # classname = m.__class__.__name__
        # if classname.find('Conv') != -1:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Initialize weight by using xavier
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                elif isinstance(m, nn.Linear): 
                    nn.init.normal_(m.weight, 0, 0.01) 

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(-1, 32*56*56)
        out = self.fc1(out)
        #out = self.fc2(out)
        return out

cnn = MyCNN() # generalize an instance
#cnn.apply(initialize_weights) # apply weight initialize

if torch.cuda.device_count() > 1:
    cnn = nn.DataParallel(cnn()).cuda()
else:
    cnn = cnn.cuda()

# Loss and Optimizer
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr = learning_rate)

# Training the Model
cnn.train()
total_train_step = len(train_dataloaders)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_dataloaders):
        images = images.cuda()
        labels = labels.cuda()

        # Forward pass
        outputs = cnn(images)
        loss = loss_function(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1)%100:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch+1, num_epochs, i+1, total_train_step, loss.item()))

# Test the model
cnn.eval()
with torch.no_grad():
    correct = 0
    total = 0

    for images, labels in val_dataloaders:
        images = images.cuda()
        labels = labels.cuda()
        outputs = cnn(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct +=(predicted == labels).sum().item()
    print('Test Accuracy of the model on test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(cnn.state_dict(), 'mycnn.ckpt')
