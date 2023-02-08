import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from steerDS import SteerDataSet
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import numpy as np

# DIR = "/home/will/RVSS_Need4Speed/on_laptop/data/"
DIR = "data/"

# functions to show an image

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(2704, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

if __name__ == '__main__':
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 4

    # import train dataset
    ds_train = SteerDataSet(DIR,".jpg",transform)
    ds_test = SteerDataSet(DIR,".jpg",transform)

    # select train images
    ds_train.filenames = ds_train.filenames[0:1500]
    print("The dataset contains %d images " % len(ds_train))
    trainloader = DataLoader(ds_train,batch_size=4,shuffle=True)

    # trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
    #                                         download=True, transform=transform)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
    #                                           shuffle=True, num_workers=2)

    # select test images
    ds_test.filenames = ds_test.filenames[1500:]
    print("The dataset contains %d images " % len(ds_test))
    testloader = DataLoader(ds_test,batch_size=len(ds_test),shuffle=True)

    # testset = torchvision.datasets.CIFAR10(root='./data', train=False,
    #                                        download=True, transform=transform)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
    #                                          shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    if False:
        net = Net()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

        for epoch in range(10):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs = data['image']
                labels = data['steering']
                labels = torch.tensor(labels)
                # print(inputs)
                # print(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if (i+1) % 100 == 0:    # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0
        print('Finished Training')

        # save
        PATH = './cifar_net.pth'
        torch.save(net.state_dict(), PATH)

    # import the model
    net = Net()
    net.load_state_dict(torch.load('model_1.pth'))

    for i, data in enumerate(testloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs = data['image']
        labels = data['steering']
        labels = torch.tensor(labels)

        # print(inputs)
        print(labels)

        outputs = net(inputs)
        print(outputs)

        _, predicted = torch.max(outputs, 1)
        print(predicted)

        print(torch.sum(predicted==labels)/len(outputs))
