#!/usr/bin/env python3
import time
import click
import math
import sys
sys.path.append("..")
import cv2
import numpy as np
import penguinPi as ppi
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# stop the robot 
ppi.set_velocity(0,0)
print("initialise camera")
camera = ppi.VideoStreamWidget('http://localhost:8080/camera/get')

#INITIALISE NETWORK HERE
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
net = Net()

#LOAD NETWORK WEIGHTS HERE
net.load_state_dict(torch.load('model_3.pth'))


transform = transforms.Compose(
    [transforms.ToTensor(), 
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#countdown before beginning
print("Get ready...")
print("10")
time.sleep(1)
print("9")
print("SIKE!")
print("GO!")

try:
    angle = 0
    while True:
        # get an image from the the robot
        image = camera.frame

        # apply any image transformations
        image = transform(cv2.resize(image[80:,:], dsize=(32,32), interpolation=cv2.INTER_CUBIC))

        # pass image through network to get a prediction for the steering angle
        steering = net(image)
        _, predicted = torch.max(steering, 0)

        print(predicted)

        if predicted == 0:	    # left
            angle = -0.2
        elif predicted == 1:	# right
            angle = 0.2
        else:			        # straight
            angle = 0

        angle = np.clip(angle, -0.5, 0.5)
        Kd = 30 #base wheel speeds, increase to go faster, decrease to go slower
        Ka = 30 #how fast to turn when given an angle
        left  = int(Kd + Ka*angle)
        right = int(Kd - Ka*angle)
        
        print(left, right)
        ppi.set_velocity(left,right) 
        
        
except KeyboardInterrupt:    
    ppi.set_velocity(0,0)