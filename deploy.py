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
        self.fc1 = nn.Linear(1040, 120)
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
net.load_state_dict(torch.load('model_7.pth'))


transform = transforms.Compose(
    [transforms.ToTensor(), 
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

print("GO!")

actions = ['left','right','straight']
prev = [0, 1, 2, 0, 1]
counter = 0

try:
    angle = 0
    while True:
        counter += 1
        # get an image from the the robot
        image = camera.frame

        if counter == 50:
            cv2.imwrite('before2.jpg', image)
            cv2.imwrite('after2.jpg', cv2.resize(image[80:,:], dsize=(64,32), interpolation=cv2.INTER_CUBIC))

        # apply any image transformations
        image = transform(cv2.resize(image[80:,:], dsize=(64,32), interpolation=cv2.INTER_CUBIC))
        

        # pass image through network to get a prediction for the steering angle
        steering = net(image)
        _, predicted = torch.max(steering, 0)
        print(actions[predicted])

        # update previous readings
        prev.insert(0, predicted)
        prev.pop()

        # compute majority
        left = 0
        right = 0
        for i in range(len(prev)):
            if prev[i] == 0:
                left += 1
            elif prev[i] == 1:
                right += 1

        # gains
        Kd = 20 #base wheel speeds, increase to go faster, decrease to go slower
        Ka = 20 #how fast to turn when given an angle

        # update turn angle 
        if predicted == 0:
            goal = -0.5
            Kd = 20
        elif predicted == 1:
            goal = 0.5
            Kd = 20
        else:
            goal = 0

        K = 0.3
        angle += K * (goal - angle)

        # update motor speeds
        left  = int(Kd + Ka*angle)
        right = int(Kd - Ka*angle)
        print(str(left) + ' ' + str(right))
        ppi.set_velocity(left,right)
        
        
except KeyboardInterrupt:    
    ppi.set_velocity(0,0)