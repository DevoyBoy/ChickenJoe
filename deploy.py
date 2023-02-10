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

# gains
Ka = 20         # how fast to turn when given an angle
Kn = 1          # how fast it gets to goal angle
maxTurn = 0.5   # how sharp to take the corners

# stop the robot 
ppi.set_velocity(0,0)
print("initialise camera")
camera = ppi.VideoStreamWidget('http://localhost:8080/camera/get')

# INITIALISE NETWORK
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

# LOAD NETWORK WEIGHTS
net.load_state_dict(torch.load('model_7.pth'))

transform = transforms.Compose(
    [transforms.ToTensor(), 
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

actions = ['left','right','straight']
prev = [0, 1, 2, 0, 1]

print("GO!")
startTime = time.time()
counter = 0

try:
    angle = 0
    while True:
        baseSpeed = 20  # base wheel speeds

        # get an image from the the robot
        image = camera.frame

        # apply any image transformations
        image = transform(cv2.resize(image[80:,:], dsize=(64,32), interpolation=cv2.INTER_CUBIC))
        
        # pass image through network to get a prediction for the steering angle
        steering = net(image)
        _, predicted = torch.max(steering, 0)
        # print(actions[predicted])

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

        # update turn angle 
        # if predicted == 0:
        #     goal = -0.5
        #     baseSpeed -= 10
        # elif predicted == 1:
        #     goal = 0.5
        #     baseSpeed -= 10
        # else:
        #     goal = 0
        if left > 3:
            goal = -maxTurn
            baseSpeed -= 10
        elif right > 3:
            goal = maxTurn
            baseSpeed -= 10
        else:
            goal = 0

        angle += Kn*(goal - angle)

        # update motor speeds
        left  = int(baseSpeed + Ka*angle)
        right = int(baseSpeed - Ka*angle)
        ppi.set_velocity(left,right)
        # print(str(left) + ' ' + str(right))

        counter += 1
        if time.time() - startTime > 1
            print(str(counter) + 'inferences')
            startTime = time.time()
        
        
except KeyboardInterrupt:    
    ppi.set_velocity(0,0)
    print('\naverage inferences/s:\t' + str(counter/time.time()))