from steerDS import SteerDataSet

import numpy as np
from glob import glob
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import cv2 
from glob import glob
from os import path

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

ds = SteerDataSet("/home/will/ChickenJoe/data_raw/",".jpg",transform)
ds2 = SteerDataSet("/home/will/ChickenJoe/data_raw/",".jpg")

print(ds)
print(ds2)

temp_image = ds2.__getitem__(1)['image']
print(temp_image)



