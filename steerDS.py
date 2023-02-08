import numpy as np
from glob import glob
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import cv2
from glob import glob
from os import path

CUTOFF = 0.2

class SteerDataSet(Dataset):
    
    def __init__(self,root_folder,img_ext = ".jpg" , transform=None):
        self.root_folder = root_folder
        self.transform = transform        
        self.img_ext = img_ext        
        self.filenames = glob(path.join(self.root_folder,"*" + self.img_ext))
        self.totensor = transforms.ToTensor()
        
    def __len__(self):        
        return len(self.filenames)
    
    def __getitem__(self,idx):
        f = self.filenames[idx]       

        # crop top of image
        img = self.preprocess(cv2.imread(f))

        if self.transform == None:
            img = self.totensor(img)
        else:
            img = self.transform(img)

        try:
            img_name = f.split("/")[-1].split(self.img_ext)[0]
            steering = img_name[-3:]
            if '-' in img_name:
                steering = -1 * np.float32(steering)
            else:
                steering = np.float32(steering)
        except:
            img_name = f.split("\\")[-1].split(self.img_ext)[0]
            steering = img_name[-3:]
            if '-' in img_name:
                steering = -1*np.float32(steering)
            else:
                steering = np.float32(steering)


        # convert steering angle to classification classes
        if steering < -1*CUTOFF:
            steer = 0   # left
        elif steering > CUTOFF:
            steer = 1   # right
        else:
            steer = 2   # straight

        sample = {"image":img , "steering":steer} 

        return sample

    def preprocess(self, image):
        '''Applies pre-processing to images
        Crops top of image
        Resizes to 32x32
        Applies greyscale
        '''
        return cv2.cvtColor(cv2.resize(image[80:,:], dsize=(32,32),interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2GRAY)
        

def test():
    transform = transforms.Compose(
        [transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    ds = SteerDataSet("data/",".jpg",transform)

    print("The dataset contains %d images " % len(ds))

    ds_dataloader = DataLoader(ds,batch_size=1,shuffle=True)
    for S in ds_dataloader:
        im = S["image"]    
        y  = S["steering"]

        cv2.imshow('image', np.array(im))
        cv2.waitKey(0)

        print(im.shape)
        print(y)
        break



if __name__ == "__main__":
    test()
