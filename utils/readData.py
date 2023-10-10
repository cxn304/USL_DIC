import torch,os,time
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio


class cxnDataset(Dataset):
    """
     return: tuple: (x,y) x:[batch,2,imgsize,imgsize]
    """
    def __init__(self, train_root,isaugument, mask_root = None):
        self.rfimage_files = np.array([x.path for x in os.scandir(train_root)
                                     if x.name.endswith(".png") and 
                                         x.name.startswith("r")])
        self.rfimage_files.sort()
        self.dfimage_files = np.array([x.path for x in os.scandir(train_root)
                                     if x.name.endswith(".png") and 
                                         x.name.startswith("d")])
        self.dfimage_files.sort()
        self.ue_files = np.array([x.path for x in os.scandir(train_root)
                                     if x.name.endswith(".mat") and 
                                        x.name.startswith("ue")]) 
        self.ue_files.sort()                     
        self.mask_files = np.array([x.path for x in os.scandir(mask_root)
                                     if x.name.endswith(".npy") or
                                     x.name.endswith(".png") or 
                                     x.name.endswith(".JPG")])
        self.mask_files.sort()
        self.mask_root = mask_root
        self.isaugument = isaugument
        

    def __getitem__(self, index):
        rfimg = self.open_image(self.rfimage_files[index])
        dfimg = self.open_image(self.dfimage_files[index])
        
        rfimg = self.to_tensor(rfimg) 
        dfimg = self.to_tensor(dfimg) 
        img=torch.cat((rfimg,dfimg),0)
        if not self.mask_root:
            return (img)
        elif self.mask_root is not None:
            mask = self.to_tensor(self.open_image(self.mask_files[index]))
            return (img,mask)
        
    
    def open_image(self,name):
        img = Image.open(name)
        img = np.array(img)
        return img
        
    
    def to_tensor(self,img):
        img = torch.from_numpy(img.astype(np.float32)).unsqueeze(0)
        # img = img.repeat(3,1,1)
        return img
        

    def __len__(self):
        return len(self.rfimage_files)
    
 












