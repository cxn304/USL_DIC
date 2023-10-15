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
        # 这个list存放所有图像的地址
        self.rfimage_files = np.array([x.path for x in os.scandir(train_root)
                                     if x.name.endswith(".png") and 
                                         x.name.startswith("r")])
        self.dfimage_files = np.array([x.path for x in os.scandir(train_root)
                                     if x.name.endswith(".mat") and 
                                         x.name.startswith("d")])
        self.ue_files = np.array([x.path for x in os.scandir(train_root)
                                     if x.name.endswith(".mat") and 
                                        x.name.startswith("ue")])                      
        self.mask_files = np.array([x.path for x in os.scandir(mask_root)
                                     if x.name.endswith(".npy") or
                                     x.name.endswith(".png") or 
                                     x.name.endswith(".JPG")])
        self.mask_root = mask_root # 是否需要mask
        self.isaugument = isaugument
        

    def __getitem__(self, index):

        rfimg = self.open_image(self.rfimage_files[index])
        dfimg = self.read_mat(self.dfimage_files[index])
        ulabel,elabel = self.read_label(self.ue_files[index])
        if self.isaugument:
            rfimg,dfimg,ulabel,elabel=self.augumentaion(rfimg,dfimg,ulabel,elabel)
        
        rfimg = self.to_tensor(rfimg) 
        dfimg = self.to_tensor(dfimg) 
        ulabel = self.to_tensor(ulabel) 
        elabel = self.to_tensor(elabel) 
        img=torch.cat((rfimg,dfimg),0)
        if not self.mask_root:
            return (img,ulabel,elabel)
        elif self.mask_root is not None:
            mask = self.to_tensor(self.open_image(self.mask_files[index]))
            return (img,ulabel,elabel,mask) 
            
    
    def read_mat(self,name):
        img = sio.loadmat(name)
        df = img['df']
        return df
        
        
    def read_label(self,name):
        img = sio.loadmat(name)
        u = img['uu']
        e = img['e']
        return u,e
        
    
    def augumentaion(self,rf,df,ul,el):
        p = np.random.choice([0, 1])
        if p:
            df=df[:,::-1]
            rf=rf[:,::-1]
            ul=ul[:,::-1,:]
            el=el[:,::-1,:]
        return rf,df,ul,el
        
    
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
    












