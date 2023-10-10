import numpy as np
import scipy.io as sio
from PIL import Image


def open_image(name):
    img = Image.open(name)
    img = np.array(img)
    return img


def cut_image():
    step_size=8
    width = int(1024/step_size)
    height = int(256/step_size)
    orf = open_image('./rf.png')
    odf = open_image('./df.png')
    Isize = 32
    n=0
    uu = np.zeros([Isize,Isize,2])
    e = np.zeros([Isize,Isize,3])
    mdic = {"uu": uu, "e":e}
    
    for i in range(height-int(Isize/step_size)+1):
        for j in range(width-int(Isize/step_size)+1):
            n = n+1
            rf=orf[i*step_size:i*step_size+Isize,j*step_size:j*step_size+Isize]
            df=odf[i*step_size:i*step_size+Isize,j*step_size:j*step_size+Isize]
            
            rname = './train/'+'rf'+'%05d'%(n)+'.png'
            dname = './train/'+'df'+'%05d'%(n)+'.png'
            uename = './train/'+'ue'+'%05d'%(n)+'.mat'
            
            sio.savemat(uename, mdic)
            rf = Image.fromarray(rf.astype(np.uint8))
            rf.save(rname)
            df = Image.fromarray(df.astype(np.uint8))
            df.save(dname)
        
    