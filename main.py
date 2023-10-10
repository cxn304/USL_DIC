from time import time
import math,random
import shutil
import pdb
import scipy.io as io
import numpy as np
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import matplotlib.pyplot as plt
import pandas as pd
from utils.readData import cxnDataset
from src.unsupervised_unet import pure_unet,pure_unet_3
import os
import scipy.io as sio
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class Args_cxn():
    # VGG_MEAN = [103.939, 116.779, 123.68]
    def __init__(self):
        #self.workers = 0
        self.print_freq = 40
        self.checkpoint_path=\
            "./drive/MyDrive/unsupervised_learning_dic/u_128.pth"
        self.loss_path="./drive/MyDrive/unsupervised_learning_dic/u_128.csv"
        self.train = './train/'
        self.val = './validate/'
        self.epochs = 101
        self.warmup = 5
        self.batch_size = 16
        self.lr = 0.0005
        self.weight_decay = 3e-2
        self.clip_grad_norm = 10
        self.gpu_id = 0
        self.disable_cos = False
        self.disable_aug = False
        self.no_cuda = False
        self.add_all_features = False 
        self.RESUME = False


        
def imagesc(outputu,output_def,xx,args):
    ru = outputu[:,0,:,:]
    rv = outputu[:,1,:,:]
    real_def = xx[:,1,:,:]
    cd,_,_=ru.shape
    jiange = cd//4
    plt.figure(figsize=(16, 14))
    plt.subplots_adjust(wspace =.4, hspace =.4)
    plt.axis('on')
    rpu = []
    rpv = []
    defi = []
    rreal_def = []

    for i in range(0,cd,jiange):
        ru1=ru[i,:,:]
        rv1=rv[i,:,:]
        defi1=output_def[i,:,:]
        real_def1=real_def[i,:,:]
        if (not args.no_cuda) and torch.cuda.is_available():
            ru1 = ru1.detach().cpu().numpy()
            rv1 = rv1.detach().cpu().numpy()
            defi1 = defi1.detach().cpu().numpy()
            real_def1 = real_def1.detach().cpu().numpy()
        else:
            ru1 = ru1.detach().numpy()
            rv1 = rv1.detach().numpy()
            defi1 = defi1.detach().numpy()
            real_def1 = real_def1.detach().numpy()
        rpu.append(ru1)
        rpv.append(rv1)     
        defi.append(defi1)
        rreal_def.append(real_def1)

    def plot_img(i,data,name,ii):
        ax = plt.subplot(4,4,4*i+ii)
        plt.imshow(data[i])
        plt.colorbar(shrink=0.8)
        ax.set_title(name)
        ax.invert_yaxis()

        
    for i in range(len(rpu)):
        plot_img(i,rpu,'predict u',1)
        sio.savemat('./mat/predict_u.mat', {"predict_u": rpu[i]})
        plot_img(i,rpv,'predict v',2)
        sio.savemat('./mat/predict_v.mat', {"predict_v": rpv[i]})
        plot_img(i,defi,'predict def',3)
        plot_img(i,rreal_def,'real def',4)

    plt.show()
    plt.close()
    
        
def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr
    if hasattr(args, 'warmup') and epoch < args.warmup:
        lr = lr / (args.warmup - epoch)
    elif not args.disable_cos:
        lr *= 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup) / (args.epochs - args.warmup)))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def cls_train(train_loader, model, criterion, optimizer, epoch,args):
    model.train()
    loss_val = 0
    n = 0
    for i, (xx) in enumerate(train_loader):
        if (not args.no_cuda) and torch.cuda.is_available():
            xx = xx.cuda(0, non_blocking=True)

        xx = xx.to(torch.float32)

        df0,newu0=model(xx)
        wk = 1
        loss0 = criterion(df0[:,wk:-wk,wk:-wk],xx[:,1,wk:-wk,wk:-wk])  
        loss = loss0

        n += xx.size(0)
        loss_val += float(loss.item() * xx.size(0)) 

        optimizer.zero_grad()
        loss.backward()

        if args.clip_grad_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), 
                        max_norm=args.clip_grad_norm, norm_type=2)

        optimizer.step()

        if args.print_freq >= 0 and i % args.print_freq == 0:
            avg_loss = (loss_val / n)
            print(f'[Epoch {epoch+1}][Train][{i}] \t Loss: {avg_loss:.4e} ')
        
        #imagesc(outputu,outpute,uu,ee,args)
    if epoch % (args.epochs//2) == 0:
        #imagesc(newu0,df0,xx,args)

        if not os.path.exists('./to_matlab'):
          os.mkdir('./to_matlab')
        outputu0 = newu0.cpu().detach().numpy()

        io.savemat('./to_matlab/outputua'+str(epoch)+'.mat',
                   {'outputu':outputu0})

    return avg_loss

def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True



if __name__ == '__main__':
    seed = 10
    seed_everything(seed)
    args = Args_cxn()
    # 400,1392 336,1808
    model = pure_unet(2,2,256,256)
    # mean_mse_loss SCCM_loss nn.MSELoss CrossEntropyLoss 
    # BCEWithLogitsLoss mse_loss() block_smooth_mse
    criterion = nn.MSELoss() 
    if (not args.no_cuda) and torch.cuda.is_available():
          torch.cuda.set_device(args.gpu_id)
          model.cuda(args.gpu_id)
          criterion = criterion.cuda(args.gpu_id)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                    weight_decay=args.weight_decay)
    if args.RESUME:
        path_checkpoint = args.checkpoint_path
        if torch.cuda.is_available():
          checkpoint = torch.load(path_checkpoint)
        else:
          checkpoint = torch.load(path_checkpoint,map_location="cpu")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
    else:
        start_epoch = 0
               
    train_dataset = cxnDataset(args.train,isaugument=False)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0)


    print("Beginning training")
    time_begin = time()
    Train_loss = []
    V_loss = []
    for epoch in range(start_epoch,args.epochs):
        adjust_learning_rate(optimizer, epoch, args)

        t_loss = cls_train(train_loader, model, criterion, optimizer, epoch, args)

        Train_loss.append(t_loss)

        if epoch>1 and epoch % (args.epochs//20) == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': t_loss,
                }, args.checkpoint_path)

    total_mins = (time() - time_begin) / 60
    df = pd.DataFrame({'train_loss':Train_loss})
    df.to_csv(args.loss_path)
