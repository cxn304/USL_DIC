import torch.nn as nn
import torch
import torch.nn.functional as F
    

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)



class pure_unet(nn.Module):
    # non survey adjustment
    def __init__(self, in_ch, out_ch, heights,widths):
        super(pure_unet, self).__init__()
        self.conv1 = DoubleConv(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(512, 1024)
        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6 = DoubleConv(1024, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)    # stride改了
        self.conv7 = DoubleConv(512, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = DoubleConv(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = DoubleConv(128, 64)
        self.convu0 = nn.Conv2d(64, 1,3,padding=1)
        self.convv0 = nn.Conv2d(64, 1,3,padding=1)

        self.heights = heights
        self.widths = widths
        heightz = torch.arange(0,self.heights).cuda(0)
        widthz = torch.arange(0,self.widths).cuda(0)
        self.yy,self.xx = torch.meshgrid(heightz,widthz)#,indexing='ij'
    
    
    def calculate_def(self,x,outu):
        '''
        use this,continunity of displacement
        '''
        refimg = x[:,0,:,:]
        refimg = refimg.unsqueeze(1)
        output1,newu1 = self.continunity(4,outu,refimg)      
        return output1.squeeze(1),newu1
    
    
    def continunity(self,windowsize,outu,refimg):
        modes = 'bicubic'
        outu1=F.interpolate(outu,scale_factor=1/windowsize,
                            mode=modes,align_corners=True)
        newu1=F.interpolate(outu1,scale_factor=windowsize,
                           mode=modes,align_corners=True)
        xi1 = (self.xx+newu1[:,0,:,:])
        yi1 = (self.yy+newu1[:,1,:,:])
        grid1 = torch.stack((xi1/(self.widths-1)*2-1, 
                            yi1/(self.heights-1)*2-1),-1)
        output1 = F.grid_sample(refimg,grid1,mode='bicubic',align_corners=True)
        return output1,newu1
        
    
    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)
        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        c7 = self.conv8(up_7)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)
        outu0 = self.convu0(c9)
        outv0 = self.convv0(c9)

        df0,newu0 = self.calculate_def(x,torch.cat((outu0,outv0),1))

        return df0,newu0

    
    
  
class pure_unet_3(nn.Module):
    # non survey adjustment
    def __init__(self, in_ch, out_ch, heights,widths):
        super(pure_unet_3, self).__init__()
        self.conv1 = DoubleConv(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256, 512)
        
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = DoubleConv(512, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = DoubleConv(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = DoubleConv(128, 64)
        self.convu0 = nn.Conv2d(64, 1,3,padding=1)
        self.convv0 = nn.Conv2d(64, 1,3,padding=1)

        self.heights = heights
        self.widths = widths
        heightz = torch.arange(0,self.heights).cuda(0)
        widthz = torch.arange(0,self.widths).cuda(0)
        self.yy,self.xx = torch.meshgrid(heightz,widthz)#,indexing='ij'
    
    
    def calculate_def(self,x,outu):
        '''
        use this,continunity of displacement
        '''
        refimg = x[:,0,:,:]
        refimg = refimg.unsqueeze(1)
        output1,newu1 = self.continunity(4,outu,refimg)      
        return output1.squeeze(1),newu1
    
    
    def continunity(self,windowsize,outu,refimg):
        modes = 'bicubic'
        outu1=F.interpolate(outu,scale_factor=1/windowsize,
                            mode=modes,align_corners=True)
        newu1=F.interpolate(outu1,scale_factor=windowsize,
                           mode=modes,align_corners=True)
        xi1 = (self.xx+newu1[:,0,:,:])
        yi1 = (self.yy+newu1[:,1,:,:])
        grid1 = torch.stack((xi1/(self.widths-1)*2-1, 
                            yi1/(self.heights-1)*2-1),-1)
        output1 = F.grid_sample(refimg,grid1,mode='bicubic',align_corners=True)
        return output1,newu1
        
    
    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)

        up_7 = self.up7(c4)
        c7 = self.conv8(up_7) 
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)
        outu0 = self.convu0(c9)
        outv0 = self.convv0(c9)

        df0,newu0 = self.calculate_def(x,torch.cat((outu0,outv0),1))

        return df0,newu0