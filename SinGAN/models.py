import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from SinGAN.partialconv.models.partialconv2d import PartialConv2d

class ConvBlock(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size, padd, stride):
        super(ConvBlock,self).__init__()
        self.add_module('conv',nn.Conv2d(in_channel ,out_channel,kernel_size=ker_size,stride=stride,padding=padd)),
        self.add_module('norm',nn.BatchNorm2d(out_channel)),
        self.add_module('LeakyRelu',nn.LeakyReLU(0.2, inplace=True))

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
   
class WDiscriminator(nn.Module):
    def __init__(self, opt):
        super(WDiscriminator, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = int(opt.nfc)
        self.head = ConvBlock(opt.nc_im,N,opt.ker_size,opt.padd_size,1)
        self.body = nn.Sequential()
        for i in range(opt.num_layer-2):
            N = int(opt.nfc/pow(2,(i+1)))
            block = ConvBlock(max(2*N,opt.min_nfc),max(N,opt.min_nfc), opt.ker_size, opt.padd_size, 1)
            self.body.add_module('block%d'%(i+1),block)
        self.tail = nn.Conv2d(max(N,opt.min_nfc),1,kernel_size=opt.ker_size,stride=1,padding=opt.padd_size)

    def forward(self,x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x

class GeneratorConcatSkip2CleanAdd(nn.Module):
    def __init__(self, opt):
        super(GeneratorConcatSkip2CleanAdd, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = opt.nfc
        self.head = ConvBlock(opt.nc_im,N,opt.ker_size,opt.padd_size,1) #GenConvTransBlock(opt.nc_z,N,opt.ker_size,opt.padd_size,opt.stride)
        self.body = nn.Sequential()
        for i in range(opt.num_layer-2):
            N = int(opt.nfc/pow(2,(i+1)))
            block = ConvBlock(max(2*N,opt.min_nfc),max(N,opt.min_nfc), opt.ker_size, opt.padd_size, 1)
            self.body.add_module('block%d'%(i+1),block)
        self.tail = nn.Sequential(
            nn.Conv2d(max(N,opt.min_nfc),opt.nc_im,kernel_size=opt.ker_size,stride =1,padding=opt.padd_size),
            nn.Tanh()
        )

    def forward(self,x,y):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        ind = int((y.shape[2]-x.shape[2])/2)
        y = y[:,:,ind:(y.shape[2]-ind),ind:(y.shape[3]-ind)]
        return x+y

class GeneratorConcatSkip2CleanAddMask(nn.Module):
    def __init__(self, opt):
        super(GeneratorConcatSkip2CleanAddMask, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = opt.nfc
        self.head = ConvBlock(opt.nc_im,N,opt.ker_size,opt.padd_size,1) #GenConvTransBlock(opt.nc_z,N,opt.ker_size,opt.padd_size,opt.stride)
        self.body = nn.Sequential()
        for i in range(opt.num_layer-2):
            N = int(opt.nfc/pow(2,(i+1)))
            block = ConvBlock(max(2*N,opt.min_nfc),max(N,opt.min_nfc), opt.ker_size, opt.padd_size, 1)
            self.body.add_module('block%d'%(i+1),block)
        self.tail = nn.Sequential(
            nn.Conv2d(max(N,opt.min_nfc),opt.nc_im,kernel_size=opt.ker_size,stride =1,padding=opt.padd_size),
            nn.Tanh()
        )
        self.mask = opt.mask

    def forward(self,x,y):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        ind = int((y.shape[2]-x.shape[2])/2)
        y = y[:,:,ind:(y.shape[2]-ind),ind:(y.shape[3]-ind)]
        return self.mask*(x+y)

"""class PartialConvBlock(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size, padd, stride):
        super(PartialConvBlock,self).__init__()
        self.add_module('conv',PartialConv2d(in_channel ,out_channel,kernel_size=ker_size,stride=stride,padding=padd, multi_channel=True, return_mask = True)),
        self.add_module('norm',nn.BatchNorm2d(out_channel)),
        self.add_module('LeakyRelu',nn.LeakyReLU(0.2, inplace=True))"""

class WDiscriminatorPartial(nn.Module):
    def __init__(self, opt):
        super(WDiscriminatorPartial, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = int(opt.nfc)
        self.head = PartialConv2d(opt.nc_im, N, kernel_size = opt.ker_size, padding = opt.padd_size, stride = 1,  multi_channel=True, return_mask = True)
        self.headbis = nn.Sequential(nn.BatchNorm2d(N),
                                    nn.LeakyReLU(0.2, inplace = True))
        self.body = nn.ModuleList([])
        self.bodybis = nn.ModuleList([])
        self.num_layer = opt.num_layer
        for i in range(opt.num_layer-2):
            N = int(opt.nfc/pow(2,(i+1)))
            block = PartialConv2d(max(2*N,opt.min_nfc),max(N,opt.min_nfc),kernel_size = opt.ker_size,padding = opt.padd_size,stride = 1,  multi_channel=True, return_mask = True)
            self.body.append(block)
            self.bodybis.append(nn.Sequential(nn.BatchNorm2d(max(N,opt.min_nfc)), nn.LeakyReLU(0.2, inplace = True)))
        self.tail = PartialConv2d(max(N, opt.min_nfc), 1, kernel_size=opt.ker_size, stride=1, padding=opt.padd_size,  multi_channel=True, return_mask = True)
        self.tailbis = nn.Sequential(nn.BatchNorm2d(1),
                                    nn.LeakyReLU(0.2, inplace = True))
        self.mask = opt.masks[opt.cur_scale]

    def forward(self,x):
        x, mask = self.head(x, self.mask)
        x = self.headbis(x)
        for i in range(self.num_layer - 2):
            x, mask = self.body[i](x, mask)
            x = self.bodybis[i](x)
        x, mask = self.tail(x, mask)
        x = self.tailbis(x)
        return x

class GeneratorConcatSkip2CleanAddPartial(nn.Module):
    def __init__(self, opt):
        super(GeneratorConcatSkip2CleanAddPartial, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = opt.nfc
        self.mask = opt.masks[opt.cur_scale]
        self.pad = nn.ZeroPad2d(int(((opt.ker_size - 1) * opt.num_layer) / 2))
        self.head = PartialConv2d(opt.nc_im, N, kernel_size = opt.ker_size, padding = opt.padd_size, stride = 1, multi_channel=True, return_mask = True)
        self.headbis = nn.Sequential(nn.BatchNorm2d(N),
                                     nn.LeakyReLU(0.2, inplace=True))
        self.body = nn.ModuleList([])
        self.bodybis = nn.ModuleList([])
        self.num_layer = opt.num_layer
        for i in range(opt.num_layer-2):
            N = int(opt.nfc/pow(2,(i+1)))
            block = PartialConv2d(max(2*N,opt.min_nfc),max(N,opt.min_nfc), kernel_size = opt.ker_size, padding = opt.padd_size, stride = 1, multi_channel=True, return_mask = True)
            self.body.append(block)
            self.bodybis.append(nn.Sequential(nn.BatchNorm2d(max(N,opt.min_nfc)), nn.LeakyReLU(0.2, inplace = True)))
        self.tail = PartialConv2d(max(N, opt.min_nfc), opt.nc_im, kernel_size=opt.ker_size, stride=1, padding=opt.padd_size, multi_channel=True, return_mask = True)
        self.tailbis = nn.Tanh()

    def forward(self,x,y):
        mask = self.pad(self.mask)
        x, mask = self.head(x, mask)
        x = self.headbis(x)
        for i in range(self.num_layer - 2):
            x, mask = self.body[i](x, mask)
            x = self.bodybis[i](x)
        x, mask = self.tail(x)
        x = self.tailbis(x)
        ind = int((y.shape[2]-x.shape[2])/2)
        y = y[:,:,ind:(y.shape[2]-ind),ind:(y.shape[3]-ind)]
        return x+y