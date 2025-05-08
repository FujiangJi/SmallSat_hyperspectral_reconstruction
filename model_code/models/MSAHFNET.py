import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2


  
class MSHFNET(nn.Module):
    def __init__(self, 
                 scale_ratio,
                 n_select_bands, 
                 n_bands):
        super(MSHFNET, self).__init__()
        self.scale_ratio = scale_ratio
        self.n_bands = n_bands
        self.n_select_bands = n_select_bands
      
      
        self.res_block1 = nn.Sequential(
              nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
              nn.PReLU(),
              nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
              nn.PReLU(),
              )
        self.res_block2 = nn.Sequential(
              nn.Conv2d(60, 60, kernel_size=3, stride=1, padding=1),
              nn.PReLU(),
              nn.Conv2d(60, 60, kernel_size=3, stride=1, padding=1),
              nn.PReLU(),
              )
      
        self.branch1_cov1 = nn.Sequential(
              nn.Conv2d(n_bands+n_select_bands, 128, kernel_size=3, stride=1, padding=1),
              nn.PReLU(),
              )
        self.branch1_cov2 = nn.Sequential(
              nn.Conv2d(128, 60, kernel_size=3, stride=1, padding=1),
              nn.PReLU(),
              )
      
        self.branch1_cov3 = nn.Sequential(
              nn.Conv2d(60, 60, kernel_size=3, stride=1, padding=1),
              nn.PReLU(),
              )
        
        self.branch1_conv_1x1 = nn.Sequential(
                  nn.Conv2d(60, 20, kernel_size=1, stride=1, padding=0),
                  nn.PReLU(),
                )
        self.branch1_conv_3x3 = nn.Sequential(
                  nn.Conv2d(60, 20, kernel_size=3, stride=1, padding=1),
                  nn.PReLU(),
                )
        self.branch1_conv_5x5 = nn.Sequential(
                  nn.Conv2d(60, 20, kernel_size=5, stride=1, padding=2),
                  nn.PReLU(),
                )

        self.branch2_cov1 = nn.Sequential(
              nn.Conv2d(120, 120, kernel_size=3, stride=1, padding=1),
              nn.PReLU(),
              )
        
        self.branch3_cov1 = nn.Sequential(
              nn.Conv2d(240, 240, kernel_size=3, stride=1, padding=1),
              nn.PReLU(),
              )
        
        self.down_conv1 = nn.Sequential(
              nn.Conv2d(60, 120, kernel_size=2, stride=2, padding=0),
              nn.PReLU(),
              )

        self.up_conv1 = nn.Sequential(
              nn.ConvTranspose2d(120, 60, kernel_size=2, stride=2, padding=0),
              nn.PReLU(),
              )
        
        self.final_conv = nn.Sequential(
              nn.Conv2d(60, n_bands, kernel_size=3, stride=1, padding=1),
              nn.PReLU(),
              )
    
    def forward(self, x_lr, x_hr):
        x_lr = F.interpolate(x_lr, scale_factor=self.scale_ratio, mode='bilinear')
        x = torch.cat((x_hr, x_lr), dim=1)
        
        x1 = self.branch1_cov1(x)
        xt = self.res_block1(x1)
        x1 = x1+xt
        
        x1 = self.branch1_cov2(x1)
        xt = self.res_block2(x1)
        x1 = x1+xt
        
        x2 = self.down_conv1(x1)
        x2 = self.branch2_cov1(x2)
        
        x1 = x1+ torch.cat((self.branch1_conv_1x1(x1),
                            self.branch1_conv_3x3(x1),
                            self.branch1_conv_5x5(x1)),
                           dim=1)
        
        ###
        x1t = self.up_conv1(x2)
        x2t = self.down_conv1(x1)
        
        x1 = self.branch1_cov3(x1)
        x2 = self.branch2_cov1(x2)

        x1 = x1+x1t
        x2 = x2+x2t
        
       
        x2 = self.up_conv1(x2)
        x = x1+x2
        x = self.final_conv(x)
      
        return x
   

class ChannelAttention(nn.Module):
    
    def __init__(self, n_bands, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(n_bands, n_bands // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(n_bands // ratio, n_bands, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        # self.register_buffer()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
    
class HRNet(nn.Module):
    def __init__(self, 
                 scale_ratio,
                 n_select_bands, 
                 n_bands):
        super(HRNet, self).__init__()
        self.scale_ratio = scale_ratio
        self.n_bands = n_bands
        self.n_select_bands = n_select_bands
      
      
        self.res_block1 = nn.Sequential(
              nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
              nn.PReLU(),
              nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
              nn.PReLU(),
              )
        self.res_block2 = nn.Sequential(
              nn.Conv2d(60, 60, kernel_size=3, stride=1, padding=1),
              nn.PReLU(),
              nn.Conv2d(60, 60, kernel_size=3, stride=1, padding=1),
              nn.PReLU(),
              )
      
        self.branch1_cov1 = nn.Sequential(
              nn.Conv2d(n_bands+n_select_bands, 128, kernel_size=3, stride=1, padding=1),
              nn.PReLU(),
              )
        self.branch1_cov2 = nn.Sequential(
              nn.Conv2d(128, 60, kernel_size=3, stride=1, padding=1),
              nn.PReLU(),
              )
      
        self.branch1_cov3 = nn.Sequential(
              nn.Conv2d(60, 60, kernel_size=3, stride=1, padding=1),
              nn.PReLU(),
              )
        
        self.branch1_conv_1x1 = nn.Sequential(
                  nn.Conv2d(60, 20, kernel_size=1, stride=1, padding=0),
                  nn.PReLU(),
                )
        self.branch1_conv_3x3 = nn.Sequential(
                  nn.Conv2d(60, 20, kernel_size=3, stride=1, padding=1),
                  nn.PReLU(),
                )
        self.branch1_conv_5x5 = nn.Sequential(
                  nn.Conv2d(60, 20, kernel_size=5, stride=1, padding=2),
                  nn.PReLU(),
                )

        self.branch2_cov1 = nn.Sequential(
              nn.Conv2d(120, 120, kernel_size=3, stride=1, padding=1),
              nn.PReLU(),
              )
        
        self.branch3_cov1 = nn.Sequential(
              nn.Conv2d(240, 240, kernel_size=3, stride=1, padding=1),
              nn.PReLU(),
              )
        
        self.down_conv1 = nn.Sequential(
              nn.Conv2d(60, 120, kernel_size=2, stride=2, padding=0),
              nn.PReLU(),
              )

        self.up_conv1 = nn.Sequential(
              nn.ConvTranspose2d(120, 60, kernel_size=2, stride=2, padding=0),
              nn.PReLU(),
              )
        
        self.final_conv = nn.Sequential(
              nn.Conv2d(60, n_bands, kernel_size=3, stride=1, padding=1),
              nn.PReLU(),
              )
    
    def forward(self, x_lr, x_hr):
        x_lr = F.interpolate(x_lr, scale_factor=self.scale_ratio, mode='bilinear')
        x = torch.cat((x_hr, x_lr), dim=1)
        
        x1 = self.branch1_cov1(x)
        xt = self.res_block1(x1)
        x1 = x1+xt
        
        x1 = self.branch1_cov2(x1)
        xt = self.res_block2(x1)
        x1 = x1+xt
        
        x2 = self.down_conv1(x1)
        x2 = self.branch2_cov1(x2)
        
        x1 = x1+ torch.cat((self.branch1_conv_1x1(x1),
                            self.branch1_conv_3x3(x1),
                            self.branch1_conv_5x5(x1)),
                           dim=1)
        
        ###
        x1t = self.up_conv1(x2)
        x2t = self.down_conv1(x1)
        
        x1 = self.branch1_cov3(x1)
        x2 = self.branch2_cov1(x2)

        x1 = x1+x1t
        x2 = x2+x2t
        
       
        x2 = self.up_conv1(x2)
        x = x1+x2
        x = self.final_conv(x)
      
        return x
  
class MSAHFNET(nn.Module):
      def __init__(self, scale_ratio, n_select_bands, n_bands):
        super(MSAHFNET, self).__init__()
        self.CAM = ChannelAttention(n_bands)
        self.SAM = SpatialAttention()
        self.hr_net = HRNet(scale_ratio, n_select_bands, n_bands)

      def forward(self, x_lr, x_hr):
            x = self.hr_net(x_lr, x_hr)
            ca = self.CAM(x_lr)
            sa = self.SAM(x_hr)
            xt = torch.mul(x, ca)
            xt = torch.mul(xt, sa)
            x = x+xt
            return x
        
        
        
        
        
        
        
        
        