import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Double Convolutional Layers (=>)
For contracting path, it consists of the repeated application of 3 x 3 convolutions(unpadded convolutions)
Each followed by a ReLU and a 2 x 2 max pooling operation with stride 2 for downsampling.
"""
class TwoConv(nn.Module):
    # 3x3 conv -> BN -> ReLU -> 3x3 conv -> BN -> ReLU
    # Definition of the two 3x3 convolutions
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.two_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    # Forward Pass
    def forward(self, x):
        return self.two_conv(x)

"""
Downsampling Layers (Contracting Path, ↓ and =>)
For contracting path, at each downsampling step we double the number of feature channels. 
A 2 x 2 max pooling operation with stride 2.
"""
class Downsampling(nn.Module):
    # 2x2 max pooling operation with stride 2 then apply two convolutional layers
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down_sampling = nn.Sequential(
            nn.MaxPool2d(2),
            TwoConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.down_sampling
    
"""
Upsampling Layers (Expansive Path, ↑ and =>)
For expansive path, each step consists an upsampling of the feature map followed by a 2 x 2 convolution that hales the number of feature channels, a concatenation with the correspondingly cropped feature map from the contracting path, and two 3 x 3 convolutions, each followed by a ReLU.
"""
class Upsampling(nn.Module):
    # 2x2 upsampling conv -> (3x3 conv -> BN-> ReLu -> 3x3 conv -> BN -> ReLU)
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up_sampling = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = TwoConv(in_channels, out_channels, in_channels // 2)
    
    # concatenation with the correspondingly cropped feature map from the contracting path
    def forward(self, x1, x2):
        # x1: features in expansive path(current one), x2: features in contracting path(downsampling one)
        x1 = self.up_sampling(x1)
        # padding
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # combine corresponding resolution features from contracting path
        x = torch.cat([x2, x1], dim=1)
        # two 3x3 convolutions and each followed by a ReLU
        x = self.conv(x)
        return x
    
"""
Final 1 x 1 Convolutional Layer (->)
At the final layer a 1 x 1 convolution is used to map each 64-component feature vector to the desired number of classes.
"""
class FinalConv(nn.Module):
    # 1x1 conv 
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        x = self.conv(x)
        return x
    
"""
U-Net
In order to localize, high resolution features from the contracting path are combined with the upsampled output. 
Two-Conv -> Down -> Down -> Down -> Down -> Up -> Up -> Up -> Up -> Final
"""
class UNet(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.two_conv = TwoConv(in_channels, 64)
        
        self.Down1 = Downsampling(64, 128)
        self.Down2 = Downsampling(128, 256)
        self.Down3 = Downsampling(256, 512)
        self.Down4 = Downsampling(512, 1024 // 2) # only take a half of it, and another half is from downsampling of size 512 
        
        # For upsampling layers, combine two part and take half size, except the last upsampling
        self.Up1 = Upsampling(1024, 512 // 2)
        self.Up2 = Upsampling(512, 256 // 2)
        self.Up3 = Upsampling(256, 128 // 2)
        self.Up4 = Upsampling(128, 64) # 128->64->64, and it won't be a half since we don't need upsampling combination with another part anymore
        
        self.final = FinalConv(64, n_classes)
        
    def forward(self, x):
        # x1~x5:features in each layer of contracting path
        x1 = self.two_conv(x)
        x2 = self.Down1(x1)
        x3 = self.Down2(x2)
        x4 = self.Down3(x3)
        x5 = self.Down4(x4)
        
        # Upsampling, combine x1~x5
        x = self.Up1(x5, x4)
        x = self.Up2(x, x3)
        x = self.Up3(x, x2)
        x = self.Up4(x, x1)
        
        out = self.final(x)
        
        return out