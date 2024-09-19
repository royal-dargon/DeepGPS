import torch
import torch.nn as nn
from typing import List, Union, Dict, Tuple, Optional
import torch.nn.functional as F
from torch.nn import MSELoss, BCELoss, L1Loss

class DoubleConv(nn.Module):
    def __init__(
        self,
        in_channels: int = None,
        out_channels: int = None,
        mid_channels: int = None,
        kernel_size: int = None,
    ) -> None:
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        #self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size, padding=1, bias=False)
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size, padding=1)
        self.act1 = nn.ReLU(inplace=True)
        self.batchnorm1 = nn.BatchNorm2d(mid_channels)
        #self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size, padding=1, bias=False)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size, padding=1)
        self.act2 = nn.ReLU(inplace=True)
        self.batchnorm2 = nn.BatchNorm2d(out_channels)

        self.post_init()

    def post_init(
        self,
    ) -> None:
        nn.init.kaiming_uniform_(self.conv1.weight)
        nn.init.constant_(self.conv1.bias, 0.0)
        nn.init.kaiming_uniform_(self.conv2.weight)
        nn.init.constant_(self.conv2.bias, 0.0)
        nn.init.constant_(self.batchnorm1.weight, 1.0)
        nn.init.constant_(self.batchnorm1.bias, 0.0)
        nn.init.constant_(self.batchnorm2.weight, 1.0)
        nn.init.constant_(self.batchnorm2.bias, 0.0)

    def forward(
        self, 
        img_in: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        img_inter = self.act1(self.batchnorm1(self.conv1(img_in)))
        img_out = self.act2(self.batchnorm2(self.conv2(img_inter)))
        return img_out

class Down(nn.Module):
    def __init__(
        self, 
        in_channels: int = None, 
        out_channels: int = None,
        kernel_size: int = None,
    ) -> None:
        super(Down, self).__init__()
        self.max_pool = nn.MaxPool2d(2)
        self.doubleconv = DoubleConv(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size)

    def forward(
        self,
        img_in: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        img_inter = self.max_pool(img_in)
        img_out = self.doubleconv(img_inter)
        # img_out = self.attention(img_out) # 
        return img_out

class Up(nn.Module):
    def __init__(
        self, 
        in_channels: int = None,
        out_channels: int = None,
        kernel_size: int = None,
    ) -> None:
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2)
        self.conv = DoubleConv(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size)

        self.post_init()

    def post_init(
        self,
    ) -> None:
        nn.init.kaiming_uniform_(self.up.weight)
        nn.init.constant_(self.up.bias, 0.0)

    def forward(
        self,
        x_img1: Optional[torch.Tensor] = None,
        x_img2: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x_img1 = self.up(x_img1)
        diffY = x_img2.size()[2] - x_img1.size()[2]
        diffX = x_img2.size()[3] - x_img1.size()[3]

        x_img1 = F.pad(x_img1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        batch = x_img2.shape[0]
        height = x_img2.shape[2]
        width = x_img2.shape[3]

        x = torch.cat([x_img2, x_img1], dim = 1)
        x = self.conv(x)
        return x

class OutConv(nn.Module):
    def __init__(
        self, 
        in_channels: int = None, 
        out_channels: int = None,
    ) -> None:
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)

        self.post_init()

    def post_init(
        self,
    ) -> None:
        nn.init.kaiming_uniform_(self.conv.weight)
        nn.init.constant_(self.conv.bias, 0.0)
        
    def forward(
        self, 
        x_img: Optional[torch.Tensor] = None
    ) -> None:
        x_img = self.conv(x_img)
        return x_img

class UnetAdd(nn.Module):
    def __init__(self, 
        config = None,
    ) -> None:
        super(UnetAdd, self).__init__()
        self.in_channels = config.in_channels
        self.out_channels = config.out_channels
        self.hid_size = config.hid_size

        self.inc = DoubleConv(in_channels=self.in_channels, out_channels = 64, kernel_size = 3)
        self.down1 = Down(64, 128, 3)
        self.down2 = Down(128, 256, 3)
        self.down3 = Down(256, 512, 3)
        self.down4 = Down(512, 1024, 3)

        self.up1 = Up(1024, 512, 3)
        self.up2 = Up(512, 256, 3)
        self.up3 = Up(256, 128, 3)
        self.up4 = Up(128, 64, 3)

        # 序列信息处理, 这边需要处理
        self.linear_s1 = nn.Linear(320, 169)
        self.act_s1 = nn.ReLU(inplace=True)
        self.batchnorm_s1 = nn.BatchNorm1d(2700)

        self.linear_sv = nn.MultiheadAttention(embed_dim=1024, num_heads=1, batch_first=True)

        # 输出网络
        self.outc = OutConv(64, self.out_channels)
        self.sigmoid = nn.Sigmoid()
        # loss函数
        if config.image_loss == "mse":
            self.loss_fct = MSELoss()
        elif config.image_loss == "bce":
            self.loss_fct = BCELoss()
        elif config.image_loss == "mae":
            self.loss_fct = L1Loss()
        
        
        self.pos_init()
        
    def pos_init(self):
        nn.init.kaiming_uniform_(self.linear_s1.weight)
        nn.init.constant_(self.linear_s1.bias, 0.0)
        nn.init.constant_(self.batchnorm_s1.weight, 1.0)
        nn.init.constant_(self.batchnorm_s1.bias, 0.0)
        
    def forward(
        self,
        hid_states: Optional[torch.Tensor] = None,
        x_img: Optional[torch.Tensor] = None,
        y_img: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor]:
        x1 = self.inc(x_img)
        x2 = self.down1(x1) 
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
    
        # N 1024 320
        # print(hid_states.shape, "hid")
        # seq_hid_states = self.batchnorm_s1(self.act_s1(self.linear_s1(hid_states)))    # N 1024 320 -> N 1024 169
        # print(seq_hid_states.shape)
        # seq_hid_states = self.batchnorm_s2(self.act_s2(self.linear_s2(seq_hid_states)))    # N 1024 4748 -> N 1024 169
        # hid_states = torch.unsqueeze(hid_states, dim=1) # N 1024 -> N 1 1024
        x5 = x5.reshape(-1, 1024, 169)
        x5 = x5.permute(0, 2, 1) # (N, 169, 1024)
        # x5_new = self.batchnorm_sv2(self.act_sv2(self.linear_sv2(torch.cat((x5, seq_hid_states), dim=-1))))
        hid_states = hid_states.unsqueeze(dim=1)
        x5_new, _ = self.linear_sv(x5, hid_states, hid_states)
        x5_new = x5_new.permute(0, 2, 1) 
        x5_new = x5_new.reshape(-1, 1024, 13, 13)    

        x_new = self.up1(x5_new, x4) 
        x_new = self.up2(x_new, x3)
        x_new = self.up3(x_new, x2)
        x_new = self.up4(x_new, x1) 
        logits = self.sigmoid(self.outc(x_new))
        loss = self.loss_fct(logits, y_img)

        return (loss, logits)
        