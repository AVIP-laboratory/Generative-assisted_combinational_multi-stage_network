import torch
import torch.nn as nn
import torch.nn.functional as F

# UNet
class UNetDown(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True, dropout=0.0):
        super().__init__()

        layers = [nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False)]

        if normalize:
            layers.append(nn.BatchNorm2d(out_channels)),

        layers.append(nn.PReLU())

        if dropout:
            layers.append(nn.Dropout(dropout))

        self.down = nn.Sequential(*layers)

    def forward(self, x):
        x = self.down(x)
        return x


class UNetDown2(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True,dropout=0.0):
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels,1,1,0,bias=False),
        ]

        if normalize:
            layers.append(nn.BatchNorm2d(out_channels)),

        if dropout:
            layers.append(nn.Dropout(dropout))

        self.up = nn.Sequential(*layers)

    def forward(self,x, skip):
        x = torch.cat((x, skip), 1)
        x = self.up(x)
        return x


class UNetUp31(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()

        layers = [
            nn.ConvTranspose2d(in_channels, out_channels,4,2,1,bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.PReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.PReLU()
        ]

        if dropout:
            layers.append(nn.Dropout(dropout))

        self.up = nn.Sequential(*layers)

    def forward(self,x,skip):
        x = torch.cat((x,skip),1)
        x = self.up(x)
        return x


class UNetUp21(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()

        layers = [
            nn.ConvTranspose2d(in_channels, out_channels,4,2,1,bias=False),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
        ]

        if dropout:
            layers.append(nn.Dropout(dropout))

        self.up = nn.Sequential(*layers)

    def forward(self,x,skip):
        x = torch.cat((x,skip),1)
        x = self.up(x)
        return x


class UNet_only4(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64,64)
        self.down3 = UNetDown(64,128)
        self.down4 = UNetDown(128,128)
        self.down5 = UNetDown(128,256)
        self.down8 = UNetDown(256,256)
        self.down9 = nn.Sequential(
            nn.Conv2d(256,512,3,stride=1,padding=1, bias=False),
            nn.PReLU()
        )

        self.up0 = nn.Sequential(nn.Conv2d(512, 256, 3, 1, 1,bias=False),
            nn.InstanceNorm2d(256),
            nn.PReLU())
        self.up1 = UNetUp31(512,256)
        self.up2 = UNetUp31(512,128)
        self.up5 = UNetUp31(256,128)
        self.up6 = UNetUp31(256,64)
        self.up7 = UNetUp31(128,64)
        self.up9 = UNetUp21(128,3)
        self.last = UNetDown2(6, out_channels, normalize=False)


    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d8 = self.down8(d5)
        d9 = self.down9(d8)
        u0 = self.up0(d9)
        u1 = self.up1(u0,d8)
        u2 = self.up2(u1,d5)
        u5 = self.up5(u2,d4)
        u6 = self.up6(u5,d3)
        u7 = self.up7(u6,d2)
        u9 = self.up9(u7, d1)
        last = self.last(u9, x)
        return last


class UNet_only5(nn.Module):
    def __init__(self, in_channels=6, out_channels=3):
        super().__init__()
        self.down1 = UNetDown(in_channels, 64, normalize=False) #128
        self.down2 = UNetDown(64,64)
        self.down3 = UNetDown(64,128)
        self.down4 = UNetDown(128,128)
        self.down5 = UNetDown(128,256)
        self.down8 = UNetDown(256,256)
        self.down9 = nn.Sequential(
            nn.Conv2d(256,512,3,stride=1,padding=1, bias=False),
            nn.PReLU()
        ) #2

        self.up0 = nn.Sequential(nn.Conv2d(512, 256,3,1,1,bias=False),
            nn.InstanceNorm2d(256),
            nn.PReLU())
        self.up1 = UNetUp31(512,256)
        self.up2 = UNetUp31(512,128)
        self.up5 = UNetUp31(256,128)
        self.up6 = UNetUp31(256,64)
        self.up7 = UNetUp31(128,64)
        self.up9 = UNetUp21(128,3)
        self.last = UNetDown2(6, out_channels, normalize=False)


    def forward(self, x, y):
        inputs = torch.cat((x, y), 1)
        d1 = self.down1(inputs)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d8 = self.down8(d5)
        d9 = self.down9(d8)
        u0 = self.up0(d9)
        u1 = self.up1(u0,d8)
        u2 = self.up2(u1,d5)
        u5 = self.up5(u2,d4)
        u6 = self.up6(u5,d3)
        u7 = self.up7(u6,d2)
        u9 = self.up9(u7, d1)
        last = self.last(u9, x)
        return last


class Dis_block(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True):
        super().__init__()

        layers = [nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.PReLU())

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        x = self.block(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        self.stage_1 = Dis_block(in_channels*2,64,normalize=False)
        self.stage_2 = Dis_block(64,128,normalize=False)
        self.stage_5 = nn.Conv2d(128, 256, 3, 1, padding=0)
        self.stage_6 = nn.Conv2d(256, 512, 3, 1, padding=0)
        self.stage_7 = nn.Conv2d(512, 512, 3, 1, padding=0)
        self.A1 = nn.PReLU()
        self.patch = nn.Conv2d(512,1,4,padding=1) # 16x16 패치 생성

    def forward(self,a,b):
        x = torch.cat((a,b),1)
        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.A1(x)
        x = self.stage_5(x)
        x = self.A1(x)
        x = self.stage_6(x)
        x = self.A1(x)
        x = self.stage_7(x)
        x = self.A1(x)
        x = self.stage_7(x)
        x = self.A1(x)
        x = self.patch(x)
        x = torch.sigmoid(x)
        return x