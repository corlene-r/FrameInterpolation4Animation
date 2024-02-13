import torch
import torch.nn as nn
import torch.nn.functional as F
from model.warplayer import warp

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),        
        nn.PReLU(out_planes)
    )

def conv_bn(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.PReLU(out_planes)
    )

class IFBlock(nn.Module):
    def __init__(self, in_planes, c=64):
        super(IFBlock, self).__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c//2, 3, 2, 1),
            conv(c//2, c, 3, 2, 1),
            )
        self.convblock0 = nn.Sequential(
            conv(c, c),
            conv(c, c)
        )
        self.convblock1 = nn.Sequential(
            conv(c, c),
            conv(c, c)
        )
        self.convblock2 = nn.Sequential(
            conv(c, c),
            conv(c, c)
        )
        self.convblock3 = nn.Sequential(
            conv(c, c),
            conv(c, c)
        )
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(c, c//2, 4, 2, 1),
            nn.PReLU(c//2),
            nn.ConvTranspose2d(c//2, 4, 4, 2, 1),
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(c, c//2, 4, 2, 1),
            nn.PReLU(c//2),
            nn.ConvTranspose2d(c//2, 1, 4, 2, 1),
        )

    def forward(self, x, flow, scale=1):
        x = F.interpolate(x, scale_factor= 1. / scale, mode="bilinear", align_corners=False, recompute_scale_factor=False)
        flow = F.interpolate(flow, scale_factor= 1. / scale, mode="bilinear", align_corners=False, recompute_scale_factor=False) * 1. / scale
        feat = self.conv0(torch.cat((x, flow), 1))
        feat = self.convblock0(feat) + feat
        feat = self.convblock1(feat) + feat
        feat = self.convblock2(feat) + feat
        feat = self.convblock3(feat) + feat        
        flow = self.conv1(feat)
        mask = self.conv2(feat)
        flow = F.interpolate(flow, scale_factor=scale, mode="bilinear", align_corners=False, recompute_scale_factor=False) * scale
        mask = F.interpolate(mask, scale_factor=scale, mode="bilinear", align_corners=False, recompute_scale_factor=False)
        return flow, mask
        
class IFNet(nn.Module):
    def __init__(self):
        super(IFNet, self).__init__()
        self.block0 = IFBlock(7+4, c=90)
        self.block1 = IFBlock(7+4, c=90)
        self.block2 = IFBlock(7+4, c=90)
        self.block_tea = IFBlock(10+4, c=90)
        # self.contextnet = Contextnet()
        # self.unet = Unet()

    def forward(self, x, scale=[4, 2, 1], timestep=0.5, training=False):
        img0 = x[:, :3]
        img1 = x[:, 3:6]
        gt = x[:, 6:] # In inference time, gt is None

        flow_list = []
        merged = []
        mask_list = []
        
        warped_img0 = img0
        warped_img1 = img1
        flow = None 
        loss_distill = 0
        stu = [self.block0, self.block1, self.block2]
        for i in range(3):
            if flow != None:
                flow_d, mask_d = stu[i](torch.cat((img0, img1, warped_img0, warped_img1, mask), 1), flow, scale=scale[i])
                flow = flow + flow_d
                mask = mask + mask_d
            else:
                flow, mask = stu[i](torch.cat((img0, img1), 1), None, scale=scale[i])
            mask_list.append(torch.sigmoid(mask))
            flow_list.append(flow)
            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])
            merged_student = (warped_img0, warped_img1)
            merged.append(merged_student)
        if gt.shape[1] == 3:
            flow_d, mask_d = self.block_tea(torch.cat((img0, img1, warped_img0, warped_img1, mask, gt), 1), flow, scale=1)
            flow_teacher = flow + flow_d
            warped_img0_teacher = warp(img0, flow_teacher[:, :2])
            warped_img1_teacher = warp(img1, flow_teacher[:, 2:4])
            mask_teacher = torch.sigmoid(mask + mask_d)
            merged_teacher = warped_img0_teacher * mask_teacher + warped_img1_teacher * (1 - mask_teacher)
        else:
            flow_teacher = None
            merged_teacher = None
        for i in range(3):
            merged[i] = merged[i][0] * mask_list[i] + merged[i][1] * (1 - mask_list[i])
            if gt.shape[1] == 3:
                loss_mask = ((merged[i] - gt).abs().mean(1, True) > (merged_teacher - gt).abs().mean(1, True) + 0.01).float().detach()
                loss_distill += (((flow_teacher.detach() - flow_list[i]) ** 2).mean(1, True) ** 0.5 * loss_mask).mean()
        c0 = self.contextnet(img0, flow[:, :2])
        c1 = self.contextnet(img1, flow[:, 2:4])
        tmp = self.unet(img0, img1, warped_img0, warped_img1, mask, flow, c0, c1)
        res = tmp[:, :3] * 2 - 1
        merged[2] = torch.clamp(merged[2] + res, 0, 1)
        return flow_list, mask_list[2], merged
