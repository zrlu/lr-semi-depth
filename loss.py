import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import scale_pyramid, warp_left, warp_right


class FcRecLoss(nn.Module):
    '''
        Full-cycle reconstruction loss
    '''
    def __init__(self):
        super(FcRecLoss, self).__init__()

    def forward(self, images_R, images_R_est):
        l1_LR = [torch.mean(torch.abs(images_R[i] - images_R_est[i])) for i in range(4)]
        # l1_RL = [torch.mean(torch.abs(images_L[i] - images_L_est[i])) for i in range(4)]
        # return sum(l1_LR + l1_RL)
        # return sum(l1_LR + l1_RL)
        return sum(l1_LR)


class FcConLoss(nn.Module):
    '''
        Full-cycle consistency loss
    '''
    def __init__(self):
        super(FcConLoss, self).__init__()

    def forward(self, disps_L, disps_R):
        l1_L = [torch.mean(torch.abs(disps_L[i] - warp_left(disps_R[i], disps_L[i]))) for i in range(4)]
        l1_R = [torch.mean(torch.abs(disps_R[i] - warp_right(disps_L[i], disps_R[i]))) for i in range(4)]
        return sum(l1_L + l1_R)


class FcGANLoss(nn.Module):
    '''
        Full-cycle GAN loss
    '''
    def __init__(self):
        super(FcGANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(1.0))
        self.register_buffer('fake_label', torch.tensor(0.0))
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, images_R, images_R_est):
        # loss_L_real = [self.loss(images_L[i], self.real_label.expand_as(images_L[i])) for i in range(4)]
        # loss_L_fake = [self.loss(images_L_est[i], self.fake_label.expand_as(images_L_est[i])) for i in range(4)]
        loss_R_real = [self.loss(images_R[i], self.real_label.expand_as(images_R[i])) for i in range(4)]
        loss_R_fake = [self.loss(images_R_est[i], self.fake_label.expand_as(images_R_est[i])) for i in range(4)]
        # return sum(loss_L_real + loss_L_fake + loss_R_real + loss_R_fake)
        return sum(loss_R_real + loss_R_fake)
