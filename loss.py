import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import scale_pyramid, warp_left, warp_right


class Loss(nn.modules.Module):
    def __init__(self, device='cuda:0'):
        super(Loss, self).__init__()
        self.device = device
    
    
    def SSIM(self, x, y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = nn.AvgPool2d(3, 1)(x)
        mu_y = nn.AvgPool2d(3, 1)(y)
        mu_x_mu_y = mu_x * mu_y
        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)

        sigma_x = nn.AvgPool2d(3, 1)(x * x) - mu_x_sq
        sigma_y = nn.AvgPool2d(3, 1)(y * y) - mu_y_sq
        sigma_xy = nn.AvgPool2d(3, 1)(x * y) - mu_x_mu_y

        SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
        SSIM = SSIM_n / SSIM_d

        return torch.clamp((1 - SSIM) / 2, 0, 1)


    def unsup_loss(self, img, img_est):
        SSIM = torch.mean(self.SSIM(img, img_est))
        l1 = torch.mean(torch.abs(img - img_est))
        alpha = 0.85
        return (1 - alpha) * l1 + alpha * SSIM


    def gradient_x(self, img):
        img = F.pad(img, (0, 1, 0, 0), mode="replicate")
        gx = img[:, :, :, :-1] - img[:, :, :, 1:]  # NCHW
        return gx
    

    def gradient_y(self, img):
        img = F.pad(img, (0, 0, 0, 1), mode="replicate")
        gy = img[:, :, :-1, :] - img[:, :, 1:, :]  # NCHW
        return gy


    def reg_loss(self, img, disp):
        eta = 1.0/255
        disp_grad_x = self.gradient_x(disp)
        disp_grad_y = self.gradient_y(disp)
        img_grad_x = self.gradient_x(img)
        img_grad_y = self.gradient_y(img)
        weights_x = torch.exp(-eta*torch.abs(img_grad_x))
        weights_y = torch.exp(-eta*torch.abs(img_grad_y))
        loss_x = torch.mean(torch.abs(weights_x * disp_grad_x))
        loss_y = torch.mean(torch.abs(weights_y * disp_grad_y))
        return (loss_x + loss_y).mean()


    def sup_loss(self, disp_est, depth_gt):
        _, _, h, w = depth_gt.shape
        disp_est = nn.functional.interpolate(disp_est, size=(h, w), mode='bilinear', align_corners=True)
        depth_est = 359.7176277195809831 * 0.54 / disp_est
        masks = depth_gt > 0
        absdiff = masks * torch.abs(depth_est - depth_gt)
        delta = 0.2*torch.max(absdiff).item()
        return torch.where(
                absdiff < delta,
                absdiff,
                (absdiff*absdiff+delta*delta)/(2*delta)
        ).mean()
    

    def lr_con_loss(self, disps_est):
        disp_est_left_warp = warp_left(disps_est[1], disps_est[0])
        disp_est_right_warp = warp_right(disps_est[0], disps_est[1])
        loss_left_right = torch.mean(torch.abs(disp_est_left_warp - disps_est[0]))
        loss_right_left = torch.mean(torch.abs(disp_est_right_warp - disps_est[1]))
        return loss_left_right + loss_right_left

    
    def forward(self, x, epoch, weights):
        w1, w2, w3, w4 = weights
        imgs, imgs_est, disps_est, depths_gt = x
        unsup_loss = torch.Tensor([0.0]).to(self.device)
        if w1 > 0.0:
            unsup_loss = sum([self.unsup_loss(imgs[0][i], imgs_est[0][i]) + self.unsup_loss(imgs[1][i], imgs_est[1][i]) for i in range(4)])
        sup_loss = torch.Tensor([0.0]).to(self.device)
        if w2 > 0.0:
            sup_loss = sum([self.sup_loss(disps_est[0][i], depths_gt[0]) + self.sup_loss(disps_est[1][i], depths_gt[1]) for i in range(4)])
        reg_loss = torch.Tensor([0.0]).to(self.device)
        if w3 > 0.0:
            reg_loss = sum([self.reg_loss(imgs[0][i], disps_est[0][i]) + self.reg_loss(imgs[1][i], disps_est[1][i]) for i in range(4)])
        lr_con_loss = torch.Tensor([0.0]).to(self.device)
        if w4 > 0.0:
            lr_con_loss = sum([self.lr_con_loss([disps_est[0][i], disps_est[1][i]]) for i in range(4)])
        # fade_in_term = torch.exp(-torch.Tensor([30.]).to(self.device)/(epoch+1))
        return w1*unsup_loss, w2*sup_loss, w3*reg_loss, w4*lr_con_loss