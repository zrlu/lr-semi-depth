import argparse
import time
import torch
import numpy as np
import torch.optim as optim
import skimage.transform
from torchvision import models
from networks import Resnet50, Discriminator
from loss import MonodepthLoss, GANLoss
from torch.utils.data import DataLoader, ConcatDataset
from dataset import KITTI, prepare_dataloader
from os import path, makedirs
from utils import warp, post_process_disparity, to_device
import numpy as np

class Model:

    def __init__(self, batch_size=4, input_channels=3,
                       g_learning_rate=1e-4, d_learning_rate=1e-4,
                       model_path='model', device='cuda:0', mode='train', train_dataset_dir='data_scene_flow/training', 
                       val_dataset_dir='data_scene_flow/testing', num_workers=4, do_augmentation=True,
                       output_directory='outputs',
                       input_height=256, input_width=512, augment_parameters=[0.8, 1.2, 0.5, 2.0, 0.8, 1.2]):
        self.batch_size = batch_size
        self.input_channels = input_channels
        self.model_path = model_path
        self.device = device
        self.g = Resnet50(self.input_channels).to(self.device)
        self.d = Discriminator(self.input_channels).to(self.device)
        self.g_learning_rate=g_learning_rate
        self.d_learning_rate=d_learning_rate
        self.mode = mode
        self.input_height = input_height
        self.input_width = input_width
        self.augment_parameters = augment_parameters
        self.train_dataset_dir = train_dataset_dir
        self.val_dataset_dir = val_dataset_dir
        self.g_best_val_loss = float('inf')
        self.num_workers = num_workers
        self.do_augmentation = do_augmentation

        if self.mode == 'train':
            self.criterion = MonodepthLoss(
                n=4,
                SSIM_w=0.85,
                disp_gradient_w=0.1, lr_w=1).to(self.device)
            self.g_optimizer = optim.Adam(self.g.parameters(), lr=self.g_learning_rate)
            self.d_optimizer = optim.Adam(self.d.parameters(), lr=self.d_learning_rate)
            self.val_n_img, self.val_loader = prepare_dataloader(self.val_dataset_dir, self.mode, self.augment_parameters,
                                                False, self.batch_size,
                                                (self.input_height, self.input_width),
                                                self.num_workers)
            self.criterion_GAN = GANLoss().to(self.device)
        else:
            self.augment_parameters = None

        self.n_img, self.loader = prepare_dataloader(self.train_dataset_dir, self.mode,
                                                    self.augment_parameters,
                                                    self.do_augmentation, self.batch_size,
                                                    (self.input_height, self.input_width),
                                                    self.num_workers)
        self.output_directory = output_directory
        if 'cuda' in self.device:
            torch.cuda.synchronize()


    def train(self, epoch):

        c_time = time.time()
        g_running_loss = 0.0
        d_running_loss = 0.0

        self.g.train()
        for data in self.loader:
            data = to_device(data, self.device)
            left = data['left_image']
            right = data['right_image']

            disps = self.g(left)
            self.g_optimizer.zero_grad()
            loss_data = self.criterion(disps, [left, right])
            fake_right = warp(left, disps[0])
            pred_fake_right = self.d(fake_right)
            loss_fake_right = self.criterion_GAN(pred_fake_right, True)
            g_loss = loss_data + loss_fake_right
            g_loss.backward()
            g_running_loss += loss_data.item()
            self.g_optimizer.step()

        self.d.train()
        for data in self.loader:
            data = to_device(data, self.device)
            left = data['left_image']
            right = data['right_image']

            self.d_optimizer.zero_grad()
            disps = self.g(left)
            fake_right = warp(left, disps[0])
            pred_fake_right = self.d(fake_right)
            pred_real_left  = self.d(left.detach())
            loss_fake_right = self.criterion_GAN(pred_fake_right, False)
            loss_real_left  = self.criterion_GAN(pred_real_left, True)
            d_loss = 0.5*(loss_fake_right + loss_real_left)
            d_loss.backward()
            d_running_loss += d_loss.item()
            self.d_optimizer.step()
        
        g_running_loss /= self.n_img / self.batch_size
        d_running_loss /= self.n_img / self.batch_size
        
        self.save()
        print(
            'Epoch: {}'.format(str(epoch).rjust(3, ' ')),
            'G: {:.3f}'.format(g_running_loss),
            'D: {:.3f}'.format(d_running_loss),
            'Time: {:.2f}s'.format(time.time() - c_time)
        )


    def path_for(self, fn):
        return path.join(self.model_path, fn)

    def save(self, best=False):
        if not path.exists(self.model_path):
            makedirs(self.model_path, exist_ok=True)
        if best:
            torch.save(self.g.state_dict(), self.path_for('g.nn.best'))
            torch.save(self.d.state_dict(), self.path_for('d.nn.best'))
        else:
            torch.save(self.g.state_dict(), self.path_for('g.nn'))
            torch.save(self.d.state_dict(), self.path_for('d.nn'))

    def load(self, best=False):
        print('load', 'best', best)
        if best:
            self.g.load_state_dict(torch.load(self.path_for('g.nn.best')))
            self.d.load_state_dict(torch.load(self.path_for('d.nn.best')))
        else:
            self.g.load_state_dict(torch.load(self.path_for('g.nn')))
            self.d.load_state_dict(torch.load(self.path_for('d.nn')))

    def test(self):
        self.g.eval()

        g_running_val_loss = 0.0
        d_running_val_loss = 0.0

        disparities = np.zeros((self.n_img, self.input_height, self.input_width), dtype=np.float32)
        disparities_pp = np.zeros((self.n_img, self.input_height, self.input_width), dtype=np.float32)

        ref_img = None

        with torch.no_grad():
            for i, data in enumerate(self.val_loader):
                
                data = to_device(data, self.device)
                left = data['left_image']
                if i == 0:
                    ref_img = left[0].cpu().numpy()

                right = data['right_image']
                disps = self.g(left)
                g_val_loss = self.criterion(disps, [left, right])
                g_running_val_loss += g_val_loss.item()

                pred_fake_right = self.d(warp(left, disps[0]))
                pred_real_left  = self.d(left.detach())
                loss_fake_right = self.criterion_GAN(pred_fake_right, False)
                loss_real_left  = self.criterion_GAN(pred_real_left, True)
                d_val_loss = 0.5*(loss_fake_right + loss_real_left)
                d_running_val_loss += d_val_loss.item()

                disp = disps[0][:, 0, :, :].unsqueeze(1)
                disparities[i] = disp[0].squeeze().cpu().numpy()
                disparities_pp[i] =  post_process_disparity(disps[0][:, 0, :, :].cpu().numpy())

            g_running_val_loss /= self.val_n_img / self.batch_size

            model_saved = '[*]'
            if g_running_val_loss < self.g_best_val_loss:
                self.save(True)
                self.g_best_val_loss = g_running_val_loss
                model_saved = '[S]'
            print(
                '      Test',
                'G: {:.3f}({:.3f})'.format(g_running_val_loss, self.g_best_val_loss),
                'D: {:.3f}'.format(d_running_val_loss),
                model_saved
            )
        
        return disparities, disparities_pp, ref_img
