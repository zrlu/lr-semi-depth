import argparse
import time
import torch
import numpy as np
import torch.optim as optim
from networks import Resnet50Encoder, Resnet50Decoder, Discriminator, fusion as fusion_func
from loss import FcRecLoss, FcGANLoss, FcConLoss
from torch.utils.data import DataLoader, ConcatDataset
from dataset import KITTI, prepare_dataloader
from os import path, makedirs
from utils import warp_left, warp_right, to_device, scale_pyramid
import numpy as np
from itertools import chain

class Model:

    def __init__(self, batch_size=4, input_channels=3, use_multiple_gpu=False,
                       g_learning_rate=1e-4, d_learning_rate=1e-4,
                       model_path='model', device='cuda:0', mode='train', train_dataset_dir='data_scene_flow/training', 
                       val_dataset_dir='data_scene_flow/testing', num_workers=4, do_augmentation=True,
                       output_directory='outputs',
                       input_height=256, input_width=512, augment_parameters=[0.8, 1.2, 0.5, 2.0, 0.8, 1.2]):

        self.batch_size = batch_size
        self.input_channels = input_channels
        self.model_path = model_path
        self.device = device
        self.use_multiple_gpu = use_multiple_gpu

        self.e_L = Resnet50Encoder().to(self.device)
        self.e_R = Resnet50Encoder().to(self.device)
        self.g_LL = Resnet50Decoder().to(self.device)
        self.g_RL = Resnet50Decoder().to(self.device)
        self.g_LR = Resnet50Decoder().to(self.device)
        self.g_RR = Resnet50Decoder().to(self.device)
        self.d_L = Discriminator(self.input_channels).to(self.device)
        self.d_R = Discriminator(self.input_channels).to(self.device)

        if self.use_multiple_gpu:
            self.e_L = torch.nn.DataParallel(self.e_L)
            self.e_R = torch.nn.DataParallel(self.e_R)
            self.g_LL = torch.nn.DataParallel(self.g_LL)
            self.g_RL = torch.nn.DataParallel(self.g_RL)
            self.g_LR = torch.nn.DataParallel(self.g_LR)
            self.g_RR = torch.nn.DataParallel(self.g_RR)
            self.d_L = torch.nn.DataParallel(self.d_L)
            self.d_R = torch.nn.DataParallel(self.d_R)

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
        self.fusion = fusion_func().to(self.device)

        if self.mode == 'train':
            self.criterion_rec = FcRecLoss().to(self.device)
            self.criterion_GAN = FcGANLoss().to(self.device)
            self.criterion_con = FcConLoss().to(self.device)
            
            self.g_optimizer = optim.Adam(
                chain(
                    self.e_L.parameters(), self.e_R.parameters(),
                    self.g_LL.parameters(), self.g_RL.parameters(),
                    self.g_LR.parameters(), self.g_RR.parameters()), 
                lr=self.g_learning_rate)
            self.d_optimizer = optim.Adam(chain(self.d_L.parameters(), self.d_R.parameters()), lr=self.d_learning_rate)
            self.val_n_img, self.val_loader = prepare_dataloader(self.val_dataset_dir, self.mode, self.augment_parameters,
                                                False, 1,
                                                (self.input_height, self.input_width),
                                                self.num_workers)
        else:
            self.augment_parameters = None
            self.do_augmentation = False

        self.n_img, self.loader = prepare_dataloader(self.train_dataset_dir, self.mode,
                                                    self.augment_parameters,
                                                    self.do_augmentation, self.batch_size,
                                                    (self.input_height, self.input_width),
                                                    self.num_workers)
        self.output_directory = output_directory
        if 'cuda' in self.device:
            torch.cuda.synchronize()


    def compute_d_loss(self):
        d_loss_GAN = self.criterion_GAN(self.images_L, self.images_L_est, self.images_R, self.images_R_est)
        d_loss = 0.1*d_loss_GAN
        return d_loss

    
    def compute_g_loss(self):
        g_loss_rec = self.criterion_rec(self.images_L, self.images_L_est, self.images_R, self.images_R_est)
        g_loss_con = self.criterion_con(self.disps_L, self.disps_R)
        g_loss = g_loss_rec + 0.1*g_loss_con
        return g_loss   


    def compute(self, left, right):
        image_LL_enc = self.e_L(left)
        image_LR_enc = self.e_L(right)

        self.images_L = scale_pyramid(left)
        self.images_R = scale_pyramid(right)

        disps_RL = self.g_LL(image_LL_enc)
        disps_RR = self.g_LR(image_LR_enc)
        self.disps_R = self.fusion(disps_RL, disps_RR)

        self.images_R_est = [warp_right(self.images_L[i], self.disps_R[i]) for i in range(4)]
        
        images_RL_enc = self.e_R(left)
        images_RR_enc = self.e_R(right)

        disps_LL = self.g_RL(images_RL_enc)
        disps_LR = self.g_RR(images_RR_enc)
        self.disps_L = self.fusion(disps_LL, disps_LR)

        self.images_L_est = [warp_left(self.images_R_est[i], self.disps_L[i]) for i in range(4)]
        

    def train(self, epoch):

        c_time = time.time()
        g_running_loss = 0.0
        d_running_loss = 0.0

        self.d_L.train()
        self.d_R.train()

        for data in self.loader:
            data = to_device(data, self.device)
            left, right= data['left_image'], data['right_image']

            self.d_optimizer.zero_grad()
            self.compute(left, right)
            d_loss = self.compute_d_loss()
            d_loss.backward()
            d_running_loss += d_loss.item()
            self.d_optimizer.step()
        
        self.e_L.train()
        self.e_R.train()
        self.g_LL.train()
        self.g_RL.train()
        self.g_LR.train()
        self.g_RR.train()

        for data in self.loader:
            data = to_device(data, self.device)
            left, right= data['left_image'], data['right_image']

            self.g_optimizer.zero_grad()
            self.compute(left, right)
            g_loss = self.compute_g_loss()
            g_loss.backward()
            g_running_loss += g_loss.item()
            self.g_optimizer.step()

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
            torch.save(self.e_L.state_dict(), self.path_for('el.nn.best'))
            torch.save(self.e_R.state_dict(), self.path_for('er.nn.best'))
            torch.save(self.g_LL.state_dict(), self.path_for('gll.nn.best'))
            torch.save(self.g_RL.state_dict(), self.path_for('grl.nn.best'))
            torch.save(self.g_LR.state_dict(), self.path_for('glr.nn.best'))
            torch.save(self.g_RR.state_dict(), self.path_for('grr.nn.best'))
            torch.save(self.d_L.state_dict(), self.path_for('dl.nn.best'))
            torch.save(self.d_R.state_dict(), self.path_for('dr.nn.best'))
        else:
            torch.save(self.e_L.state_dict(), self.path_for('el.nn'))
            torch.save(self.e_R.state_dict(), self.path_for('er.nn'))
            torch.save(self.g_LL.state_dict(), self.path_for('gll.nn'))
            torch.save(self.g_RL.state_dict(), self.path_for('grl.nn'))
            torch.save(self.g_LR.state_dict(), self.path_for('glr.nn'))
            torch.save(self.g_RR.state_dict(), self.path_for('grr.nn'))
            torch.save(self.d_L.state_dict(), self.path_for('dl.nn'))
            torch.save(self.d_R.state_dict(), self.path_for('dr.nn'))

    def load(self, best=False):
        print('load', 'best', best)
        if best:
            self.e_L.load_state_dict(torch.load(self.path_for('el.nn.best')))
            self.e_R.load_state_dict(torch.load(self.path_for('er.nn.best')))
            self.g_LL.load_state_dict(torch.load(self.path_for('gll.nn.best')))
            self.g_RL.load_state_dict(torch.load(self.path_for('grl.nn.best')))
            self.g_LR.load_state_dict(torch.load(self.path_for('glr.nn.best')))
            self.g_RR.load_state_dict(torch.load(self.path_for('grr.nn.best')))
            self.d_L.load_state_dict(torch.load(self.path_for('dl.nn.best')))
            self.d_R.load_state_dict(torch.load(self.path_for('dr.nn.best')))
        else:
            self.e_L.load_state_dict(torch.load(self.path_for('el.nn')))
            self.e_R.load_state_dict(torch.load(self.path_for('er.nn')))
            self.g_LL.load_state_dict(torch.load(self.path_for('gll.nn')))
            self.g_RL.load_state_dict(torch.load(self.path_for('grl.nn')))
            self.g_LR.load_state_dict(torch.load(self.path_for('glr.nn')))
            self.g_RR.load_state_dict(torch.load(self.path_for('grr.nn')))
            self.d_L.load_state_dict(torch.load(self.path_for('dl.nn')))
            self.d_R.load_state_dict(torch.load(self.path_for('dr.nn')))
    

    def combine_2_disps(self, disp_L, disp_R):
        return 0.5*(disp_L + warp_left(disp_R, disp_L))


    def test(self):

        self.e_L.eval()
        self.e_R.eval()
        self.g_LL.eval()
        self.g_RL.eval()
        self.g_LR.eval()
        self.g_RR.eval()

        g_running_val_loss = 0.0
        d_running_val_loss = 0.0

        disparities = np.zeros((self.n_img, self.input_height, self.input_width), dtype=np.float32)
        disparities_L = np.zeros((self.n_img, self.input_height, self.input_width), dtype=np.float32)
        disparities_R = np.zeros((self.n_img, self.input_height, self.input_width), dtype=np.float32)

        ref_img = None

        with torch.no_grad():
            for i, data in enumerate(self.val_loader):
                
                data = to_device(data, self.device)
                left, right= data['left_image'], data['right_image']
                if i == 0:
                    ref_img = left[0].cpu().numpy()

                self.compute(left, right)
                g_loss = self.compute_g_loss()
                d_loss = self.compute_d_loss()
                d_running_val_loss += d_loss.item()
                g_running_val_loss += g_loss.item()

                disp_L = self.disps_L[0]
                disp_R = self.disps_R[0]

                disp = self.combine_2_disps(disp_L, disp_R).cpu().numpy()[:, 0, :, :]
                disp_L = disp_L.cpu().numpy()[:, 0, :, :]
                disp_R = disp_R.cpu().numpy()[:, 0, :, :]

                disparities[i] = disp
                disparities_L[i] = disp_L
                disparities_R[i] = disp_R

            g_running_val_loss /= self.val_n_img
            d_running_val_loss /= self.val_n_img

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
        
        return disparities, disparities_L, disparities_R, ref_img
