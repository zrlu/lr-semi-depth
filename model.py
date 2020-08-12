import argparse
import time
import torch
import numpy as np
import torch.optim as optim
from networks import Resnet50_md, Discriminator
from loss import MonodepthLoss
from torch.utils.data import DataLoader, ConcatDataset
from dataset import KITTI, prepare_dataloader
from os import path, makedirs
from utils import warp_left, warp_right, to_device, scale_pyramid, adjust_learning_rate
import numpy as np
from itertools import chain

class Model:

    def __init__(self, batch_size=4, input_channels=3, use_multiple_gpu=False,
                       learning_rate=1e-4,
                       model_path='model', device='cuda:0', train_dataset_dir='data_scene_flow/training', 
                       val_dataset_dir='data_scene_flow/testing', num_workers=4, do_augmentation=False,
                       output_directory='outputs',
                       input_height=256, input_width=512, augment_parameters=[0.8, 1.2, 0.5, 2.0, 0.8, 1.2]):

        self.batch_size = batch_size
        self.input_channels = input_channels
        self.model_path = model_path
        self.device = device
        self.use_multiple_gpu = use_multiple_gpu

        self.net = Resnet50_md(self.input_channels).to(self.device)

        if self.use_multiple_gpu:
            self.net = torch.nn.DataParallel(self.net)

        self.learning_rate=learning_rate
        self.input_height = input_height
        self.input_width = input_width
        self.augment_parameters = augment_parameters
        self.train_dataset_dir = train_dataset_dir
        self.val_dataset_dir = val_dataset_dir
        self.best_val_loss = float('inf')
        self.num_workers = num_workers
        self.do_augmentation = do_augmentation

        self.criterion_mono = MonodepthLoss()
        
        self.optimizer = optim.Adam(
            chain(
                self.net.parameters()
            ),
            lr=self.learning_rate
        )
        self.val_n_img, self.val_loader = prepare_dataloader(self.val_dataset_dir, self.augment_parameters,
                                            False, self.batch_size,
                                            (self.input_height, self.input_width),
                                            self.num_workers, shuffle=False)

        self.n_img, self.loader = prepare_dataloader(self.train_dataset_dir,
                                                    self.augment_parameters,
                                                    self.do_augmentation, self.batch_size,
                                                    (self.input_height, self.input_width),
                                                    self.num_workers)
        self.output_directory = output_directory
        if 'cuda' in self.device:
            torch.cuda.synchronize()
    
    def compute_loss(self):
        loss_mono = self.criterion_mono(
            (self.disps_RLL_est, self.disps_RLL_est),
            (self.images_L, self.images_R)
        )
        loss = loss_mono
        return loss   


    def compute(self, left, right):
        self.images_L = scale_pyramid(left)
        self.images_R = scale_pyramid(right)
        self.disps_RL = self.net(left)

        self.disps_RLR_est = [d[:, 1, :, :].unsqueeze(1) for d in self.disps_RL]
        self.disps_RLL_est = [d[:, 0, :, :].unsqueeze(1) for d in self.disps_RL]

        self.images_R_est = [warp_right(self.images_L[i], self.disps_RLR_est[i]) for i in range(4)]
        self.images_L_est = [warp_right(self.images_L[i], self.disps_RLR_est[i]) for i in range(4)]
        

    def train(self, epoch):

        adjust_learning_rate(self.optimizer, epoch, self.learning_rate)
        
        c_time = time.time()
        running_loss = 0.0
        
        self.net.train()

        for data in self.loader:
            data = to_device(data, self.device)
            left, right= data['left_image'], data['right_image']

            self.optimizer.zero_grad()
            self.compute(left, right)
            loss = self.compute_loss()
            loss.backward()
            running_loss += loss.item()
            self.optimizer.step()

        running_loss /= self.n_img / self.batch_size
        
        self.save()
        print(
            'Epoch: {}'.format(str(epoch).rjust(3, ' ')),
            'G: {:.3f}'.format(running_loss),
            'Time: {:.2f}s'.format(time.time() - c_time)
        )


    def path_for(self, fn):
        return path.join(self.model_path, fn)

    def save(self, best=False):
        if not path.exists(self.model_path):
            makedirs(self.model_path, exist_ok=True)
        if best:
            torch.save(self.net.state_dict(), self.path_for('net.nn.best'))
        else:
            torch.save(self.net.state_dict(), self.path_for('net.nn'))

    def load(self, best=False):
        print('load', 'best', best)
        if best:
            self.net.load_state_dict(torch.load(self.path_for('net.nn.best')))
        else:

            self.net.load_state_dict(torch.load(self.path_for('net.nn')))

    def test(self):

        c_time = time.time()

        self.net.eval()

        running_val_loss = 0.0

        disparities_R = np.zeros((self.n_img, self.input_height, self.input_width), dtype=np.float32)
        images_L = np.zeros((self.n_img, 3, self.input_height, self.input_width), dtype=np.float32)
        images_R = np.zeros((self.n_img, 3, self.input_height, self.input_width), dtype=np.float32)
        images_est_R = np.zeros((self.n_img, 3, self.input_height, self.input_width), dtype=np.float32)

        ref_img = None

        # RMSE = 0.0
        # RMSE_log = 0.0
        # AbsRel = 0.0
        # SqrRel = 0.0

        with torch.no_grad():
            for i, data in enumerate(self.val_loader):
                
                data = to_device(data, self.device)
                left, right= data['left_image'], data['right_image']

                self.compute(left, right)
                loss = self.compute_loss()
                running_val_loss += loss.item()

                DR = self.disps_RLR_est[0].cpu().numpy()[:, 0, :, :]
                ndata, _, _ = DR.shape
                disparities_R[i*self.batch_size:i*self.batch_size+ndata] = DR
                images_est_R[i*self.batch_size:i*self.batch_size+ndata] = warp_right(left, self.disps_RLR_est[0]).cpu().numpy()
                images_L[i*self.batch_size:i*self.batch_size+ndata] = left.cpu().numpy()
                images_R[i*self.batch_size:i*self.batch_size+ndata] = right.cpu().numpy()

                # RMSE += torch.sqrt(torch.mean((left - warp_right(left, self.disps_RLR_est[0]))**2, dim=[2,3])).sum(dim=[0,1])
                # RMSE_log += torch.sqrt(torch.mean((torch.log(left) - torch.log(warp_right(left, self.disps_RLR_est[0])))**2, dim=[2,3])).sum(dim=[0,1])
                # AbsRel += torch.mean(torch.abs(left - warp_right(left, self.disps_RLR_est[0])) / left, dim=[2,3]).sum(dim=[0,1])
                # SqrRel += torch.mean((left - warp_right(left, self.disps_RLR_est[0])**2) / left, dim=[2,3]).sum(dim=[0,1])

            # RMSE /= self.val_n_img / self.batch_size
            # RMSE_log /= self.val_n_img / self.batch_size
            # AbsRel /= self.val_n_img / self.batch_size
            # SqrRel /= self.val_n_img / self.batch_size
            # print('RMSE', RMSE)
            # print('RMSE_log', RMSE_log)
            # print('AbsRel', AbsRel)
            # print('SqrRel', SqrRel)


            running_val_loss /= self.val_n_img / self.batch_size

            model_saved = '[*]'
            if running_val_loss < self.best_val_loss:
                self.save(True)
                self.best_val_loss = running_val_loss
                model_saved = '[S]'
            print(
                '      Test',
                'G: {:.3f}({:.3f})'.format(running_val_loss, self.best_val_loss),
                'Time: {:.2f}s'.format(time.time() - c_time),
                model_saved
            )
        
        return disparities_R, images_L, images_R, images_est_R
