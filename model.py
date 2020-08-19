import argparse
import time
import torch
import numpy as np
import torch.optim as optim
from networks import Resnet50_md
from loss import Loss
from torch.utils.data import DataLoader, ConcatDataset
from dataset import KITTI, prepare_dataloader
from os import path, makedirs, listdir
from utils import warp_left, warp_right, to_device, scale_pyramid
import numpy as np
from itertools import chain
from torchvision import transforms
from PIL import Image
import cv2

class Model:

    def __init__(self, batch_size=4, input_channels=3, use_multiple_gpu=False,
                       learning_rate=1e-4,
                       loss_weights=[1.0,1.0,1.0,1.0],
                       model_path='model', device='cuda:0', train_dataset_dir='data_scene_flow/training', 
                       val_dataset_dir='data_scene_flow/testing', num_workers=4, do_augmentation=False,
                       output_directory='outputs',
                       input_height=256, input_width=512, augment_parameters=[0.8, 1.2, 0.5, 2.0, 0.8, 1.2]):

        self.loss_weights = loss_weights
        self.batch_size = batch_size
        self.input_channels = input_channels
        self.model_path = model_path
        self.device = device
        self.use_multiple_gpu = use_multiple_gpu

        self.net0 = Resnet50_md(3).to(self.device)
        self.net1 = Resnet50_md(3).to(self.device)
        self.nets = [self.net0, self.net1]
        self.net_names = ['net0', 'net1']

        if self.use_multiple_gpu:
            self.net0 = torch.nn.DataParallel(self.net1)
            self.net1 = torch.nn.DataParallel(self.net1)

        self.learning_rate=learning_rate
        self.input_height = input_height
        self.input_width = input_width
        self.augment_parameters = augment_parameters
        self.train_dataset_dir = train_dataset_dir
        self.val_dataset_dir = val_dataset_dir
        self.best_val_loss = float('inf')
        self.num_workers = num_workers
        self.do_augmentation = do_augmentation

        self.criterion = Loss().to(self.device)
        
        self.optimizer = optim.Adam(chain(self.net0.parameters(), self.net0.parameters()), lr=self.learning_rate)
        self.val_n_img, self.val_loader = prepare_dataloader(self.val_dataset_dir,
                                            self.augment_parameters,
                                            False, self.batch_size,
                                            (self.input_height, self.input_width),
                                            self.num_workers, shuffle=False, drop_last=True)

        self.n_img, self.loader = prepare_dataloader(self.train_dataset_dir,
                                            self.augment_parameters,
                                            self.do_augmentation, self.batch_size,
                                            (self.input_height, self.input_width),
                                            self.num_workers, drop_last=True)
        self.output_directory = output_directory
        if 'cuda' in self.device:
            torch.cuda.synchronize()


    def compute(self, data):
        img_left, img_right, depth_gt_left, depth_gt_right = data
        img_pyramid_left = scale_pyramid(img_left)
        img_pyramid_right = scale_pyramid(img_right)
        disp_pyramid_est_left = self.net0(img_left)
        disp_pyramid_est_right = self.net1(img_right)
        disp_est_left = disp_pyramid_est_left[0]
        disp_est_right = disp_pyramid_est_right[0]
        disps_est = [disp_est_left, disp_est_right]
        img_est_left = warp_left(img_right, disp_est_left)
        img_est_right = warp_right(img_left, disp_est_right)
        img_est_pyramid_left = [warp_left(img_pyramid_right[i], disp_pyramid_est_left[i]) for i in range(4)]
        img_est_pyramid_right = [warp_right(img_pyramid_left[i], disp_pyramid_est_right[i]) for i in range(4)]

        img_pyramids = [img_pyramid_left, img_pyramid_right]
        img_est_pyramids = [img_est_pyramid_left, img_est_pyramid_right]
        disp_est_pyramids = [disp_pyramid_est_left, disp_pyramid_est_right]
        depths_gt = [depth_gt_left, depth_gt_right]

        return [img_pyramids, img_est_pyramids, disp_est_pyramids, depths_gt]
        

    def train(self, epoch):
        
        c_time = time.time()
        running_losses = np.zeros(4)

        self.net0.train()
        self.net1.train()

        for data in self.loader:
            data = to_device(data, self.device)
            self.optimizer.zero_grad()
            out = self.compute(data)
            losses = self.criterion(out, epoch, self.loss_weights)
            loss = sum(losses)
            loss.backward()
            running_losses += np.array([l.item() for l in losses])
            self.optimizer.step()

        running_losses /= self.n_img / self.batch_size
        
        self.save()
        print(
            'Epoch: {}'.format(str(epoch).rjust(3, ' ')),
            'G: {}'.format(running_losses),
            'Time: {:.2f}s'.format(time.time() - c_time)
        )


    def path_for(self, fn):
        return path.join(self.model_path, fn)

    def save(self, best=False):
        if not path.exists(self.model_path):
            makedirs(self.model_path, exist_ok=True)
        for i, net in enumerate(self.nets):
            name = self.net_names[i]
            if best:
                name += '.best'
            torch.save(self.nets[i].state_dict(), self.path_for(name))

    def load(self, best=False):
        print('load', 'best', best)
        for i, net in enumerate(self.nets):
            name = self.net_names[i]
            if best:
                name += '.best'
            self.nets[i].load_state_dict(torch.load(self.path_for(name)))


    def eval(self, eval_dir):
        self.net0.eval()
        with torch.no_grad():
            img_dir = path.join(eval_dir, 'training', 'image_2')
            fns = listdir(img_dir)
            N = len(fns)
            disps = np.zeros((N, 375, 1242), dtype=np.float32)
            resize = transforms.Resize((self.input_height, self.input_width))
            to_tensor = transforms.ToTensor()
            for idx, fn in enumerate(fns):
                fp = path.join(img_dir, fn)
                im = Image.open(fp)
                im = resize(im)
                im = to_tensor(im).to(self.device)
                im = torch.unsqueeze(im, 0)
                disp = self.net0(im)
                disp_est = disp[0].cpu().numpy()[0, 0, :, :]
                disp_est = cv2.resize(disp_est, (1242, 375), interpolation=cv2.INTER_LINEAR)
                disps[idx] = disp_est
            return disps


    def test(self, epoch, save=False):

        c_time = time.time()

        self.net0.eval()

        running_val_losses = np.zeros(4)

        N = self.val_n_img

        disparities_L = np.zeros((N, self.input_height, self.input_width), dtype=np.float32)
        images_L = np.zeros((N, 3, self.input_height, self.input_width), dtype=np.float32)
        gt_L = np.zeros((N, 375, 1242), dtype=np.float32)

        with torch.no_grad():
            for i, data in enumerate(self.val_loader):              
                data = to_device(data, self.device)
                out = self.compute(data)
                losses = self.criterion(out, epoch, self.loss_weights)
                loss = sum(losses)
                running_val_losses += np.array([l.item() for l in losses])

                left = out[0][0][0]
                right = out[0][1][0]
                disp_est_left = out[2][0][0]
                gt = out[3][0][:, 0, :, :]
                DR = disp_est_left.cpu().numpy()[:, 0, :, :]

                ndata, _, _ = DR.shape
                start = i*self.batch_size
                end = i*self.batch_size + ndata

                disparities_L[start:end] = DR
                gt_L[start:end] = gt.cpu().numpy()
                images_L[start:end] = left.cpu().numpy()

            running_val_losses /= self.val_n_img / self.batch_size
            running_val_loss = sum(running_val_losses)

            model_saved = '[*]'
            if save and running_val_loss < self.best_val_loss:
                self.save(True)
                self.best_val_loss = running_val_loss
                model_saved = '[S]'
            print(
                '      Test',
                'G: {} {:.3f}({:.3f})'.format(running_val_losses, running_val_loss, self.best_val_loss),
                'Time: {:.2f}s'.format(time.time() - c_time),
                model_saved
            )
        
        return disparities_L, images_L, gt_L
