import torch
import torchvision.transforms as transforms
import numpy as np
from utils import gt_depth_to_disp, fill_depth
from PIL import Image



def image_transforms(augment_parameters=[0.8, 1.2, 0.5, 2.0, 0.8, 1.2],
                     do_augmentation=True,  size=(256, 512)):
    data_transform = transforms.Compose([
        InterpolateDepth(),
        DepthToDisparity(),
        Resize(size=size),
        RandomFlip(do_augmentation),
        ToTensor(),
        AugmentImagePair(augment_parameters, do_augmentation)
    ])
    return data_transform


class InterpolateDepth(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        i0, i1, d0, d1 = sample
        d0 = fill_depth(np.asarray(d0))
        d1 = fill_depth(np.asarray(d1))
        return i0, i1, d0, d1


class DepthToDisparity(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        i0, i1, d0, d1 = sample
        d0 = gt_depth_to_disp(d0)
        d1 = gt_depth_to_disp(d1)
        return i0, i1, d0, d1


class Resize(object):
    def __init__(self, size=(256, 512)):
        self.transform = transforms.Resize(size)

    def __call__(self, sample):
        i0, i1, d0, d1 = sample
        i0 = self.transform(i0)
        i1 = self.transform(i1)
        d0 = Image.fromarray(d0)
        d1 = Image.fromarray(d1)
        d0 = self.transform(d0)
        d1 = self.transform(d1)
        return i0, i1, d0, d1


class ToTensor(object):
    def __init__(self):
        self.transform = transforms.ToTensor()

    def __call__(self, sample):
        i0, i1, d0, d1 = sample
        i0 = self.transform(i0)
        i1 = self.transform(i1)
        d0 = self.transform(d0)
        d1 = self.transform(d1)
        return i0, i1, d0, d1

class RandomFlip(object):
    def __init__(self, do_augmentation):
        self.transform = transforms.RandomHorizontalFlip(p=1)
        self.do_augmentation = do_augmentation

    def __call__(self, sample):
        i0, i1, d0, d1 = sample
        i0 = self.transform(i0)
        i1 = self.transform(i1)
        d0 = self.transform(d0)
        d1 = self.transform(d1)
        k = np.random.uniform(0, 1, 1)
        if self.do_augmentation:
            if k > 0.5:
                fliped_left = self.transform(i1)
                fliped_right = self.transform(i0)
                flipped_d_left = self.transform(d1)
                flipped_d_right = self.transform(d0)
                return i0, i1, d0, d1
        return sample


class AugmentImagePair(object):
    def __init__(self, augment_parameters, do_augmentation):
        self.augment_parameters = augment_parameters
        self.do_augmentation = do_augmentation

    def __call__(self, sample):
        i0, i1, d0, d1 = sample
        p = np.random.uniform(0, 1, 1)
        if self.do_augmentation:
            self.gamma_low = self.augment_parameters[0]  # 0.8
            self.gamma_high = self.augment_parameters[1]  # 1.2
            self.brightness_low = self.augment_parameters[2]  # 0.5
            self.brightness_high = self.augment_parameters[3]  # 2.0
            self.color_low = self.augment_parameters[4]  # 0.8
            self.color_high = self.augment_parameters[5]  # 1.2
            if p > 0.5:
                # randomly shift gamma
                random_gamma = np.random.uniform(self.gamma_low, self.gamma_high)
                left_image_aug = i0 ** random_gamma
                right_image_aug = i1 ** random_gamma

                # randomly shift brightness
                random_brightness = np.random.uniform(self.brightness_low, self.brightness_high)
                left_image_aug = left_image_aug * random_brightness
                right_image_aug = right_image_aug * random_brightness

                # randomly shift color
                random_colors = np.random.uniform(self.color_low, self.color_high, 3)
                for i in range(3):
                    left_image_aug[i, :, :] *= random_colors[i]
                    right_image_aug[i, :, :] *= random_colors[i]

                # saturate
                left_image_aug = torch.clamp(left_image_aug, 0, 1)
                right_image_aug = torch.clamp(right_image_aug, 0, 1)

                sample = (left_image_aug, right_image_aug, d0, d1)
        return sample
