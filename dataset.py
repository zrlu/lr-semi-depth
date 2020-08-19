import os
from PIL import Image

from torch.utils.data import Dataset, DataLoader, ConcatDataset, SubsetRandomSampler
from transforms import image_transforms
import random

class KITTI(Dataset):
    def __init__(self, root_dir, transform=None):
        left_dir = os.path.join(root_dir, 'image_02/data')
        left_fns =  set([fname for fname in os.listdir(left_dir)])
        right_dir = os.path.join(root_dir, 'image_03/data')
        right_fns = set([fname for fname in os.listdir(right_dir)])
        left_depth_dir = os.path.join(root_dir, 'proj_depth/groundtruth/image_02')
        left_path_fns =  set([fname for fname in os.listdir(left_depth_dir)])
        right_depth_dir = os.path.join(root_dir, 'proj_depth/groundtruth/image_03')
        right_path_fns = set([fname for fname in os.listdir(right_depth_dir)])
        fns = left_fns & right_fns & left_path_fns & right_path_fns
        # fns = random.sample(fns, min(len(fns), 5))
        self.left_paths = sorted([os.path.join(left_dir, fname) for fname in fns])
        self.right_paths = sorted([os.path.join(right_dir, fname) for fname in fns])
        self.left_depth_paths = sorted([os.path.join(left_depth_dir, fname) for fname in fns])
        self.right_depth_paths = sorted([os.path.join(right_depth_dir, fname) for fname in fns])
        assert len(self.right_paths) == len(self.left_paths) == len(self.left_depth_paths) == len(self.right_depth_paths)
        self.transform = transform

    def __len__(self):
        return len(self.left_paths)

    def __getitem__(self, idx):
        left_image = Image.open(self.left_paths[idx])
        right_image = Image.open(self.right_paths[idx])
        left_depth = Image.open(self.left_depth_paths[idx])
        right_depth = Image.open(self.right_depth_paths[idx])
        sample = [left_image, right_image, left_depth, right_depth]
        if self.transform:
            sample = self.transform(sample)
            return sample
        else:
            return sample

def prepare_dataloader(data_directory, augment_parameters,
                       do_augmentation, batch_size, size, num_workers, shuffle=True, drop_last=True):
    data_transform = image_transforms(
        augment_parameters=augment_parameters,
        do_augmentation=do_augmentation,
        size = size)
    dataset = ConcatDataset([KITTI(os.path.join(data_directory, data_dir), transform=data_transform) for data_dir in os.listdir(data_directory)])
    n_img = len(dataset)
    print('Use a dataset with', n_img, 'instances')
    print('Batch size', batch_size)
    print('Drop last', drop_last)
    loader = DataLoader(dataset, batch_size=batch_size,
                        drop_last=drop_last,
                        shuffle=shuffle, num_workers=num_workers,
                        pin_memory=True)
    return n_img, loader