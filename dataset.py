import os
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from transforms import image_transforms, ToTensor


class KITTI(Dataset):
    def __init__(self, root_dir, mode, transform=None):
        left_dir = os.path.join(root_dir, 'image_2/')
        self.left_paths = sorted([os.path.join(left_dir, fname) for fname\
                           in os.listdir(left_dir)])
        if mode == 'train':
            right_dir = os.path.join(root_dir, 'image_3/')
            self.right_paths = sorted([os.path.join(right_dir, fname) for fname\
                                in os.listdir(right_dir)])
            assert len(self.right_paths) == len(self.left_paths)
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.left_paths)

    def __getitem__(self, idx):
        left_image = Image.open(self.left_paths[idx])
        if self.mode == 'train':
            right_image = Image.open(self.right_paths[idx])
            sample = {'left_image': left_image, 'right_image': right_image}

            if self.transform:
                sample = self.transform(sample)
                return sample
            else:
                return sample
        else:
            if self.transform:
                left_image = self.transform(left_image)
            return left_image


def prepare_dataloader(data_directory, mode, augment_parameters,
                       do_augmentation, batch_size, size, num_workers):
    data_transform = image_transforms(
        mode=mode,
        augment_parameters=augment_parameters,
        do_augmentation=do_augmentation,
        size = size)
    dataset = KITTI(data_directory, mode, transform=data_transform)
    n_img = len(dataset)
    print('Use a dataset with', n_img, 'images (mode={})'.format(mode))
    if mode == 'train':
        loader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=num_workers,
                            pin_memory=True)
    else:
        loader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers,
                            pin_memory=True)
    return n_img, loader