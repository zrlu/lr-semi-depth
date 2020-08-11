import os
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from transforms import image_transforms, ToTensor


class KITTI(Dataset):
    def __init__(self, root_dir, transform=None):
        left_dir = os.path.join(root_dir, 'image_2/')
        self.left_paths = sorted([os.path.join(left_dir, fname) for fname\
                           in os.listdir(left_dir)])
        right_dir = os.path.join(root_dir, 'image_3/')
        self.right_paths = sorted([os.path.join(right_dir, fname) for fname\
                            in os.listdir(right_dir)])
        assert len(self.right_paths) == len(self.left_paths)
        self.transform = transform

    def __len__(self):
        return len(self.left_paths)

    def __getitem__(self, idx):
        left_image = Image.open(self.left_paths[idx])
        right_image = Image.open(self.right_paths[idx])
        sample = {'left_image': left_image, 'right_image': right_image}

        if self.transform:
            sample = self.transform(sample)
            return sample
        else:
            return sample


def prepare_dataloader(data_directory, augment_parameters,
                       do_augmentation, batch_size, size, num_workers, shuffle=True):
    data_transform = image_transforms(
        augment_parameters=augment_parameters,
        do_augmentation=do_augmentation,
        size = size)
    dataset = KITTI(data_directory, transform=data_transform)
    n_img = len(dataset)
    print('Use a dataset with', n_img, 'images')
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=shuffle, num_workers=num_workers,
                        pin_memory=True)
    return n_img, loader