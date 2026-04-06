import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class HouseSegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, img_size=256):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.img_size = img_size

        self.image_files = sorted([
            f for f in os.listdir(images_dir) if f.endswith('.png') or f.endswith('.jpg')
        ])

        self.image_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, img_name.replace('.jpg', '.png'))

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        image = self.image_transform(image)
        mask = self.mask_transform(mask)

        # Binarize mask
        mask = (mask > 0.5).float()

        return image, mask


def get_dataloaders(data_dir, batch_size=8, img_size=256):
    splits = ['train', 'val', 'test']
    loaders = {}

    for split in splits:
        images_dir = os.path.join(data_dir, split, 'images')
        masks_dir = os.path.join(data_dir, split, 'masks')

        dataset = HouseSegmentationDataset(images_dir, masks_dir, img_size)

        shuffle = (split == 'train')
        loaders[split] = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0
        )

    return loaders