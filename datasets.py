from typing import Optional
from torch.utils.data import Dataset
from torch.utils.data import WeightedRandomSampler
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import pydicom
from torchvision import transforms
import numpy as np
import torch.nn.functional as F
import random
from torchvision import transforms	

available_transforms = [
    'RandomPerspective', 
    'RandomVerticalFlip', 
    'RandomHorizontalFlip',
    'RandomRotation', 
    'GaussianBlur', 
    'RandomAffine',
    'ElasticTransform',
    ]

def get_transforms(ts: list[str]):
    ts_dict = {
        "GaussianBlur": {
            "kernel_size": random.choice(list(range(1, 9, 2))),
            "sigma": random.uniform(0.1, 2.0),
        }, 
        "RandomRotation": {
            "degrees": random.uniform(0, 360),
            "interpolation": random.choice([
                transforms.InterpolationMode.NEAREST, 
                transforms.InterpolationMode.BILINEAR]),	
        },
        "RandomAffine": {
            "degrees": 0,
            "shear": (
                *sorted(random.sample(list(range(-15, 15)), 2)),
                *sorted(random.sample(list(range(-15, 15)), 2)),
            )
        },
        'RandomPerspective': {
            'distortion_scale': random.uniform(0, 1.0),
        },
        'ColorJitter': {
            'brightness': random.uniform(0, 1.0),
            'contrast': random.uniform(0, 1.0),
        },
        'ElasticTransform': {
            'alpha': random.uniform(0, 50.0),
            'sigma': random.uniform(0, 5.0),
        },
        'RandomAdjustSharpness': {
            'sharpness_factor': random.uniform(0, 2.0),
            'p': random.uniform(0, 1.0),
        },
        'RandomHorizontalFlip': {
            'p': random.uniform(0, 1.0),
        },
        'RandomVerticalFlip': {
            'p': random.uniform(0, 1.0),
        }
    }

    return [
        getattr(transforms, t)(**ts_dict[t]) for t in ts if t in ts_dict.keys()
    ]

def apply_random_transforms(img: torch.Tensor, p: float, max_transforms: int) -> torch.Tensor:
    assert max_transforms <= len(available_transforms)-1, f"max_transforms must be <= {len(available_transforms)-1}"
    assert p >= 0 and p <= 1, "p must be between 0 and 1"

    if random.random() < 1 - p:
        return img
    
    ts = random.sample(available_transforms, random.randint(1, max_transforms))
    selected_transforms = get_transforms(ts)
    return transforms.Compose(selected_transforms)(img)

def crop_dark_pixels(img: torch.Tensor, threshold: float = 0.3168) -> torch.Tensor:
    img = img.squeeze(0).numpy()
    mask = img < threshold
    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    cropped_np = img[y0:y1, x0:x1]
    img = torch.tensor(cropped_np).unsqueeze(0)
    return img

def pad_to_square(img: torch.Tensor) -> torch.Tensor:
    _, H, W = img.shape
    max_dim = torch.max(torch.tensor([H, W]))
    greater_dim = torch.argmax(torch.tensor([H, W]))

    if greater_dim == 0:
        bottom_pad = top_pad = 0
        left_pad = (max_dim - W) // 2
        right_pad = max_dim - W - left_pad
    else:
        left_pad = right_pad = 0
        top_pad = (max_dim - H) // 2
        bottom_pad = max_dim - H - top_pad

    pad = (left_pad, right_pad, top_pad, bottom_pad)  # (left, right, top, bottom)
    padded_img = F.pad(img, pad, mode='constant', value=0)
    return padded_img

class MammogramDataset(Dataset):
    def __init__(
            self, 
            split_filepath: str, 
            max_images_per_study: int,  # Set your desired max number
            img_metadata_cat_cols: list[str],
            transform: transforms.Compose = None,
            eps: float = 1e-7,
            img_size: tuple = (1024, 1024),
            use_weighted_random_sampler: Optional[bool] = None,
            add_random_transforms: Optional[bool] = None,
            randomize_image_order: Optional[bool] = None,
            images_metadata_df: pd.DataFrame = None,
            dataset_size: Optional[int] = -1,
            pos_weight_scaler: Optional[float] = 1,
            remove_dark_pixels: Optional[bool] = False,
            add_padding_pixels: Optional[bool] = False,
            random_transforms_p: Optional[float] = 0,
            random_transforms_max: Optional[int] = 0,
            use_vit_b_16: Optional[bool] = False,
        ):
        
        if 'test' in split_filepath:
            self.df = pd.read_csv(Path(split_filepath), usecols=['AccessionNumber'])
        else:
            self.df = pd.read_csv(Path(split_filepath), usecols=['AccessionNumber', 'target', 'path'])[:dataset_size]

        self.images_metadata_df = images_metadata_df
        self.use_vit_b_16 = use_vit_b_16
        self.img_metadata_cat_cols = img_metadata_cat_cols
        self.remove_dark_pixels = remove_dark_pixels
        self.add_padding_pixels = add_padding_pixels
        self.transform = transform
        self.eps = eps
        self.max_images = max_images_per_study
        self.img_size = img_size
        self.sampler = None
        self.pos_weight = None
        self.add_random_transforms = add_random_transforms
        self.random_transforms_p = random_transforms_p
        self.random_transforms_max = random_transforms_max
        self.randomize_image_order = randomize_image_order

        if 'train' in split_filepath:
            self.split = 'train'
        elif 'val' in split_filepath:
            self.split = 'val'
        elif 'test' in split_filepath:
            self.split = 'test'
        else:
            raise ValueError(f"Invalid split_filepath: {split_filepath}")

        if self.split == 'train':
            assert isinstance(pos_weight_scaler, float) and pos_weight_scaler > 0, "pos_weight_scaler must be a float and greater than 0"
            self.pos_weight = torch.tensor(self.df['target'].value_counts()[0] / self.df['target'].value_counts()[1], dtype=torch.float)
            self.pos_weight = self.pos_weight * pos_weight_scaler

        # Create weighted random sampler to balance traning examples
        if self.split == 'train' and use_weighted_random_sampler:
            # Inverse frequency for each class
            class_sample_counts = self.df['target'].value_counts().to_dict()
            class_weights = {cls: 1.0 / count for cls, count in class_sample_counts.items()}
            sample_weights = self.df['target'].map(class_weights).values

            self.sampler = WeightedRandomSampler(
                weights=torch.DoubleTensor(sample_weights),
                num_samples=len(sample_weights),
                replacement=True
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_study_data = self.df.iloc[idx]

        # Get image metadata
        images_metadata = self.images_metadata_df[self.images_metadata_df['AccessionNumber'] == image_study_data['AccessionNumber']]

        # Randomize order in train dataset
        if self.split == 'train':
            if self.randomize_image_order:
                images_metadata = images_metadata.sample(frac=1)

        study_images_path = images_metadata['path']

        # Prepare images
        imgs = []
        for img_path in study_images_path[:self.max_images]:  # Truncate if too many
            dicom_img = pydicom.dcmread(img_path, force=True)
            img = np.array(dicom_img.pixel_array, dtype=np.float32)

            # Normalize
            img = img / 65535.0
            img = (img - img.mean()) / (img.std() + self.eps)

            img = torch.from_numpy(img).float().view(1, *img.shape)  # Shape: [1, H, W]

            # Resize if needed
            if img.shape[-2:] != self.img_size:
                img = torch.nn.functional.interpolate(img.unsqueeze(0), size=self.img_size, mode='bilinear', align_corners=False).squeeze(0)

            # Image transformations
            if self.remove_dark_pixels:
                img = crop_dark_pixels(img)

            if self.add_padding_pixels:
                img = pad_to_square(img)

            if self.transform is not None:
                img = self.transform(img)

            if self.add_random_transforms:
                img = apply_random_transforms(img, p=self.random_transforms_p, max_transforms=self.random_transforms_max)

            imgs.append(img)

        # Padding if fewer than max_images
        num_images = len(imgs)
        mask = [1] * num_images + [0] * (self.max_images - num_images)

        if num_images < self.max_images:
            padding_img = torch.zeros_like(imgs[0])
            for _ in range(self.max_images - num_images):
                imgs.append(padding_img)

        # Prepare images metadata [img_idx, num_metadata]
        imgs_metadata = images_metadata[self.img_metadata_cat_cols + ['PatientAge']].to_numpy()
        imgs_metadata = np.pad(imgs_metadata, ((0, self.max_images - num_images), (0, 0)))

        imgs = torch.stack(imgs, dim=0)  # [max_images, C, H, W]

        # Adjust n channels for pretrained models
        if self.use_vit_b_16:
            imgs = imgs.repeat(1, 3, 1, 1)

        mask = torch.tensor(mask, dtype=torch.uint8)
        imgs_metadata = torch.tensor(imgs_metadata, dtype=torch.uint8)
        
        if self.split == 'test':
            return imgs, torch.tensor(image_study_data['AccessionNumber'], dtype=torch.long), mask, imgs_metadata
        else:
            target = torch.tensor(image_study_data['target']).float()
            return imgs, target, mask, imgs_metadata
