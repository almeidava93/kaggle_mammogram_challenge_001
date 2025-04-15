from typing import Tuple
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
from skimage import morphology

from config import MammogramClassifierConfig

# Create relevant directories if they don't exist
dataset_cache_path = Path('datasets_cache')
Path(dataset_cache_path, 'train').mkdir(parents=True, exist_ok=True)
Path(dataset_cache_path, 'val').mkdir(parents=True, exist_ok=True)
Path(dataset_cache_path, 'test').mkdir(parents=True, exist_ok=True)

def load_metadata(config: MammogramClassifierConfig) -> Tuple[pd.DataFrame, MammogramClassifierConfig]:
    images_metadata_df = pd.read_csv(config.images_metadata_path, index_col=0)
    train_split_df = pd.read_csv(config.train_split_path, index_col=0)

    ## PatientAge
    images_metadata_df['PatientAge'] = images_metadata_df['PatientAge'].str.extract(r'(\d+)').astype(float) # Convert to float
    mean_age = images_metadata_df[images_metadata_df['AccessionNumber'].isin(train_split_df['AccessionNumber'])]['PatientAge'].mean() # Get mean age from training set
    images_metadata_df['PatientAge'] = images_metadata_df['PatientAge'].fillna(mean_age)/120 # Scaled to [0, 1], filled with mean age from training set

    ## PatientOrientation
    images_metadata_df['PatientOrientation_0'] = images_metadata_df['PatientOrientation'].apply(lambda x: eval(x)[0])
    images_metadata_df['PatientOrientation_1'] = images_metadata_df['PatientOrientation'].apply(lambda x: eval(x)[1])

    ## Transform columns to categorical and encode them
    n_categories = {}
    for col in config.img_metadata_cat_cols:
        n_categories[col] = images_metadata_df[col].nunique()
        images_metadata_df[col] = images_metadata_df[col].astype('category').cat.codes

    config.classes_per_cat = n_categories

    # Load basic transformations
    transform = transforms.Compose([
                        transforms.Resize((config.img_size, config.img_size)),
                    ])
    config.transform = transform

    # Get max images per study
    config.max_images_per_study = images_metadata_df.groupby(by=['AccessionNumber'])['PatientID'].count().max().item()

    return images_metadata_df, config


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

def crop_dark_pixels(img: torch.Tensor, threshold: float = None) -> torch.Tensor:
    if threshold is None:
        threshold = img.min().item()
    img = img.squeeze(0).numpy()
    mask = img <= threshold

    # Remove small artifacts from the image outside the main breast region
    inv_mask = mask == False
    cleaned = morphology.remove_small_objects(inv_mask, min_size=1000)
    mask = cleaned

    # Apply mask
    masked_img = np.zeros(img.shape)
    masked_img[mask] = img[mask]

    try:
        # Get mask bounderies
        coords = np.argwhere(masked_img)
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1

        # Crop masked img and return img tensor
        cropped_np = masked_img[y0:y1, x0:x1]
        img = torch.tensor(cropped_np, dtype=torch.float32).unsqueeze(0)
        return img
    except Exception as e:
        print("Error cropping dark pixels:", e)
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
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
            split: str,
            config: MammogramClassifierConfig,
        ):

        assert split in ['train', 'val', 'test'], "split must be one of ['train', 'val', 'test']"
        
        if split == 'test':
            self.df = pd.read_csv(config.test_split_path, usecols=['AccessionNumber'])
        elif split == 'train':
            self.df = pd.read_csv(
                config.train_split_path, 
                usecols=['AccessionNumber', 'target', 'path'])[:config.dataset_size]
        elif split == 'val':
            self.df = pd.read_csv(
                config.val_split_path, 
                usecols=['AccessionNumber', 'target', 'path'])[:config.dataset_size]
            
        self.split = split
        self.images_metadata_df, self.config = load_metadata(config)
        self.sampler = None
        self.pos_weight = None
        self.dataset_cache_path = dataset_cache_path / split

        if self.split == 'train':
            assert isinstance(self.config.pos_weight_scaler, float) and self.config.pos_weight_scaler > 0, "pos_weight_scaler must be a float and greater than 0"
            self.pos_weight = torch.tensor(self.df['target'].value_counts()[0] / self.df['target'].value_counts()[1], dtype=torch.float)
            self.pos_weight = self.pos_weight * self.config.pos_weight_scaler

        # Create weighted random sampler to balance traning examples
        if self.split == 'train' and self.config.use_weighted_random_sampler:
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
        # Load cached data if available
        cache_path = self.dataset_cache_path / f"{idx}.pt"
        if cache_path.exists():
            data = torch.load(cache_path, weights_only=False, map_location='cuda')
            imgs, target, mask, imgs_metadata = data['imgs'], data['target'], data['mask'], data['imgs_metadata']
            return imgs, target, mask, imgs_metadata

        # Else, prepare data
        image_study_data = self.df.iloc[idx]

        # Get image metadata
        images_metadata = self.images_metadata_df[self.images_metadata_df['AccessionNumber'] == image_study_data['AccessionNumber']]

        # Randomize order in train dataset
        if self.split == 'train':
            if self.config.randomize_image_order:
                images_metadata = images_metadata.sample(frac=1)

        study_images_path = images_metadata['path']

        # Prepare images
        imgs = []
        for img_path in study_images_path[:self.config.max_images_per_study]:  # Truncate if too many
            dicom_img = pydicom.dcmread(img_path, force=True)
            img = np.array(dicom_img.pixel_array, dtype=np.float32)
            
            # If no preprocessing requested, just resize and return
            if self.config.no_preprocessing:
                img = torch.from_numpy(img)
                img = img.float().view(1, *img.shape)
                if self.config.transform is not None:
                    img = self.config.transform(img)
                imgs.append(img)
                continue

            # Invert images with white background
            if self.config.invert_background:
                img_mode = torch.tensor(img).mode().values.max().item()
                if img_mode != 0:
                    img = img*-1 + img.max()

            # Normalize
            img = ((img - img.mean())/(img.std() + self.config.eps))

            # To tensor and reshape
            img = torch.from_numpy(img).float().view(1, *img.shape)  # Shape: [1, H, W]

            # Image transformations
            if self.config.remove_dark_pixels:
                img = crop_dark_pixels(img)

            if self.config.add_padding_pixels:
                img = pad_to_square(img)

            if self.config.transform is not None:
                img = self.config.transform(img)

            if self.config.add_random_transforms:
                img = apply_random_transforms(img, p=self.config.random_transforms_prob, max_transforms=self.config.random_transforms_max)

            imgs.append(img)

        # Padding if fewer than max_images
        num_images = len(imgs)
        mask = [1] * num_images + [0] * (self.config.max_images_per_study - num_images)

        if num_images < self.config.max_images_per_study:
            padding_img = torch.zeros_like(imgs[0])
            for _ in range(self.config.max_images_per_study - num_images):
                imgs.append(padding_img)

        # Prepare images metadata [img_idx, num_metadata]
        imgs_metadata = images_metadata[self.config.img_metadata_cat_cols + ['PatientAge']].to_numpy()
        imgs_metadata = np.pad(imgs_metadata, ((0, self.config.max_images_per_study - num_images), (0, 0)))

        imgs = torch.stack(imgs, dim=0)  # [max_images, C, H, W]

        # Adjust n channels for pretrained models
        if self.config.use_vit_b_16:
            imgs = imgs.repeat(1, 3, 1, 1)

        mask = torch.tensor(mask, dtype=torch.uint8)
        imgs_metadata = torch.tensor(imgs_metadata, dtype=torch.uint8)
        
        if self.split == 'test':
            accession_number = torch.tensor(image_study_data['AccessionNumber'], dtype=torch.long)
            torch.save({
                'imgs': imgs,
                'target': accession_number,
                'mask': mask,
                'imgs_metadata': imgs_metadata
            }, cache_path)
            return imgs, accession_number, mask, imgs_metadata
        else:
            target = torch.tensor(image_study_data['target']).float()
            torch.save({
                'imgs': imgs,
                'target': target,
                'mask': mask,
                'imgs_metadata': imgs_metadata
            }, cache_path)
            return imgs, target, mask, imgs_metadata
