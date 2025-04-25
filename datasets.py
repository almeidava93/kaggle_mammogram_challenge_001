import logging
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
from tqdm import tqdm
from logs import get_logger
import shutil
from torch.utils.data import DataLoader

logger = get_logger(__name__, log_level=logging.DEBUG, log_to_file=True)

from config import MammogramClassifierConfig

# Create relevant directories if they don't exist
dataset_cache_path = Path('datasets_cache')
Path(dataset_cache_path, 'train').mkdir(parents=True, exist_ok=True)
Path(dataset_cache_path, 'val').mkdir(parents=True, exist_ok=True)
Path(dataset_cache_path, 'test').mkdir(parents=True, exist_ok=True)

def collate_fn(batch):
    """
    batch: list of tuples (imgs, target, mask)
      imgs: [max_images, C, H, W]
      target: scalar
      mask: [max_images]
      imgs_metadata: [max_images, num_metadata]
    """
    # Get max images per study
    max_images = max([len(item[0]) for item in batch])

    # Padding
    images = []
    masks = []
    imgs_metadata = []

    for i in range(len(batch)):
    # Pad images to max_images
        img = batch[i][0]
        if len(img) < max_images:
            padding_img = torch.zeros_like(img[0].unsqueeze(0))
            for _ in range(max_images - len(img)):
                img = torch.cat((img, padding_img), dim=0)
        images.append(img)

        # Pad masks to max_images
        mask = batch[i][2]
        if len(mask) < max_images:
            padding_mask = torch.zeros_like(mask[0].unsqueeze(0))
            for _ in range(max_images - len(mask)):
                mask = torch.cat((mask, padding_mask), dim=0)
        masks.append(mask)

        # Pad metadata to max_images
        img_metadata = batch[i][3]
        if len(img_metadata) < max_images:
            padding_metadata = torch.zeros_like(img_metadata[0].unsqueeze(0))
            for _ in range(max_images - len(img_metadata)):
                img_metadata = torch.cat((img_metadata, padding_metadata), dim=0)
        imgs_metadata.append(img_metadata)

    # Stack images and masks
    images = torch.stack(images, dim=0)  # [B, max_images, C, H, W]
    masks = torch.stack(masks, dim=0)  # [B, max_images]
    imgs_metadata = torch.stack(imgs_metadata, dim=0)  # [B, max_images, num_metadata]
    targets = torch.stack([item[1] for item in batch], dim=0)     # [B]

    return images, targets, masks, imgs_metadata

def load_metadata(config: MammogramClassifierConfig) -> Tuple[pd.DataFrame, MammogramClassifierConfig]:
    images_metadata_df = pd.read_csv(config.images_metadata_path, index_col=0, low_memory=False)
    train_split_df = pd.read_csv(config.train_split_path, index_col=0)

    ## PatientAge
    images_metadata_df['PatientAge'] = images_metadata_df['PatientAge'].str.extract(r'(\d+)').astype(float) # Convert to float
    mean_age = images_metadata_df[images_metadata_df['AccessionNumber'].isin(train_split_df['AccessionNumber'])]['PatientAge'].mean() # Get mean age from training set
    images_metadata_df['PatientAge'] = images_metadata_df['PatientAge'].fillna(mean_age)/120 # Scaled to [0, 1], filled with mean age from training set

    ## PatientOrientation
    images_metadata_df['PatientOrientation_0'] = images_metadata_df['PatientOrientation'].apply(lambda x: eval(x)[0])
    images_metadata_df['PatientOrientation_1'] = images_metadata_df['PatientOrientation'].apply(lambda x: eval(x)[1])

    ## Prepare numerical columns
    for col in config.img_metadata_num_cols:
        # Convert to float
        images_metadata_df[col] = images_metadata_df[col].astype(float)
        # Fill missing values with mean of train data
        train_data_mean_value = images_metadata_df[images_metadata_df['AccessionNumber'].isin(train_split_df['AccessionNumber'])]['PatientAge'].mean()
        images_metadata_df[col] = images_metadata_df[col].fillna(train_data_mean_value)
        # Scale to [0, 1]
        images_metadata_df[col] = (images_metadata_df[col] - images_metadata_df[col].min()) / (images_metadata_df[col].max() - images_metadata_df[col].min())

    ## Transform columns to categorical and encode them
    n_categories = {}
    for col in config.img_metadata_cat_cols:
        images_metadata_df[col] = images_metadata_df[col].astype(str).fillna('MISSING')
        n_categories[col] = images_metadata_df[col].nunique()
        n_categories[f'{col}_categories'] = images_metadata_df[col].astype('category').cat.categories.tolist()
        images_metadata_df[col] = images_metadata_df[col].astype('category').cat.codes

    config.classes_per_cat = n_categories

    # Load basic transformations
    transform = transforms.Compose([
                        transforms.Resize((config.img_size, config.img_size)),
                    ])
    config.transform = transform

    # Get max images per study
    config.max_images_per_study = images_metadata_df.groupby(by=['AccessionNumber'])['PatientID'].count().max()

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
            
            # Check if dynamic curriculum is enabled
            if config.use_dynamic_curriculum:
                assert config.workers == 0, "Dynamic curriculum only works with workers=0"
                self.config.sampler_pos_prob = self.config.dynamic_curriculum_starting_pos_prob
                # Define minimum probability of sampling positive samples if not given
                if self.config.dynamic_curriculum_min_pos_prob is None:
                    self.config.dynamic_curriculum_min_pos_prob = self.df['target'].mean()
            
            # Define probability of sampling positive samples and negative samples
            if self.config.sampler_pos_prob is not None:
                pos_prob = self.config.sampler_pos_prob
                neg_prob = 1 - pos_prob
            else:
                pos_prob = self.df['target'].mean()
                neg_prob = 1 - pos_prob

            # Inverse frequency for each class
            class_sample_counts = self.df['target'].value_counts().to_dict()
            class_weights = {
                0: neg_prob / class_sample_counts[0],
                1: pos_prob / class_sample_counts[1]
            }
            sample_weights = self.df['target'].map(class_weights).values

            self.sampler = WeightedRandomSampler(
                weights=torch.DoubleTensor(sample_weights),
                num_samples=len(sample_weights),
                replacement=True
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Prepare data
        image_study_data = self.df.iloc[idx]

        # Get image metadata
        images_metadata = self.images_metadata_df[self.images_metadata_df['AccessionNumber'] == image_study_data['AccessionNumber']]

        # Load cached data if available
        cache_path = self.dataset_cache_path / f"{idx}.pt"
        if self.config.use_cache and self.config.cache_data:
            if cache_path.exists():
                data = torch.load(cache_path, weights_only=False, map_location='cpu')
                imgs, target, mask = data['imgs'], data['target'], data['mask']
                num_images = len(images_metadata)
                # Prepare images metadata [img_idx, num_metadata]
                imgs_metadata = images_metadata[
                    self.config.img_metadata_cat_cols + 
                    self.config.img_metadata_num_cols +
                    ['PatientAge']
                    ].to_numpy()
                imgs_metadata = np.pad(imgs_metadata, ((0, self.config.max_images_per_study - num_images), (0, 0)))
                imgs_metadata = torch.tensor(imgs_metadata, dtype=torch.uint8)
                return imgs, target, mask, imgs_metadata
            
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
        mask = [1] * num_images

        # Prepare images metadata [img_idx, num_metadata]
        imgs_metadata = images_metadata[
            self.config.img_metadata_cat_cols + 
            self.config.img_metadata_num_cols +
            ['PatientAge']
            ].to_numpy()

        imgs = torch.stack(imgs, dim=0)  # [max_images, C, H, W]

        # Adjust n channels for pretrained models
        if self.config.use_vit_b_16:
            imgs = imgs.repeat(1, 3, 1, 1)

        mask = torch.tensor(mask, dtype=torch.uint8)
        imgs_metadata = torch.tensor(imgs_metadata, dtype=torch.uint8)
        
        if self.split == 'test':
            accession_number = torch.tensor(image_study_data['AccessionNumber'], dtype=torch.long)
            if self.config.cache_data:
                torch.save({
                    'imgs': imgs,
                    'target': accession_number,
                    'mask': mask,
                    'imgs_metadata': imgs_metadata
                }, cache_path)
            return imgs, accession_number, mask, imgs_metadata
        else:
            target = torch.tensor(image_study_data['target']).float()

            if self.config.cache_data:
                torch.save({
                    'imgs': imgs,
                    'target': target,
                    'mask': mask,
                    'imgs_metadata': imgs_metadata
                }, cache_path)
            return imgs, target, mask, imgs_metadata
        
    def build_cache(self):
        self.config.use_cache = False
        self.config.cache_data = True
        logger.info(f"Caching dataset {self.split} to {self.dataset_cache_path}")

        dataloader = DataLoader(
            self, 
            batch_size=self.config.batch_size, 
            collate_fn=collate_fn, 
            num_workers=4,
        )

        for batch in tqdm(dataloader, desc=f"Building cache for {self.split} dataset", total=len(self)/self.config.batch_size):
            pass

    def delete_cache(self):
        if self.config.cache_data and self.dataset_cache_path.exists():
            shutil.rmtree(self.dataset_cache_path)