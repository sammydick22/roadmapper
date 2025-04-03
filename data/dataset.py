import os
import cv2
import torch
import numpy as np
import albumentations as album
from torch.utils.data import Dataset
from utils.encoding import one_hot_encode, reverse_one_hot

class RoadsDataset(Dataset):
    """
    DeepGlobe Road Extraction Challenge Dataset
    Read images, apply augmentation and preprocessing transformations.
    
    Args:
        df: DataFrame containing images / labels paths
        class_rgb_values: RGB values of select classes to extract from segmentation mask
        augmentation: data transformation pipeline (e.g. flip, scale, etc.)
        preprocessing: data preprocessing (e.g. normalization, shape manipulation, etc.)
        target_size: target size for the images and masks
    """
    
    def __init__(
            self, 
            df,
            class_rgb_values=None, 
            augmentation=None, 
            preprocessing=None,
            target_size=(1024, 1024)
    ):
        self.image_paths = df['sat_image_path'].tolist()
        self.mask_paths = df['mask_path'].tolist()
        
        self.class_rgb_values = class_rgb_values
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.target_size = target_size
    
    def __getitem__(self, i):
        # Read images and masks
        image = cv2.cvtColor(cv2.imread(self.image_paths[i]), cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(cv2.imread(self.mask_paths[i]), cv2.COLOR_BGR2RGB)
        
        # One-hot-encode the mask
        mask = one_hot_encode(mask, self.class_rgb_values).astype('float')
        
        # Apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        image = cv2.resize(image, self.target_size)
        mask = cv2.resize(mask, self.target_size)
        
        # Apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask
        
    def __len__(self):
        return len(self.image_paths)


def get_training_augmentation():
    """
    Create training augmentation transform
    
    Returns:
        album.Compose: Composed augmentations
    """
    train_transform = [
        album.HorizontalFlip(p=0.5),
        album.VerticalFlip(p=0.5),
    ]
    return album.Compose(train_transform)


def to_tensor(x, **kwargs):
    """
    Convert image to PyTorch tensor
    
    Args:
        x: Input image
        
    Returns:
        Tensor with shape (C, H, W)
    """
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn=None):
    """
    Construct preprocessing transform
    
    Args:
        preprocessing_fn: Data normalization function 
            (can be specific for each pretrained neural network)
            
    Returns:
        album.Compose: Composed preprocessing functions
    """
    _transform = []
    if preprocessing_fn:
        _transform.append(album.Lambda(image=preprocessing_fn))
    _transform.append(album.Lambda(image=to_tensor, mask=to_tensor))
        
    return album.Compose(_transform)
