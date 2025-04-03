import os
import cv2
import numpy as np
import torch
import pandas as pd
import random
import logging
from torch.utils.data import DataLoader

from utils.encoding import reverse_one_hot, colour_code_segmentation
from utils.visualization import save_visualization
from data.dataset import RoadsDataset, get_preprocessing

def setup_logging(log_dir):
    """
    Set up logging configuration
    
    Args:
        log_dir: Directory to save logs
        
    Returns:
        logger: Logger object
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'predict.log')),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def predict(
    data_dir,
    model_path,
    output_dir,
    classes,
    class_rgb_values,
    preprocessing_fn=None,
    target_size=(1024, 1024),
    device=None,
    num_samples=5,
    save_all_predictions=False
):
    """
    Make predictions using a trained model
    
    Args:
        data_dir: Directory containing the data
        model_path: Path to the trained model
        output_dir: Directory to save predictions and logs
        classes: List of class names
        class_rgb_values: List of RGB values for each class
        preprocessing_fn: Preprocessing function
        target_size: Target size for the images and masks
        device: Device to use (cuda or cpu)
        num_samples: Number of random samples to visualize
        save_all_predictions: Whether to save all predictions or just the visualized ones
    """
    # Set up logging
    logger = setup_logging(os.path.join(output_dir, 'logs'))
    
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create output directories
    predictions_dir = os.path.join(output_dir, 'predictions')
    if not os.path.exists(predictions_dir):
        os.makedirs(predictions_dir)
    
    # Load metadata
    metadata_df = pd.read_csv(os.path.join(data_dir, 'metadata.csv'))
    
    # Use validation split for predictions
    test_df = metadata_df[metadata_df['split'] == 'valid']
    if len(test_df) == 0:  # If no validation split is available, use a random subset of training data
        test_df = metadata_df[metadata_df['split'] == 'train'].sample(frac=0.1, random_state=42)
    
    test_df = test_df[['image_id', 'sat_image_path', 'mask_path']]
    test_df['sat_image_path'] = test_df['sat_image_path'].apply(
        lambda img_pth: os.path.join(data_dir, img_pth))
    test_df['mask_path'] = test_df['mask_path'].apply(
        lambda img_pth: os.path.join(data_dir, img_pth))
    
    logger.info(f"Test samples: {len(test_df)}")
    
    # Load model
    logger.info(f"Loading model from {model_path}")
    best_model = torch.load(model_path, map_location=device)
    best_model.eval()
    
    # Create test dataset
    test_dataset = RoadsDataset(
        test_df,
        preprocessing=get_preprocessing(preprocessing_fn),
        class_rgb_values=class_rgb_values,
        target_size=target_size
    )
    
    # Create test dataset for visualization
    test_dataset_vis = RoadsDataset(
        test_df,
        class_rgb_values=class_rgb_values,
        target_size=target_size
    )
    
    # Select random samples if specified
    if num_samples > 0 and num_samples < len(test_dataset):
        indices = random.sample(range(len(test_dataset)), num_samples)
    else:
        indices = range(len(test_dataset))
    
    # Make predictions
    for i, idx in enumerate(indices):
        image, gt_mask = test_dataset[idx]
        image_vis = test_dataset_vis[idx][0].astype('uint8')
        
        # Convert to tensor and make prediction
        x_tensor = torch.from_numpy(image).to(device).unsqueeze(0)
        with torch.no_grad():
            pred_mask = best_model(x_tensor)
            pred_mask = pred_mask.detach().squeeze().cpu().numpy()
        
        # Convert pred_mask from CHW format to HWC format
        pred_mask = np.transpose(pred_mask, (1, 2, 0))
        
        # Get prediction channel corresponding to foreground (road class)
        pred_road_heatmap = pred_mask[:, :, classes.index('road')]
        
        # Convert prediction to RGB
        pred_mask_vis = colour_code_segmentation(
            reverse_one_hot(pred_mask), 
            class_rgb_values
        )
        
        # Convert ground truth mask to RGB
        gt_mask = np.transpose(gt_mask, (1, 2, 0))
        gt_mask_vis = colour_code_segmentation(
            reverse_one_hot(gt_mask), 
            class_rgb_values
        )
        
        # Save visualization
        save_visualization(
            predictions_dir,
            f"prediction_{idx}.png",
            original_image=image_vis,
            ground_truth_mask=gt_mask_vis,
            predicted_mask=pred_mask_vis,
            pred_road_heatmap=pred_road_heatmap
        )
        
        # Save side-by-side comparison using OpenCV
        comparison = np.hstack([image_vis, gt_mask_vis, pred_mask_vis])
        cv2.imwrite(
            os.path.join(predictions_dir, f"comparison_{idx}.png"),
            comparison[:, :, ::-1]  # Convert RGB to BGR for OpenCV
        )
        
        logger.info(f"Saved prediction {i+1}/{len(indices)}")
    
    logger.info(f"Predictions completed! Saved to {predictions_dir}")
