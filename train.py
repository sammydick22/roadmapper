import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils as smp_utils
import logging

from data.dataset import RoadsDataset, get_training_augmentation, get_preprocessing
from models.deeplabv3plus import create_model, get_loss, get_metrics

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
            logging.FileHandler(os.path.join(log_dir, 'train.log')),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def train(
    data_dir,
    output_dir,
    batch_size=4,
    epochs=40,
    learning_rate=0.001,
    encoder_name='resnet50',
    encoder_weights='imagenet',
    classes=None,
    class_rgb_values=None,
    target_size=(1024, 1024),
    device=None,
    num_workers=0  # Set to 0 by default to avoid multiprocessing issues
):
    """
    Train the DeepLabV3+ model
    
    Args:
        data_dir: Directory containing the data
        output_dir: Directory to save model and logs
        batch_size: Batch size for training and validation
        epochs: Number of epochs to train
        learning_rate: Learning rate for the optimizer
        encoder_name: Name of the encoder backbone
        encoder_weights: Name of the weights to use
        classes: List of class names
        class_rgb_values: List of RGB values for each class
        target_size: Target size for the images and masks
        device: Device to use (cuda or cpu)
        num_workers: Number of workers for data loading
    """
    # Set up logging
    logger = setup_logging(os.path.join(output_dir, 'logs'))
    
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load metadata
    metadata_df = pd.read_csv(os.path.join(data_dir, 'metadata.csv'))
    metadata_df = metadata_df[metadata_df['split'] == 'train']
    metadata_df = metadata_df[['image_id', 'sat_image_path', 'mask_path']]
    metadata_df['sat_image_path'] = metadata_df['sat_image_path'].apply(
        lambda img_pth: os.path.join(data_dir, img_pth))
    metadata_df['mask_path'] = metadata_df['mask_path'].apply(
        lambda img_pth: os.path.join(data_dir, img_pth))
    
    # Shuffle DataFrame
    metadata_df = metadata_df.sample(frac=1).reset_index(drop=True)
    
    # Create train/val split
    valid_df = metadata_df.sample(frac=0.1, random_state=42)
    train_df = metadata_df.drop(valid_df.index)
    logger.info(f"Training samples: {len(train_df)}, Validation samples: {len(valid_df)}")
    
    # Create model
    model, preprocessing_fn = create_model(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        classes=len(classes),
        activation='sigmoid'
    )
    
    # Create train and validation datasets
    train_dataset = RoadsDataset(
        train_df,
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        class_rgb_values=class_rgb_values,
        target_size=target_size
    )
    
    valid_dataset = RoadsDataset(
        valid_df,
        preprocessing=get_preprocessing(preprocessing_fn),
        class_rgb_values=class_rgb_values,
        target_size=target_size
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    # Define loss function and metrics
    loss = get_loss()
    metrics = get_metrics()
    
    # Define optimizer
    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=learning_rate),
    ])
    
    # Define training and validation epochs
    train_epoch = smp_utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=device,
        verbose=True,
    )
    
    valid_epoch = smp_utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=device,
        verbose=True,
    )
    
    # Create output directories
    model_dir = os.path.join(output_dir, 'models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Training loop
    best_iou_score = 0.0
    train_logs_list, valid_logs_list = [], []
    
    for i in range(epochs):
        # Log current epoch
        logger.info(f'\nEpoch: {i}')
        
        # Train model
        train_logs = train_epoch.run(train_loader)
        train_logs_list.append(train_logs)
        
        # Validate model
        valid_logs = valid_epoch.run(valid_loader)
        valid_logs_list.append(valid_logs)
        
        # Log metrics
        logger.info(f'Train IoU: {train_logs["iou_score"]:.4f}')
        logger.info(f'Valid IoU: {valid_logs["iou_score"]:.4f}')
        logger.info(f'Train Dice Loss: {train_logs["dice_loss"]:.4f}')
        logger.info(f'Valid Dice Loss: {valid_logs["dice_loss"]:.4f}')
        
        # Save model if a better validation IoU score is obtained
        if best_iou_score < valid_logs['iou_score']:
            best_iou_score = valid_logs['iou_score']
            torch.save(
                model, 
                os.path.join(model_dir, f'best_model_{encoder_name}.pth')
            )
            logger.info(f'Model saved! Best IoU: {best_iou_score:.4f}')
    
    # Save training history
    train_logs_df = pd.DataFrame(train_logs_list)
    valid_logs_df = pd.DataFrame(valid_logs_list)
    
    # Plot and save IoU score
    plt.figure(figsize=(20, 8))
    plt.plot(train_logs_df.index.tolist(), train_logs_df.iou_score.tolist(), lw=3, label='Train')
    plt.plot(valid_logs_df.index.tolist(), valid_logs_df.iou_score.tolist(), lw=3, label='Valid')
    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel('IoU Score', fontsize=20)
    plt.title('IoU Score Plot', fontsize=20)
    plt.legend(loc='best', fontsize=16)
    plt.grid()
    plt.savefig(os.path.join(output_dir, 'iou_score_plot.png'))
    
    # Plot and save Dice loss
    plt.figure(figsize=(20, 8))
    plt.plot(train_logs_df.index.tolist(), train_logs_df.dice_loss.tolist(), lw=3, label='Train')
    plt.plot(valid_logs_df.index.tolist(), valid_logs_df.dice_loss.tolist(), lw=3, label='Valid')
    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel('Dice Loss', fontsize=20)
    plt.title('Dice Loss Plot', fontsize=20)
    plt.legend(loc='best', fontsize=16)
    plt.grid()
    plt.savefig(os.path.join(output_dir, 'dice_loss_plot.png'))
    
    # Save logs as CSV
    train_logs_df.to_csv(os.path.join(output_dir, 'train_logs.csv'), index=False)
    valid_logs_df.to_csv(os.path.join(output_dir, 'valid_logs.csv'), index=False)
    
    logger.info(f'Training completed! Best IoU: {best_iou_score:.4f}')
    
    return train_logs_df, valid_logs_df
