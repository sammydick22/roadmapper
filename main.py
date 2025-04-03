#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import numpy as np
import logging
import torch
import segmentation_models_pytorch as smp

from models.deeplabv3plus import create_model
from train import train
from predict import predict

def setup_logging(log_dir='logs'):
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
            logging.FileHandler(os.path.join(log_dir, 'main.log')),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def parse_args():
    """
    Parse command line arguments
    
    Returns:
        args: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Road Segmentation from Satellite Images')
    
    # Required arguments
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'predict'], 
                        help='Mode to run the script in')
    parser.add_argument('--data_dir', type=str, required=True, 
                        help='Directory containing the data')
    parser.add_argument('--output_dir', type=str, required=True, 
                        help='Directory to save outputs')
    
    # Optional arguments
    parser.add_argument('--model_path', type=str, 
                        help='Path to the model checkpoint (required for prediction)')
    parser.add_argument('--batch_size', type=int, default=4, 
                        help='Batch size for training and validation')
    parser.add_argument('--epochs', type=int, default=40, 
                        help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.00001, 
                        help='Learning rate for the optimizer')
    parser.add_argument('--encoder_name', type=str, default='resnet50', 
                        help='Name of the encoder backbone')
    parser.add_argument('--encoder_weights', type=str, default='imagenet', 
                        help='Name of the weights to use')
    parser.add_argument('--target_size', type=int, nargs=2, default=[1024, 1024], 
                        help='Target size for the images and masks')
    parser.add_argument('--num_workers', type=int, default=0, 
                        help='Number of workers for data loading')
    parser.add_argument('--num_samples', type=int, default=5, 
                        help='Number of random samples to visualize in prediction mode')
    
    # Resume training arguments
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    
    # Learning rate scheduler arguments
    parser.add_argument('--lr_scheduler', type=str, choices=['step', 'plateau', 'cosine'], default=None,
                        help='Type of learning rate scheduler to use')
    parser.add_argument('--step_size', type=int, default=10,
                        help='Number of epochs between learning rate decay (for StepLR)')
    parser.add_argument('--patience', type=int, default=5,
                        help='Number of epochs with no improvement after which LR will be reduced (for ReduceLROnPlateau)')
    parser.add_argument('--t_max', type=int, default=10,
                        help='Maximum number of iterations for cosine annealing (for CosineAnnealingLR)')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='Multiplicative factor of learning rate decay (for StepLR and ReduceLROnPlateau)')
    
    return parser.parse_args()

def main():
    """
    Main function
    """
    # Parse arguments
    args = parse_args()
    
    # Set up logging
    logger = setup_logging(os.path.join(args.output_dir, 'logs'))
    logger.info(f"Starting in {args.mode} mode")
    
    # Check if output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Get classes and RGB values
    try:
        class_dict = pd.read_csv(os.path.join(args.data_dir, 'class_dict.csv'))
        class_names = class_dict['name'].tolist()
        class_rgb_values = class_dict[['r', 'g', 'b']].values.tolist()
        
        # Select specific classes (in this case, we're only interested in roads)
        select_classes = ['background', 'road']
        select_class_indices = [class_names.index(cls.lower()) for cls in select_classes]
        select_class_rgb_values = np.array(class_rgb_values)[select_class_indices]
        
        logger.info(f"Selected classes: {select_classes}")
        logger.info(f"Selected RGB values: {select_class_rgb_values.tolist()}")
    except FileNotFoundError:
        logger.warning("class_dict.csv not found, using default values")
        select_classes = ['background', 'road']
        select_class_rgb_values = np.array([[0, 0, 0], [255, 255, 255]])
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    if args.mode == 'train':
        # Train model
        logger.info("Starting training")
        train(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            encoder_name=args.encoder_name,
            encoder_weights=args.encoder_weights,
            classes=select_classes,
            class_rgb_values=select_class_rgb_values,
            target_size=tuple(args.target_size),
            device=device,
            num_workers=args.num_workers,
            resume_from=args.resume_from,
            lr_scheduler=args.lr_scheduler,
            step_size=args.step_size,
            patience=args.patience,
            t_max=args.t_max,
            gamma=args.gamma
        )
    elif args.mode == 'predict':
        # Check if model path is provided
        if not args.model_path:
            # Try to find the model in the output directory
            model_dir = os.path.join(args.output_dir, 'models')
            if os.path.exists(model_dir):
                model_files = [f for f in os.listdir(model_dir) if f.startswith('best_model_')]
                if model_files:
                    args.model_path = os.path.join(model_dir, model_files[0])
                    logger.info(f"Found model: {args.model_path}")
            
            if not args.model_path:
                logger.error("Model path is required for prediction mode")
                return
        
        # Create model to get preprocessing function
        _, preprocessing_fn = create_model(
            encoder_name=args.encoder_name,
            encoder_weights=args.encoder_weights,
            classes=len(select_classes),
            activation='sigmoid'
        )
        
        # Predict
        logger.info("Starting prediction")
        predict(
            data_dir=args.data_dir,
            model_path=args.model_path,
            output_dir=args.output_dir,
            classes=select_classes,
            class_rgb_values=select_class_rgb_values,
            preprocessing_fn=preprocessing_fn,
            target_size=tuple(args.target_size),
            device=device,
            num_samples=args.num_samples
        )
    
    logger.info(f"{args.mode.capitalize()} completed!")

if __name__ == "__main__":
    # Guard to prevent multiprocessing issues
    main()