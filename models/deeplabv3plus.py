import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils as smp_utils

def create_model(
    encoder_name='resnet50',
    encoder_weights='imagenet',
    classes=2,
    activation='sigmoid',
):
    """
    Create a DeepLabV3+ model with the specified parameters
    
    Args:
        encoder_name: Name of the encoder backbone (default: 'resnet50')
        encoder_weights: Name of the weights to use (default: 'imagenet')
        classes: Number of classes (default: 2)
        activation: Activation function (default: 'sigmoid')
        
    Returns:
        model: DeepLabV3+ model
        preprocessing_fn: Preprocessing function for the encoder
    """
    # Create segmentation model with pretrained encoder
    model = smp.DeepLabV3Plus(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        classes=classes,
        activation=activation,
    )
    
    # Get preprocessing function for the encoder
    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder_name, encoder_weights)
    
    return model, preprocessing_fn

def get_loss():
    """
    Get the loss function for segmentation
    
    Returns:
        loss: Loss function
    """
    return smp_utils.losses.DiceLoss()

def get_metrics():
    """
    Get the metrics for segmentation
    
    Returns:
        metrics: List of metrics
    """
    return [
        smp_utils.metrics.IoU(threshold=0.5),
    ]
