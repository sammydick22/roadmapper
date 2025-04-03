# Road Segmentation from Satellite Images

This project implements road segmentation from satellite imagery using DeepLabV3+ architecture. The code is structured as a Python package for better maintainability and to avoid multiprocessing issues.

## Overview

Road segmentation is the task of identifying road networks in satellite or aerial imagery. This project uses DeepLabV3+ with a ResNet50 encoder to perform semantic segmentation of roads in the DeepGlobe Road Extraction Challenge dataset.

## Features

- Data loading and preprocessing with proper augmentation
- DeepLabV3+ model implementation with pretrained encoders
- Training pipeline with metrics tracking
- Prediction and visualization capabilities
- Modular code structure suitable for production environments

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd road_segmentation
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Directory Structure

```
road_segmentation/
├── data/               # Data handling and dataset classes
│   ├── __init__.py
│   └── dataset.py      # RoadsDataset class & preprocessing functions
│
├── models/             # Model definitions
│   ├── __init__.py
│   └── deeplabv3plus.py # DeepLabV3+ model configurations
│
├── utils/              # Utility functions
│   ├── __init__.py
│   ├── encoding.py     # One-hot encoding functions
│   └── visualization.py # Visualization functions
│
├── train.py            # Training script
├── predict.py          # Prediction script
├── main.py             # Main entry point
└── requirements.txt    # Project dependencies
```

## Usage

### Training

To train the model:

```bash
python main.py --mode train --data_dir /path/to/data --output_dir /path/to/output \
    --batch_size 4 --epochs 40 --learning_rate 0.0001
```

### Prediction

To make predictions using a trained model:

```bash
python main.py --mode predict --data_dir /path/to/data \
    --output_dir /path/to/output --model_path /path/to/model.pth
```

### Arguments

- `--mode`: Operation mode ('train' or 'predict')
- `--data_dir`: Directory containing the dataset
- `--output_dir`: Directory to save outputs
- `--model_path`: Path to a trained model (for prediction mode)
- `--batch_size`: Batch size for training
- `--epochs`: Number of training epochs
- `--learning_rate`: Learning rate for the optimizer
- `--encoder_name`: Encoder backbone name
- `--target_size`: Target size for images (width height)
- `--num_workers`: Number of worker processes for data loading
- `--num_samples`: Number of samples to visualize in prediction mode

## Data Organization

The dataset should be organized as follows:

```
data_dir/
├── metadata.csv        # Information about images and masks
├── class_dict.csv      # Class names and RGB values
├── satellite_images/   # Directory containing satellite images
└── masks/              # Directory containing ground truth masks
```

## Note on Multiprocessing

By default, the package sets `num_workers=0` to avoid multiprocessing issues in Jupyter notebooks. For better performance in production environments, you can increase this value based on your system capabilities.

## Requirements

See `requirements.txt` for detailed dependencies.
