import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

def visualize(**images):
    """
    Plot images in one row
    
    Args:
        **images: Dictionary of images to display
    """
    n_images = len(images)
    plt.figure(figsize=(20, 8))
    for idx, (name, image) in enumerate(images.items()):
        plt.subplot(1, n_images, idx + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(name.replace('_', ' ').title(), fontsize=20)
        plt.imshow(image)
    
    return plt.gcf()

def save_visualization(output_dir, filename, **images):
    """
    Save visualization of images to the specified output directory
    
    Args:
        output_dir: Directory to save the visualization
        filename: Name of the file to save
        **images: Dictionary of images to display
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    fig = visualize(**images)
    fig.savefig(os.path.join(output_dir, filename))
    plt.close(fig)
    
    # Also save a concatenated version for easier viewing
    if len(images) > 1:
        # Convert all images to numpy arrays with same height
        img_list = []
        for name, image in images.items():
            # Convert to uint8 if not already
            if image.dtype != np.uint8:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
            
            # Handle grayscale images
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 1:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                
            img_list.append(image)
        
        # Resize all images to same height
        height = min(im.shape[0] for im in img_list)
        resized_imgs = []
        for img in img_list:
            aspect = img.shape[1] / img.shape[0]
            new_width = int(aspect * height)
            resized_imgs.append(cv2.resize(img, (new_width, height)))
        
        # Concatenate horizontally
        concat_image = np.hstack(resized_imgs)
        cv2.imwrite(os.path.join(output_dir, f"concat_{filename}"), concat_image[:, :, ::-1])
