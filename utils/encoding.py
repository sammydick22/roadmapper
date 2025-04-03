import numpy as np

def one_hot_encode(label, label_values):
    """
    Convert a segmentation image label array to one-hot format
    by replacing each pixel value with a vector of length num_classes
    
    Args:
        label: The 2D array segmentation image label
        label_values: List of RGB values corresponding to each class
        
    Returns:
        A 2D array with the same width and height as the input, but
        with a depth size of num_classes
    """
    semantic_map = []
    for colour in label_values:
        equality = np.equal(label, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)
    
    return semantic_map

def reverse_one_hot(image):
    """
    Transform a 2D array in one-hot format (depth is num_classes),
    to a 2D array with only 1 channel, where each pixel value is
    the classified class key.
    
    Args:
        image: The one-hot format image 
        
    Returns:
        A 2D array with the same width and height as the input, but
        with a depth size of 1, where each pixel value is the classified class key.
    """
    x = np.argmax(image, axis=-1)
    return x

def colour_code_segmentation(image, label_values):
    """
    Given a 1-channel array of class keys, colour code the segmentation results.
    
    Args:
        image: single channel array where each value represents the class key.
        label_values: List of RGB values for each class
        
    Returns:
        Colour coded image for segmentation visualization
    """
    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]
    
    return x
