import cv2
import numpy as np

import cv2
import numpy as np

def filter_binary_mask(binary_mask):
    """
    Strips all components except the two largest islands of pixels.
    """
    # 1. Ensure mask is binary uint8
    binary_mask = (binary_mask > 0).astype(np.uint8) * 255

    # 2. Label all connected components
    # labels: map of the islands; stats: contains area at cv2.CC_STAT_AREA
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

    # If there are 2 or fewer components (including background), return as is
    if num_labels <= 2:
        return binary_mask
    
    # 3. Sort indices by area, excluding the background (index 0)
    # stats[1:, 4] is the Area column for all components except background
    # argsort gives indices of sorted areas; [::-1] makes it descending
    areas = stats[1:, cv2.CC_STAT_AREA]
    sorted_indices = np.argsort(areas)[::-1]
    
    # Take the top 2 indices and add 1 to map back to original label IDs
    top_2_labels = sorted_indices[:2] + 1

    # 4. Create output mask and fill with only the top 2
    cleaned_mask = np.zeros_like(binary_mask)
    for label_id in top_2_labels:
        cleaned_mask[labels == label_id] = 255

    return cleaned_mask

def erode_mask(binary_mask, iterations=1, kernel_size=3):
    """
    Shrinks the binary mask to remove excess borders.
    
    Args:
        binary_mask: 2D numpy array (0/1 or 0/255).
        iterations: Number of pixels to shave off from the edges.
        kernel_size: The size of the kernel used for erosion.
        
    Returns:
        The eroded (smaller) binary mask.
    """
    #Ensure input is uint8
    mask_uint8 = binary_mask.astype(np.uint8)
    
    #Create the kernel 
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    #Apply erosion
    #iterations=1 removes 1 pixel from the boundary
    eroded_mask = cv2.erode(mask_uint8, kernel, iterations=iterations)
    
    return eroded_mask