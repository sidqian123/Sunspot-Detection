from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from scipy.ndimage import binary_erosion, generate_binary_structure, binary_fill_holes
from skimage.filters import threshold_otsu
from scipy.ndimage import binary_erosion, generate_binary_structure, label


def find_outer_contour(image_data, min_size=50):
    # Otsu's threshold
    thresh = threshold_otsu(image_data)
    outer_boundary = image_data < thresh

    structure = generate_binary_structure(2, 2)
    labeled_array, num_features = label(outer_boundary, structure)
    
    # filter out small objects
    outer_contour = np.zeros_like(outer_boundary, dtype=bool)
    
    # Iterate over all connected components
    for region_num in range(1, num_features + 1):
        region = (labeled_array == region_num)
        region_size = np.sum(region)
        
        if region_size > min_size:
            eroded_region = binary_erosion(region, structure)
            region_contour = region & (~eroded_region)
            outer_contour |= region_contour  
            # Combine contours   
    return outer_contour


def find_inner_contour(image_data, outer_contour):
    outer_mask = binary_fill_holes(outer_contour)
    sunspot_region = image_data[outer_mask]
    
    # Apply Otsu's 
    if len(sunspot_region) > 0: 
        umbra_thresh = threshold_otsu(sunspot_region)
        umbra_mask = image_data < umbra_thresh
        
        # Ensure the umbra mask is only within the sunspot area
        umbra_mask[~outer_mask] = False
        
        # Erode the umbra boundary to get the contour
        structure = generate_binary_structure(2, 2)
        eroded_umbra_mask = binary_erosion(umbra_mask, structure)
        inner_contour = umbra_mask & (~eroded_umbra_mask)
        
        return inner_contour
    else:
        return np.zeros_like(image_data, dtype=bool)
    
def main():
    image_path = 'test_image.png'
    image = Image.open(image_path).convert('L')
    image_data = np.array(image)
    outer_contour = find_outer_contour(image_data)
    inner_contour = find_inner_contour(image_data, outer_contour)
    color_overlay = np.stack((image_data,) * 3, axis=-1)
    color_overlay[outer_contour] = [255, 0, 0]  
    color_overlay[inner_contour] = [0, 255, 0] 
    final_image = Image.fromarray(color_overlay)
    final_image.show()

if __name__ == '__main__':
    main()