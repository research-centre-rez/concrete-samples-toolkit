import numpy as np
import cv2 as cv

def local_contrast_normalisation(image, kernel_size=31, eps=1e-5):
    """Normalize each pixel by local mean and std."""
    image = image.astype(np.float64)
    local_mean = cv.blur(image, (kernel_size, kernel_size))
    local_sq_mean = cv.blur(image**2, (kernel_size, kernel_size))
    local_std = np.sqrt(np.maximum(local_sq_mean - local_mean**2, 0))
    normalized = (image - local_mean) / (local_std + eps)
    return normalized
