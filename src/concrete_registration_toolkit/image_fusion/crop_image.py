import numpy as np

def crop_image(image:np.ndarray, mask:np.ndarray):
    """
    Crops an image based on non-zero coordinates in the mask
    """
    # Extract nonzero coordinates
    x,y = np.nonzero(mask)
    if x.any() and y.any():
        xl, xr = x.min(), x.max()
        yl, yr = y.min(), y.max()

        # crop the image via index slicing
        return image[xl : xr+1, yl : yr+1]
    return image

