import sys
import cv2 as cv


def compute_and_apply_homography(
    fixed_features,
    moving_features,
    match,
    moving_frame,
    _hom_config,
    resolution,
    frame_id,
):
    """
    Utility function for performing homography matching with multithreading. Allows for processing batches directly.
    """
    # If there are less than 4 matches we can't perform homography estimation. Therefore we return None.
    if len(match) <= 4:
        return None

    try:
        # Reference feature points
        points_fixed = fixed_features[match[:, 0]]
        # Moving feature points
        points_moved = moving_features[match[:, 1]]

        # Finding the homography between moving image and reference image. We have to pass in the whole homography config
        H, _ = cv.findHomography(points_moved, points_fixed, **_hom_config)

        # Warp the image accordingly
        warped = cv.warpPerspective(moving_frame, H, resolution)

        # Since this is async, return the frame_id together with the homography and warped image.
        return (frame_id, H, warped)

    except Exception as e:
        print(e)
        sys.exit()
