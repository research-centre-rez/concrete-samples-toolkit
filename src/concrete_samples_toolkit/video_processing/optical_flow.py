import sys
import logging
from enum import Enum

from circle_fit.circle_fit import taubinSVD
import cv2 as cv
import numpy as np
from tqdm import tqdm

from ..utils.prep_cap import prep_cap


class Format(Enum):
    MEAN = 0
    MEDIAN = 1


# TODO: I might add the circle centers to utils/ eventually but right now its only being used here
def fit_circle_from_points(points: np.ndarray):
    """
    Given a set of points in cartesian coordinates, try to fit a circle that describes them.
    Args:
        points (np.ndarray): Points in 2D cartesian coordinates
    Returns:
        xc,yc: Center of the found circle
        R: Radius
        error: Residual error of the solution
    """
    xc, yc, R, error = taubinSVD(points)
    return xc, yc, R, error


def estimate_rotation_center_for_each_trajectory(
    trajectories: list[np.ndarray], output_format: str
):
    """
    Given a set of trajectories that we get from optical flow, estimate a center of rotation. Returns their mean or median.
    Args:
        trajectories (list[np.ndarray]): List of trajectories, each one containing an `np.ndarray` that describes the trajectory of the point through the video frames.
        output_format (str): Whether to use median for the estimation or mean. Pass in either 'median' or 'mean'.
    Returns:
        center: Of all the trajectories
        error: Mean / median of the residual errors
    """
    centers = []
    err = []

    for traj in tqdm(trajectories, desc="Finding center of rotation"):
        # Need at least 3 points
        if len(traj) < 3:
            continue

        xc, yc, _, error = fit_circle_from_points(traj)
        centers.append((xc, yc))
        err.append(error)

    err_arr = np.array(err)
    centers_arr = np.array(centers)

    out_format = Format[output_format.upper()]

    if out_format == Format.MEAN:
        center = centers_arr.mean(axis=0)
        error = err_arr.mean()
        return center, error

    center = np.median(centers_arr, axis=0)
    error = np.median(err_arr)

    return center, error


def estimate_rotation_center(trajectories):
    """
    Estimate the center of rotation for a set of trajectories.

    Args:
        trajectories (list): List of trajectory arrays with shape (n_frames, 2)

    Returns:
        tuple: (center_x, center_y) - the estimated center of rotation
        float: redidual error for each trajectory
    """
    # Combine all trajectory points for initial guess
    all_points = np.vstack([traj for traj in trajectories])

    xc, yc, _, error = taubinSVD(all_points)

    center = (xc, yc)

    return center, error


def calculate_angular_movement(trajectories, center):
    """
    Calculates angular movement for optical flow around a center of rotation. The coordinates are in the coordinate system of the video frame (i.e. [0,0] is left top corner, first coordinate corresponds with the x-axis).
    """
    center_x, center_y = center

    max_frames = max(len(traj) for traj in trajectories)

    frame_angles = [[] for _ in range(max_frames - 1)]
    all_angles = []  # list of all trajectories and their respective angle changes

    for trajectory in trajectories:
        if len(trajectory) < 2:
            continue

        dx = trajectory[:, 0] - center_x
        dy = trajectory[:, 1] - center_y

        angles = np.arctan2(dy, dx)

        angle_changes = np.diff(np.unwrap(angles))

        for i, change in enumerate(angle_changes):
            frame_angles[i].append(change)

        all_angles.extend(angle_changes)

    avg_angle_per_frame = [0.0]
    var_angle_per_frame = [0.0]
    med_angle_per_frame = [0.0]

    for angles in frame_angles:
        if angles:
            avg_angle = np.mean(angles)
            var_angle = np.var(angles)
            med_angle = np.median(angles)
        else:
            avg_angle = np.nan
            var_angle = np.nan
            med_angle = np.nan

        avg_angle_per_frame.append(avg_angle)
        var_angle_per_frame.append(var_angle)
        med_angle_per_frame.append(med_angle)


    cum_angles = np.nancumsum(med_angle_per_frame)

    avg_angle_per_frame_deg = [angle * 180 / np.pi for angle in avg_angle_per_frame]
    med_angle_per_frame_deg = [angle * 180 / np.pi for angle in med_angle_per_frame] 
    cum_angles_deg = cum_angles * 180 / np.pi

    # These results are later used for drawing graphs
    results = {
        "center": (center_x, center_y),
        "average_angle_per_frame_deg": avg_angle_per_frame_deg,
        "median_angle_per_frame_deg": med_angle_per_frame_deg,
        "variance_angle_per_frame": var_angle_per_frame,
        "cumulative_angles_deg": cum_angles_deg,
        "mean_angular_velocity_deg_per_frame": np.nanmean(avg_angle_per_frame_deg),
        "median_angular_velocity_deg_per_frame": np.nanmedian(avg_angle_per_frame_deg),
        "total_rotation_deg": cum_angles_deg[-1],
        "frame_count": max_frames,
    }
    return results


def analyse_sparse_optical_flow(
    vid_path: str, start_at: int, f_params: dict, lk_params: dict
) -> list[np.ndarray]:
    """
    Analyses optical flow in a video by tracking feature points. Can be used for registered and non-registered videos. In a registered video this will effectively measure how well is the video registered, in a non-registered video it will measure the optical flow of the video.

    Args:
        video_path (str): Path to the video
        start_at (int): Frame where optical flow analysis will be started from
        f_params (dict): Parameters for feature detection
        lk_params (dict): Parameters for the Lukas-Kanade optical flow

    Returns:
        List of all trajectories
    """

    # Init logger
    logger = logging.getLogger(__name__)
    cap = prep_cap(vid_path, start_at)

    ret, first_frame = cap.read()
    if not ret:
        logger.error("Error with reading the first frame of video")
        sys.exit(-1)

    # Convert first frame to grayscale
    first_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)

    # Feature params
    if f_params is None:
        logger.error("No configuration was passed for detecting features.")
        sys.exit(-1)

    # Lukas-kane sparse optical flow params
    if lk_params is None:
        logger.error("No configuration was passed for Lukas-Kanade optical flow")
        sys.exit(-1)

    # Detecting feature points in the first frame
    corners = cv.goodFeaturesToTrack(first_gray, mask=None, **f_params)

    trajectories = [[] for _ in range(len(corners))]
    for i, corner in enumerate(corners):
        x, y = corner.ravel()
        trajectories[i].append((x, y))

    frame_count = 0
    prev_gray = first_gray

    with tqdm(
        desc="Calculating optical flow", total=int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    ) as pbar:
        i = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            new_corners, status, error = cv.calcOpticalFlowPyrLK(
                prev_gray, gray, corners, None, **lk_params
            )

            mean_error = np.mean(error[status == 1])
            pbar.set_postfix(mean_error=f"{mean_error:.4f}")

            # Select good points
            good_new_corners = new_corners[status == 1]

            # Store trajectories
            for i, new in enumerate(good_new_corners):
                if i < len(trajectories):
                    x, y = new.ravel()
                    trajectories[i].append((x, y))

            prev_gray = gray.copy()
            corners = good_new_corners.reshape(-1, 1, 2)

            pbar.update(1)
            i += 1

        pbar.close()
        cap.release()

    # Convert to a list of np arrays for analysis
    # Not converting directly to an np array because it is
    # not guaranteed that all trajectories have the same length
    np_trajectories = []
    for trajectory in trajectories:
        logger.debug("Length of trajectory %s", len(trajectory))
        if len(trajectory) > 0:
            np_trajectories.append(np.array(trajectory))

    return np_trajectories
