import logging
import os
import sys

from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

from utils.filename_builder import append_file_extension, create_out_filename

# Leclercq's util functions

def show_overlay(saving_dest:str, img:cv.typing.MatLike, mask:cv.typing.MatLike, alpha=0.5) -> None:
    '''
    Saves an overlay of an image and its mask to `saving_dest`. The mask is shown in a green colour.
    Args:
        saving_dest: where to save the output of the image
        img: opencv image
        mask: single channel mask
        alpha: how transparent should the mask be, if 1 then there is no transparency
    '''
    colour_mask = cv.merge([mask * 0, mask, mask * 0])
    overlaid    = cv.addWeighted(img, 1, colour_mask, alpha, 0)

    cv.imwrite(saving_dest, overlaid)

def imshow(title= None, **images) -> None:
    '''
    Displays images in one row.
    Args:
        title: What the title of the plot should be
        **images: the name of the variable determines its title in the graph. 
            - `image` variable is reserved for torch tensors, channels are permuted
            - else it prints an image
    '''

    n = len(images)

    cols = min(n, 3)
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*3), constrained_layout=True)

    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]

    for ax, (name, image) in zip(axes, images.items()):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(' '.join(name.split('_')).title())

        if name == 'image':
            ax.imshow(image.permute(1,2,0))
        else:
            ax.imshow(image, cmap="gray")

    for ax in axes[len(images):]:
        ax.axis('off')

    plt.show()
    plt.pause(0.1)

def plot_optical_flow(trajectories:list[np.ndarray], metrics:dict, graph_config:dict):
    '''
    Plots optical flow and its metrics.

    Args:
        trajectories (list[np.ndarray]): List of point trajectories
        metrics (dict): Dictionary containing trajectory metrics
    '''

    if 'error' in metrics:
        print(metrics['error'])
        return

    fig = plt.figure(figsize=(16,10))

    # Plot 1: X jitter over time for few sample trajectories
    ax1 = fig.add_subplot(221)
    for i, tm in enumerate(metrics["trajectories"][:7]):  # Plot first few trajectories
        ax1.plot(tm["frames"], tm["x_jitter"], '-', label=f'Point {i+1}')

    ax1.set_title('X Jitter Over Time')
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('X Jitter (pixels)')
    ax1.legend()

    # Plot 2: Y jitter over time for few sample trajectories
    ax2 = fig.add_subplot(222)
    for i, tm in enumerate(metrics["trajectories"][:7]):  # Plot first few trajectories
        ax2.plot(tm["frames"], tm["y_jitter"], '-', label=f'Point {i+1}')

    ax2.set_title('Y Jitter Over Time')
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Y Jitter (pixels)')
    ax2.legend()

    # Plot 3: Trajectories
    ax3 = fig.add_subplot(223)
    for i, traj in enumerate(trajectories):
        if len(traj) > 0:
            ax3.plot(traj[:, 0], traj[:, 1], '-', label=f'Point {i+1}')
    ax3.set_title('Feature Point Trajectories')
    ax3.set_xlabel('X position (pixels)')
    ax3.set_ylabel('Y position (pixels)')

    # Plot 4: Jitter magnitude histogram
    ax4 = fig.add_subplot(224)
    all_jitter = np.concatenate([tm["jitter_magnitudes"] for tm in metrics["trajectories"]])
    ax4.hist(all_jitter, bins=30)
    ax4.set_title('Jitter Magnitude Histogram')
    ax4.set_xlabel('Jitter Magnitude (pixels)')
    ax4.set_ylabel('Frequency')

    fig.suptitle(graph_config['title'])
    # Add overall metrics as text
    plt.figtext(0.5, 0.01,
                f"Overall Metrics:\n"
                f"Mean Jitter: {metrics['mean_jitter']:.2f} pixels\n"
                f"Max Jitter: {metrics['max_jitter']:.2f} pixels\n"
                f"Std Jitter: {metrics['std_jitter']:.2f} pixels\n"
                f"Jitter-to-Motion Ratio: {metrics['jitter_to_motion_ratio']:.4f}",
                ha="center", fontsize=12, bbox={"facecolor":"beige", "alpha":1, "pad":2})

    plt.tight_layout()

    if graph_config['save']:
        plt.savefig(graph_config['save_as'], dpi=600)
    if graph_config['show']:
        plt.show()

    plt.close()

def visualize_rotation_analysis(trajectories, rotation_results, frames=None, graph_config=None):
    """
    Visualize the rotation analysis results.
    
    Args:
        trajectories (list): List of trajectory arrays
        rotation_results (dict): Results from calculate_angular_movement
        frames (list): Specific frames to visualize, or None for all
        graph_config (dict): Configuration for plotting
    """
    if graph_config is None:
        graph_config = {
            'title_prefix': 'Rotation Analysis',
            'save_prefix': 'rotation_analysis',
            'save': False,
            'show': True
        }

    # Get center of rotation
    center_x, center_y = rotation_results["center"]

    # Plot 1: Trajectories and rotation center
    fig = plt.figure(figsize=(16,10))
    fig.suptitle(f"Optical flow analysis for {graph_config['sample_name']}")

    ax1 = fig.add_subplot(221)
    # Plot trajectories
    for traj in trajectories:
        ax1.plot(traj[:, 0], traj[:, 1], '-', alpha=0.5, markersize=1)

    # Plot center
    ax1.plot(center_x, center_y, 'r*', markersize=10, label='Estimated Center')

    ax1.axis('equal')
    ax1.set_title("Trajectories and Center")
    ax1.text(0.05, 0.05, f'Center at {center_x, center_y}', transform=ax1.transAxes)
    ax1.set_xlabel('X (px)')
    ax1.set_ylabel('Y (px)')
    ax1.legend()

    # Plot 2: Angular velocity per frame
    ax2 = fig.add_subplot(222)

    # Get angle data
    angles_deg = rotation_results["average_angle_per_frame_deg"]
    med_angle_deg = rotation_results["median_angle_per_frame_deg"]
    angles_var = rotation_results["variance_angle_per_frame"]
    frames_idx = range(len(angles_deg))

    std_dev = np.sqrt(angles_var)

    ax2.plot(frames_idx, angles_deg, 'b-', label='Mean angle per frame')
    ax2.fill_between(frames_idx, angles_deg - std_dev, angles_deg + std_dev, color='blue', alpha=0.2, label='±1 std dev')
    ax2.axhline(y=rotation_results["mean_angular_velocity_deg_per_frame"],
              color='r', linestyle='--',
              label=f'Mean: {rotation_results["mean_angular_velocity_deg_per_frame"]:.5f}° / frame')

    ax2.axhline(y=rotation_results["median_angular_velocity_deg_per_frame"],
              color='g', linestyle='--',
              label=f'Median: {rotation_results["median_angular_velocity_deg_per_frame"]:.5f}° / frame')
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Angular Movement (degrees)')
    ax2.set_title("Angular Velocity")
    ax2.legend()

    # Plot 3: Cumulative rotation
    ax3 = fig.add_subplot(223)

    cum_angles_deg = rotation_results["cumulative_angles_deg"]
    frames_cum = range(len(cum_angles_deg))

    ax3.plot(frames_cum, cum_angles_deg, 'g-')
    ax3.set_xlabel('Frame')
    ax3.set_ylabel('Cumulative Rotation (degrees)')
    ax3.set_title("Cumulative Rotation")

    # Plot 4: Median rotation per frame
    ax4 = fig.add_subplot(224)

    # Get angle data
    med_angle_deg = rotation_results["median_angle_per_frame_deg"]
    angles_var = rotation_results["variance_angle_per_frame"]
    frames_idx = range(len(angles_deg))

    std_dev = np.sqrt(angles_var)

    ax4.plot(frames_idx, med_angle_deg, 'g-', label='Median angle per frame')
    ax4.axhline(y=rotation_results["mean_angular_velocity_deg_per_frame"],
              color='r', linestyle='--',
              label=f'Mean: {rotation_results["mean_angular_velocity_deg_per_frame"]:.5f}° / frame')

    ax4.axhline(y=rotation_results["median_angular_velocity_deg_per_frame"],
             color='g', linestyle='--',
              label=f'Median: {rotation_results["median_angular_velocity_deg_per_frame"]:.5f}° / frame')
    ax4.set_xlabel('Frame')
    ax4.set_ylabel('Median Angular Movement (degrees)')
    ax4.set_title("Angular Velocity")
    ax4.legend()

    if graph_config['save']:
        plt.savefig(graph_config['save_as'], dpi=600)

    if not graph_config['show']:
        plt.close()
    else:
        plt.show()


    # Get angle data
#    angles_deg = rotation_results["average_angle_per_frame_deg"]
#    angles_var = rotation_results["variance_angle_per_frame"]
#    frames_idx = range(len(angles_deg))
#
#    std_dev = np.sqrt(angles_var)
#
#    plt.plot(frames_idx, angles_deg, 'b-', label='Mean angle per frame')
#    plt.fill_between(frames_idx, angles_deg - std_dev, angles_deg + std_dev, color='blue', alpha=0.2, label='±1 std dev')
#    plt.axhline(y=rotation_results["mean_angular_velocity_deg_per_frame"],
#                color='r', linestyle='--',
#                label=f'Mean: {rotation_results["mean_angular_velocity_deg_per_frame"]:.5f}° / frame')
#
#    plt.axhline(y=rotation_results["median_angular_velocity_deg_per_frame"],
#                color='g', linestyle='--',
#                label=f'Median: {rotation_results["median_angular_velocity_deg_per_frame"]:.5f}° / frame')
#    plt.xlabel('Frame')
#    plt.ylabel('Angular Movement (degrees)')
#    plt.title("Angular Velocity")
#    plt.legend()
    plt.show()
