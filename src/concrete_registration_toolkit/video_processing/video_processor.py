from enum import Enum
import os
import logging
import csv
import cv2 as cv
import numpy as np
from tqdm import tqdm

from ..utils.filename_builder import append_file_extension, create_out_filename
from ..utils.pprint import pprint_dict
from ..utils.prep_cap import prep_cap
from ..utils import visualisers

from .optical_flow import (
    analyse_sparse_optical_flow,
    calculate_angular_movement,
    estimate_rotation_center_for_each_trajectory,
)

logger = logging.getLogger(__name__)


class ProcessorMethod(Enum):
    NONE = 0
    OPT_FLOW = 1
    APPROX = 2


class VideoProcessor:
    """
    VideoProcessor class. The role of this class is to take a video and perform video processing on it, mainly rotation correction, downscaling, subsampling and grayscaling the resulting video.The user has three options:
        `ProcessorMethod.NONE`: No rotation correction is performed.
        `ProcessorMethod.APPROX`: Rotation correction with the assumption that each frame is rotated by a fixed angle, around a fixed center of rotation.
        `ProcessorMethod.OPT_FLOW`: Perform optical flow, then do rotation correction based on that. This yields the best results for rotation correction.
    All the other procesing methods such as downscaling, subsampling etc. are method agnostic. They will happen no matter what method you choose.
    """

    def __init__(self, method: ProcessorMethod, config: dict) -> None:
        """
        Init function. Here you set which method you want the processor to use and a corresponding config.
        Args:
            `method` (ProcessorMethod): which method do you want to use for processing videos
            `config` (dict): Config file containing parameters for the video processing
        Returns:
            None
        """
        # Due to some bugs with codecs, we are not able to keep the original 4k resolution
        # Therefore for downsampling this should be set to at least 2
        self.config = config
        self.downscale_factor_out = self.config.get("downscale_factor")
        self.sampling_rate_out = self.config.get("sampling_rate")
        self.to_grayscale_out = self.config.get("grayscale")
        self.start_at_frame = self.config.get("start_at")
        self.fps_out = self.config.get("fps_out")

        if method == ProcessorMethod.OPT_FLOW:
            self.method = self._get_optical_flow_analysis
        elif method == ProcessorMethod.APPROX:
            self.method = self._get_estimate_analysis
        elif method == ProcessorMethod.NONE:
            self.method = self._get_none_analysis

    def get_rotation_analysis(self, video_path: str) -> dict:
        """
        Calculates rotation analysis for a given video, based on what is specified at initialisation.
        Args:
            video_path (str): path to the video to be analysed
        Returns:
            A dictionary with two keys: `center` and `angles`. These contain the rotation centre and the rotation angles for each frame in the input video.
        """

        center, angles = self.method(video_path)
        analysis = {"center": center, "angles": angles}

        return analysis

    def write_out_video(self, source_path: str, analysis: dict, save_as: str) -> None:
        """
        Corrects the rotation of `source_path` with the use of `analysis`, saving the result as `save_as`
        Args:
            source_path (str): Path to the input video
            analysis (dict): Analysis that is obtained from `self.get_rotation_analysis`
            save_as (str): How do you want to save the new video.
        Returns:
            None. However it does save the new video and also flattened homography transformations as a csv file.
        """

        angle_idx = 0
        angle = np.float64(0.0)

        src_capture = prep_cap(source_path, self.start_at_frame)
        total_frame_count = int(
            src_capture.get(cv.CAP_PROP_FRAME_COUNT) - self.start_at_frame
        )

        assert (
            len(analysis["angles"]) == total_frame_count
        ), "Length of angles is not the same as video length"

        path, _ = os.path.split(save_as)
        os.makedirs(path, exist_ok=True)

        # Information about dimensions
        capture_width = int(src_capture.get(cv.CAP_PROP_FRAME_WIDTH))
        capture_height = int(src_capture.get(cv.CAP_PROP_FRAME_HEIGHT))
        frame_width_out = int(capture_width // self.downscale_factor_out)
        frame_height_out = int(capture_height // self.downscale_factor_out)

        # Get cv.VideoWriter
        fourcc = cv.VideoWriter_fourcc(*"mp4v")
        writer_out = cv.VideoWriter(
            save_as, fourcc, self.fps_out, (frame_width_out, frame_height_out)
        )

        # Write out the processed video
        transformations = []
        var = np.var(analysis["angles"])
        with tqdm(desc="Correcting rotation", total=total_frame_count) as pbar:
            i = 0
            while True:
                ret, frame = src_capture.read()

                if not ret or angle_idx >= len(analysis["angles"]):
                    break

                angle += analysis["angles"][angle_idx]
                angle_idx += 1

                if self.sampling_rate_out != 1 and i % self.sampling_rate_out != 0:
                    i += 1
                    pbar.update(1)
                    continue

                if self.to_grayscale_out:
                    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                    frame = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)

                M = cv.getRotationMatrix2D(
                    center=analysis["center"], angle=angle, scale=1
                )
                rotated_frame = cv.warpAffine(frame, M, (capture_width, capture_height))
                scaled_frame = cv.resize(
                    rotated_frame, (frame_width_out, frame_height_out)
                )

                writer_out.write(scaled_frame)
                transformations.append(M)
                pbar.set_postfix(angle=f"{angle:.4f}")
                pbar.update(1)
                i += 1

        writer_out.release()
        src_capture.release()
        logger.info("Video saved in %s", save_as)
        self._write_trans_into_csv(transformations, save_as)

    def _get_optical_flow_analysis(self, video_path):
        """
        Performs optical flow analysis on the the video.
        """
        opt_flow_params = self.config.get("opt_flow_params")
        f_params = opt_flow_params["f_params"]
        lk_params = opt_flow_params["lk_params"]

        np_trajectories = analyse_sparse_optical_flow(
            video_path,
            lk_params=lk_params,
            f_params=f_params,
            start_at=self.start_at_frame,
        )

        center, quality = estimate_rotation_center_for_each_trajectory(
            np_trajectories, "median"
        )
        logger.info("Estimated rotation center: (%s, %s)", center[0], center[1])
        logger.info("Center quality metric: %s (lower is better)", quality)

        rotation_res = calculate_angular_movement(np_trajectories, center)

        _, name = os.path.split(video_path)
        base, _ = os.path.splitext(name)

        graph_config = {
            "save_as": create_out_filename(f'./_images/{base}', [], ["optical", "flow", "analysis"]),
            "sample_name": name,
            "save": False,
            "show": False,
        }
        logger.info("Saving rotation analysis graph")
        visualisers.visualize_rotation_analysis(
            np_trajectories, rotation_res, graph_config=graph_config
        )
        angles = rotation_res["median_angle_per_frame_deg"]

        return center, angles

    def _get_none_analysis(self, video_path):
        """
        Performs basic video processing (subsampling, downscale etc..). In order to not copy code, it gets passed into _rotated_around_center with 0 angle rotation.
        """
        cap = prep_cap(video_path, 0)
        total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT) - self.start_at_frame)
        center = (0, 0)
        angles = [0 for _ in range(total_frames)]

        return center, angles

    def _get_estimate_analysis(self, video_path):
        """
        Rotation around a fixed center, under the assumption that each frame is rotated by a fixed angle.
        """
        estimate_params = self.config.get("rough_rotation_estimation")

        rotation_center = estimate_params["rotation_center"]
        rotation_center = np.array((rotation_center["x"], rotation_center["y"]))
        rot_per_frame = estimate_params["rotation_per_frame"]

        cap = prep_cap(video_path)
        frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT)) - self.start_at_frame

        logger.info("Rotation center: %s", rotation_center)
        logger.info("Rotation per frame: %s", rot_per_frame)

        angles = [0]
        angles.extend([rot_per_frame for _ in range(frame_count - 1)])

        return rotation_center, angles

    def _write_trans_into_csv(self, transformations: list[np.ndarray], out_file_name):
        """
        Writes out the transformation matrices into a csv file.
        """
        base, _ = os.path.splitext(out_file_name)

        save_as = create_out_filename(base, [], ["transformations"])
        save_as = append_file_extension(save_as, ".csv")

        # TODO: Add a description of this to README
        with open(save_as, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame_id", "m00", "m01", "m02", "m10", "m11", "m12"])

            for i, M in enumerate(transformations):
                writer.writerow([i] + M.ravel().tolist())

        logger.info("Stored csv data about transformations in %s", save_as)
