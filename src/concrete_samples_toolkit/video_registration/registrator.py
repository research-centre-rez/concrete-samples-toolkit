from concurrent.futures import ProcessPoolExecutor, as_completed
from enum import Enum
import logging
import os
import sys
import csv
from typing import Callable
import numpy as np
import cv2 as cv
from cv2 import ORB
import muDIC as dic
from lightglue.utils import Extractor, numpy_image_to_torch
from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED, DoGHardNet
import torch
import torch.nn.functional as F

from skimage.measure import EllipseModel
from tqdm import tqdm
from .mudic_utils import extract_medians
from ..utils.filename_builder import append_file_extension, create_out_filename
from ..utils.prep_cap import prep_cap

from .mt_homography import compute_and_apply_homography
from .mudic_utils import correlate_matrix, create_mesh
from .video_matrix import create_video_matrix

logger = logging.getLogger(__name__)


class RegMethod(Enum):
    ORB = 0
    MUDIC = 1
    LIGHTGLUE = 2


class VideoRegistrator:
    """
    Video registration class. All of the configuration for this class can be found in `./default_config.json5`.
    There are 3 main ways of performing registration:
        1. ORB: A feature-based method that utilises ORB for registering a video.
        2. MUDIC: An area-based method using a digital image correlation library called muDIC.
        3. LIGHTGLUE: A deep-learning based method.
    """

    def __init__(self, method: RegMethod, config: dict) -> None:
        """
        Init method for video registration.
        Args:
            method (RegMethod): Tells the class which method you want to use for performing video registration
            config (dict): A validated configuration dictionary that specifies parameters for each method
        Returns:
            None
        """

        self.config = config

        self.method: Callable[[str], tuple[np.ndarray, np.ndarray]]
        self.extractor = None
        self.matcher = None

        if method == RegMethod.ORB:
            self.method = self._get_orb_registration

        elif method == RegMethod.MUDIC:
            self.method = self._get_mudic_registration

        elif method == RegMethod.LIGHTGLUE:
            self.method = self._get_lightglue_registration

    def set_method(self, new_method: RegMethod) -> None:
        """
        Changes the internal method of the registrator to `new_method`
        """
        if new_method == RegMethod.ORB:
            self.method = self._get_orb_registration

        elif new_method == RegMethod.MUDIC:
            self.method = self._get_mudic_registration

        elif new_method == RegMethod.LIGHTGLUE:
            self.method = self._get_lightglue_registration

    def save_registered_block(
        self, reg_analysis: dict[str, np.ndarray], save_as: str
    ) -> None:
        """
        Write the registered block and the transformations to disc.
        """
        np.save(save_as, reg_analysis["registered_block"])
        logger.info("Saved registered stack as %s", save_as)
        self._write_transformation_into_csv(reg_analysis["transformations"], save_as)

    def get_registered_block(self, video_input_path: str) -> dict[str, np.ndarray]:
        """
        Performs video registration based on the method that was chosen at the initialisation.

        Args:
            video_input_path (str): path to the video you want to register

        Returns:
            dictionary that contains two fields:
                "registered_block" (np.ndarray): registered video stack
                "transformations" (np.ndarray): transformations that were performed at each step
        """
        logger.info("Processing %s", video_input_path)
        reg_block, transformations = self.method(video_input_path)
        result = {"registered_block": reg_block, "transformations": transformations}
        return result

    def _get_orb_registration(self, input_video: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Performs orb registration with moving the fixed image every few frames.
        """
        _orb_config = self.config.get("ORB_parameters")
        _homography_config = _orb_config["homography"]
        _matcher_config = _orb_config["matcher"]
        _update_every_nth_frame = _orb_config["update_every_frame"]

        if isinstance(_homography_config["method"], str):
            _homography_config["method"] = getattr(cv, _homography_config["method"])
            _matcher_config["normType"] = getattr(cv, _matcher_config["normType"])

        input_cap = prep_cap(input_video, set_to=0)
        frame_w = int(input_cap.get(cv.CAP_PROP_FRAME_WIDTH))
        frame_h = int(input_cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(input_cap.get(cv.CAP_PROP_FRAME_COUNT))

        registered_frames = []
        transformations = []

        orb = ORB.create(**_orb_config["init_params"])
        matcher = cv.BFMatcher.create(**_matcher_config)

        ret, fixed_frame = input_cap.read()
        if not ret:
            raise ValueError("Couldn't load a frame from the video")

        fixed_frame = cv.cvtColor(fixed_frame, cv.COLOR_BGR2GRAY)
        registered_frames.append(fixed_frame)
        transformations.append(np.eye(3))

        fixed_kp, fixed_des = orb.detectAndCompute(fixed_frame, None)

        with tqdm(total=total_frames - 1, desc="Registering with ORB") as pbar:
            i = 0
            while True:
                ret, moving_frame = input_cap.read()
                if not ret:
                    break

                moving_frame = cv.cvtColor(moving_frame, cv.COLOR_BGR2GRAY)
                moving_kp, moving_des = orb.detectAndCompute(moving_frame, None)

                matches = matcher.match(moving_des, fixed_des)
                dist_matches = np.array([m.distance for m in matches])

                # Filter out matches that are too far
                mask = dist_matches <= _orb_config["max_match_distance"]
                indices = np.nonzero(mask)[0]

                pbar.set_postfix(
                    num_matches=f"{len(matches)}", good_matches=f"{len(indices)}"
                )

                matches = [matches[i] for i in indices]

                fixed_pts = np.float32(
                    [fixed_kp[m.trainIdx].pt for m in matches]
                ).reshape(-1, 1, 2)

                moving_pts = np.float32(
                    [moving_kp[m.queryIdx].pt for m in matches]
                ).reshape(-1, 1, 2)

                H, _ = cv.findHomography(moving_pts, fixed_pts, **_homography_config)

                transformations.append(H)

                reg_frame = cv.warpPerspective(moving_frame, H, (frame_w, frame_h))

                # Change the fixed frame
                if i % _update_every_nth_frame == 0:
                    fixed_frame = reg_frame
                    fixed_kp, fixed_des = orb.detectAndCompute(fixed_frame, None)

                registered_frames.append(reg_frame)
                pbar.update(1)
                i += 1

        input_cap.release()
        registered_frames = np.array(registered_frames)
        transformations = np.array(transformations)

        return registered_frames, transformations

    def _get_lightglue_registration(
        self, input_video: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Performs LightGlue registration.
        """
        _lglue_config = self.config.get("lightGlue")
        _homography_config = _lglue_config["homography"]
        _matcher = _lglue_config["matcher"]
        extractor = _lglue_config["extractor"]
        max_number_keypoints = _lglue_config["max_num_keypoints"]
        batch_size = _lglue_config["batch_size"]

        logger.info("Initialising the extractor and matcher")

        if self.extractor is None or self.matcher is None:
            self.extractor, self.matcher = self._get_extractor_matcher(
                extractor_name=extractor,
                matcher_config=_matcher,
                max_num_kp=max_number_keypoints,
            )

        if isinstance(_homography_config["method"], str):
            # Parsing the human-readable string into an opencv enum
            _homography_config["method"] = getattr(cv, _homography_config["method"])

        # Parse the video in a matrix
        vid_stack = create_video_matrix(input_video, grayscale=True, max_gb_memory=8)
        if vid_stack.size == 0:
            sys.exit()

        # First image in the sequence has no transformation
        transformations = np.zeros((len(vid_stack), 3, 3))
        transformations[0] = np.eye(3, 3)

        # The original dimensions of the images
        out_res = (vid_stack[0].shape[1], vid_stack[0].shape[0])

        fixed_image = numpy_image_to_torch(
            cv.cvtColor(vid_stack[0], cv.COLOR_GRAY2RGB)
        ).cuda()
        fixed_feats = self.extractor.extract(fixed_image)

        # Copy over the fixed features for performing parallel kp extraction
        fixed_keypoints = fixed_feats["keypoints"]
        fixed_descriptors = fixed_feats["descriptors"]
        image_size = fixed_feats["image_size"]
        fixed_keypoints = fixed_keypoints.repeat((batch_size, 1, 1))
        fixed_descriptors = fixed_descriptors.repeat((batch_size, 1, 1))
        image_size = image_size.repeat((batch_size, 1))
        fixed_feats = {
            "keypoints": fixed_keypoints,
            "descriptors": fixed_descriptors,
            "image_size": image_size,
        }

        futures = []

        with torch.no_grad(), ProcessPoolExecutor(max_workers=None) as executor, tqdm(
            total=len(vid_stack[1:]), desc="Glueing"
        ) as pbar:
            for i in range(1, len(vid_stack[1:]), batch_size):
                # Batch up the the images
                moving_feats = self._create_extracted_batch(
                    vid_stack, i, batch_size
                )
                matches = self.matcher({"image0": fixed_feats, "image1": moving_feats})

                cpu_fixed_kps = fixed_feats["keypoints"].cpu().numpy()
                cpu_moving_kps = moving_feats["keypoints"].cpu().numpy()

                for idx, match in enumerate(matches["matches"]):
                    if idx + i < len(vid_stack):
                        futures.append(
                            executor.submit(
                                compute_and_apply_homography,
                                cpu_fixed_kps[idx],
                                cpu_moving_kps[idx],
                                match.cpu().numpy(),
                                vid_stack[idx + i],
                                _homography_config,
                                out_res,
                                idx + i,
                            )
                        )
                        pbar.update(1)
                    else:
                        break

            # Because we are doing homography asynchronously, we place the warped
            # images into the resulting matrix directly
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    frame_idx, H, warped = result
                    transformations[frame_idx] = H
                    vid_stack[frame_idx] = warped

        return vid_stack, transformations

    def _get_mudic_registration(
        self, input_video: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Performs video registration with the use of muDIC.
        """

        mudic_config = self.config.get("mudic")

        vid_stack = create_video_matrix(input_video, grayscale=True)
        image_stack = dic.image_stack_from_list(list(vid_stack))

        frame_h, frame_w = vid_stack.shape[1:]

        mesh = create_mesh(
            frame_h, frame_w, image_stack, **mudic_config.get("mesh_parameters")
        )

        displacement = correlate_matrix(
            image_stack, mesh, mudic_config.get("ref_range"), mudic_config.get("max_it")
        )
        displacement = displacement.squeeze()
        meds = extract_medians(displacement)

        return self._shift_by_vector(vid_stack, meds)

    def _shift_by_vector(
        self, image_stack, displacement
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Performs the registration for muDIC.
        """
        n, h, w = image_stack.shape

        x_c, y_c = self._fit_ellipse(displacement)
        transformations = [np.eye(3, 3)]

        for i in tqdm(range(1, n), desc="Registering by shift"):
            image = image_stack[i]
            x_d, y_d = displacement[i]

            T = np.array([[1, 0, -x_d + x_c], [0, 1, -y_d + y_c]])
            transformations.append(np.vstack([T, [0, 0, 1]]))

            im_translated = cv.warpAffine(image, T, (w, h))
            image_stack[i] = im_translated

        return image_stack, np.array(transformations)

    def _fit_ellipse(self, disp: np.ndarray) -> tuple:
        model = EllipseModel()
        success = model.estimate(disp)

        if not success:
            logger.error("failed to find an ellipse")
            sys.exit(-1)  # TODO: Figure out what to do with this

        xc, yc, _, _, _ = model.params

        return xc, yc

    def _write_transformation_into_csv(
        self, trans: list[np.ndarray], save_as: str
    ) -> None:
        """
        Writes the homogenenous transformations into the specified csv file.
        """
        base, _ = os.path.splitext(save_as)

        save_as = create_out_filename(base, [], ["transformations"])
        save_as = append_file_extension(save_as, ".csv")

        with open(save_as, "w", newline="") as f:
            writer = csv.writer(f)

            for i, M in enumerate(trans):
                writer.writerow([i] + M.ravel().tolist())

        logger.info("Stored csv data about transformation in %s", save_as)

    def _create_extracted_batch(
        self, vid_stack, video_index, batch_size
    ) -> dict[str, torch.Tensor]:
        """
        Creates a batch that can be then passed into LightGlue.
        """

        batched_kp = []
        batched_dp = []
        batched_imsize = []
        max_kp = 0
        for j in range(batch_size):
            try:
                moving_image = numpy_image_to_torch(
                    cv.cvtColor(vid_stack[j + video_index], cv.COLOR_GRAY2RGB)
                ).cuda()
            except IndexError as e:
                # Padding with empty image
                moving_image = numpy_image_to_torch(
                    cv.cvtColor(np.zeros_like(vid_stack[0]), cv.COLOR_GRAY2RGB)
                ).cuda()

            moving_feats = self.extractor.extract(moving_image)
            v_kp = moving_feats["keypoints"]
            v_dp = moving_feats["descriptors"]
            image_size = moving_feats["image_size"]
            max_kp = max(max_kp, v_kp.shape[1])
            batched_kp.append(v_kp)
            batched_dp.append(v_dp)
            batched_imsize.append(image_size)

        # Pad the keypoints and descriptors for pytorch
        padded_kp = [F.pad(kp, (0, 0, 0, max_kp - kp.shape[1])) for kp in batched_kp]
        padded_dp = [F.pad(kp, (0, 0, 0, max_kp - kp.shape[1])) for kp in batched_dp]
        stacked_kp = torch.cat(padded_kp)
        stacked_dp = torch.cat(padded_dp)

        moving_feats = {
            "keypoints": stacked_kp,
            "descriptors": stacked_dp,
            "image_size": torch.cat(batched_imsize),
        }
        return moving_feats

    def _get_extractor_matcher(
        self, extractor_name: str, matcher_config: dict, max_num_kp: int
    ) -> tuple[Extractor, LightGlue]:
        """
        Gets the correct extractor and matcher.
        """
        if extractor_name == "SuperPoint":
            extractor = SuperPoint(max_num_keypoints=max_num_kp).eval().cuda()
            matcher = LightGlue(features="superpoint", **matcher_config).eval().cuda()
        elif extractor_name == "DISK":
            extractor = DISK(max_num_keypoints=max_num_kp).eval().cuda()
            matcher = LightGlue(features="disk", **matcher_config).eval().cuda()
        elif extractor_name == "SIFT":
            extractor = SIFT(max_num_keypoints=max_num_kp).eval().cuda()
            matcher = LightGlue(features="sift", **matcher_config).eval().cuda()
        elif extractor_name == "ALIKED":
            extractor = ALIKED(max_num_keypoints=max_num_kp).eval().cuda()
            matcher = LightGlue(features="aliked", **matcher_config).eval().cuda()
        elif extractor_name == "DoGHardNet":
            extractor = DoGHardNet(max_num_keypoints=max_num_kp).eval().cuda()
            matcher = LightGlue(features="doghardnet", **matcher_config).eval().cuda()
        else:
            logger.error("Invalid keypoint extractor given, exiting.")
            sys.exit()

        return extractor, matcher
