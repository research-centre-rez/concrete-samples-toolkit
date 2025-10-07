import sys
from enum import Enum, auto
import logging
import os
from typing import Union
import numpy as np
import cv2 as cv

from ..utils import pprint
from ..utils.filename_builder import append_file_extension, create_out_filename

logger = logging.getLogger(__name__)


class FuseMethod(Enum):
    MIN = auto()
    MAX = auto()
    VAR = auto()
    MED = auto()
    MEAN = auto()


class Fuser:
    """
    Default Fuser class, each fuse method needs to implement the `get_fused_image` on its own accord.
    """

    def __init__(self, method: FuseMethod) -> None:
        """
        Init method, the name of the fuse method is used for writing images and for logging activity.
        """
        self.method = method
        self.method_name = method.name

    def get_fused_image(self, video_stack: Union[str, np.ndarray]) -> np.ndarray:
        """
        Subclasses implement this.
        """
        raise NotImplementedError("Each fuser should implement this on its own")

    def save_image_to_disc(self, fused_image: np.ndarray, save_as: str) -> None:
        """
        Takes a fused image and saves it to disc. Appends the name of the fuse method at the end of the file.
        """
        path, name = os.path.split(save_as)
        base, ext = os.path.splitext(name)

        # Gracefully wrong file extensions
        if ext is None or ext not in [".png"]:
            logger.warning(
                "Invalid extension, must be one of %s, but instead got: %s. Setting extension to be '.png'"
            )
            ext = ".png"

        out_filename = os.path.join(path, base)
        out_filename = create_out_filename(out_filename, [], [self.method_name])
        out_filename = append_file_extension(out_filename, ext)

        os.makedirs(path, exist_ok=True)
        cv.imwrite(out_filename, fused_image)
        logger.info("Successfully saved image stack as %s", out_filename)

    def _log_activity(self):
        """
        Print what image is being extracted.
        """
        logger.info("Extracting %s image", self.method_name)

    def _verify_video_stack(self, video_stack: Union[str, np.ndarray]) -> np.ndarray:
        """
        Utility function which allows for taking in a path to a video stack or an `np.ndarray` video stack.
        """
        if isinstance(video_stack, str):
            return np.load(video_stack)

        if not isinstance(video_stack, np.ndarray):
            raise TypeError(
                f"Input must be a file path (str) or a NumPy array. Got: {type(video_stack)}"
            )

        return video_stack


class MinFuser(Fuser):
    """
    Take the min value of pixels across the image stack.
    """

    def __init__(self) -> None:
        super().__init__(FuseMethod.MIN)

    def get_fused_image(self, video_stack: Union[str, np.ndarray]) -> np.ndarray:
        verified_stack = self._verify_video_stack(video_stack)
        self._log_activity()
        return verified_stack.min(axis=0)

    def get_min_mask(self, video_stack: Union[str, np.ndarray]) -> np.ndarray:
        """
        The min image can be used to create a mask that is centered on the concrete sample.
        """

        # First, we get the min image
        min_img = self.get_fused_image(self._verify_video_stack(video_stack))

        # Create a mask
        mask = (min_img > 0).astype(dtype=np.uint8)
        kernel_size = (5, 5)
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, kernel_size)
        morphed_mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
        return morphed_mask


class MaxFuser(Fuser):
    """
    Take the max value of pixels across the image stack.
    """

    def __init__(self) -> None:
        super().__init__(FuseMethod.MAX)

    def get_fused_image(self, video_stack: Union[str, np.ndarray]) -> np.ndarray:
        verified_stack = self._verify_video_stack(video_stack)
        self._log_activity()
        return verified_stack.max(axis=0)

    def get_fused_image_(self, video_stack: Union[str, np.ndarray]) -> np.ndarray:
        verified_stack = self._verify_video_stack(video_stack)
        self._log_activity()

        fused = verified_stack.max(axis=0)

        # Compute mean intensity of the whole stack and of the fused image
        stack_mean = verified_stack.mean()
        fused_mean = fused.mean()

        # Scale fused image to match original stack brightness
        scale = stack_mean / (fused_mean + 1e-8)  # avoid division by zero
        return np.clip(fused * scale, 0, 255).astype(verified_stack.dtype)

class MeanFuser(Fuser):
    """
    Take the mean value of pixels across the image stack.
    """

    def __init__(self) -> None:
        super().__init__(FuseMethod.MEAN)

    def get_fused_image(self, video_stack: Union[str, np.ndarray]) -> np.ndarray:
        verified_stack = self._verify_video_stack(video_stack)
        self._log_activity()
        return verified_stack.mean(axis=0)


class MedianFuser(Fuser):
    """
    Take the median value of pixels across the image stack.
    """

    def __init__(self) -> None:
        super().__init__(FuseMethod.MED)

    def get_fused_image(self, video_stack: Union[str, np.ndarray]) -> np.ndarray:
        verified_stack = self._verify_video_stack(video_stack)
        self._log_activity()
        return np.median(verified_stack, axis=0)


class VarFuser(Fuser):
    """
    Returns the variance across the video stack.
    WARNING:
        `np.var` has a really high space complexity of O = n^2. Therefore it could happen that you run out of RAM memory.
    """

    def __init__(self) -> None:
        super().__init__(FuseMethod.VAR)

    def get_fused_image(self, video_stack: Union[str, np.ndarray]) -> np.ndarray:
        verified_stack = self._verify_video_stack(video_stack)
        self._log_activity()
        var_img = verified_stack.var(axis=0)
        var_img = cv.normalize(
            var_img, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F
        )
        return var_img


class ImageFuserFactory:
    """
    The factory returns a specialised fuse method.
    """

    _registry = {
        FuseMethod.MIN: MinFuser,
        FuseMethod.MAX: MaxFuser,
        FuseMethod.VAR: VarFuser,
        FuseMethod.MED: MedianFuser,
        FuseMethod.MEAN: MeanFuser,
    }

    @classmethod
    def get_fuser(cls, method: FuseMethod) -> Fuser:
        """
        Returns a specified strategy, has to be in the `self._registry`
        """
        if method not in cls._registry:
            raise ValueError(f"Fuse method {method} not supported.")
        return cls._registry[method]()

    def print_registry(self):
        """
        Someone might want to print out what the `_registry` contains.
        """
        pprint.pprint_dict(self._registry, "ImageFuserFactory registry:")
