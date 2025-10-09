from enum import Enum, auto
import sys
from typing import Union
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


class normType(Enum):
    grad_mag = auto()
    l1_norm = auto()


class Metric:

    def __init__(self) -> None:
        pass

    def calculate_metric(
        self, img_path: str, normalise: bool, normalisationType: normType
    ) -> np.ndarray:
        raise ValueError("Each subclass should implement this on its own")

    def _normalise_img(self, image: np.ndarray, normalisationType: normType):
        """
        Normalised the image to have values in range (0,1).
        Arguments:
            image (np.ndarray): image to be normalised
            normType (normType): Type of the normalisation to be applied. Can be one of the following:
                `l1_norm`: image / np.sum(image)
                `grad_mag`: gradient normalisation with the use of Sobel filters
        Returns:
            Normalised image
        """
        image = image.astype(np.float64)

        if normalisationType == normType.l1_norm:
            return image / np.sum(image) * image.size

        if normalisationType == normType.grad_mag:
            gx = cv.Sobel(image, cv.CV_64F, 1, 0, ksize=3)
            gy = cv.Sobel(image, cv.CV_64F, 0, 1, ksize=3)
            grad_mag = np.sqrt(gx**2 + gy**2)
            return grad_mag

    def _mask_sample(self, img: np.ndarray) -> np.ndarray:
        """
        Create a circular mask for the samples and mask the input image. Cuts out the background while keeping the sample in the image.
        Args:
            img (np.ndarray): Image to be masked
        Returns:
            Masked image
        """
        h, w = img.shape[:2]
        mask = np.zeros((h, w), np.uint8)
        radius = h // 2
        centre = (w // 2, h // 2)
        cv.circle(mask, centre, radius, 255, thickness=-1)
        masked_img = cv.bitwise_and(img, img, mask=mask)
        return masked_img

    def _load_img_to_memory(self, img_path: str) -> np.ndarray:
        """
        Utility funtion that load the image into memory.
        Args:
            img_path (str): Relative path to the image file
        """
        image = cv.imread(img_path, cv.IMREAD_GRAYSCALE)

        return self._mask_sample(image)

    def _preprocess_image(
        self,
        img_path: Union[str, np.ndarray],
        normalise: bool,
        normalisationType: Union[None, normType],
    ) -> np.ndarray:
        """
        Wrapper for pre-processing the fused image. Normalises the image if required and the user can specify the normalisation they wish to apply.
        Arguments:
            img_path (Union[str, np.ndarray]): either path to the image or the loaded image as an object.
            normalise (bool): whether the image should be normalised or not.
            normType (normType): What kind of normalisation you wish to apply.

        Returns:
            Preprocessed image
        """
        if isinstance(img_path, str):

            image = self._load_img_to_memory(img_path)
            image = self._mask_sample(image)


            if image.dtype == np.uint8:
                image = image.astype(np.float64) / 255.0
            if normalise and normalisationType is not None:
                image = self._normalise_img(image, normalisationType)

            return image

        if isinstance(img_path, np.ndarray):
            image = self._mask_sample(img_path)
            if image.dtype == np.uint8:
                image = image.astype(np.float64) / 255.0
            if normalise and normalisationType is not None:
                image = self._normalise_img(image, normalisationType)

            return image


class NGLV(Metric):
    """
    Calculates the normalised grey-level variance of a fused image.
    """

    def calculate_metric(
        self,
        img_path: Union[str, np.ndarray],
        normalise: bool,
        normalisationType: normType,
    ) -> np.ndarray:


        image = self._preprocess_image(img_path, normalise, normalisationType)

        mean, std_dev = cv.meanStdDev(image)
        result = std_dev[0] ** 2 / mean[0]
        return result[0]


class BrennerMethod(Metric):
    """
    Calculates the Brenner score for a fused image. The score is normalised by the size of the input image.
    """

    def calculate_metric(
        self,
        img_path: Union[str, np.ndarray],
        normalise: bool,
        normalisationType: normType,
    ) -> np.ndarray:
        image = self._preprocess_image(img_path, normalise, normalisationType)

        diff_x = np.abs(np.subtract(image[:-2, 2:], image[:-2, :-2], dtype=np.float64))
        diff_y = np.abs(np.subtract(image[2:, :-2], image[:-2, :-2], dtype=np.float64))

        raw_brenner = np.sum(np.maximum(diff_x, diff_y) ** 2)
        return raw_brenner / image.size


class AbsoluteGradient(Metric):

    def calculate_metric(
        self, img_path: Union[str, np.ndarray], normalise: bool, normalisationType
    ) -> np.ndarray:
        image = self._preprocess_image(img_path, normalise, normalisationType)

        I_x = np.abs(np.diff(image, 1, 1, 0))
        I_y = np.abs(np.diff(image, 1, 0, 0))
        grad = np.maximum(I_x, I_y)
        return np.sum(grad) / grad.size


class MutualInformation(Metric):
    """
    Calculates the mean Mutual information for a given registered stack.
    """

    def _load_video_stack(
        self, stack_path: Union[str, np.ndarray], normalise: bool
    ) -> np.ndarray:
        if isinstance(stack_path, str):
            stack = np.load(stack_path)
            if normalise:
                return stack.astype(np.float64) / 255.0

            return stack

        if isinstance(stack_path, np.ndarray):
            if normalise:
                return stack_path.astype(np.float64) / 255.0
            return stack_path

        raise ValueError(f"Wrong type of stack, got {type(stack_path)}")

    def _mutual_information(
        self, img1: np.ndarray, img2: np.ndarray, bins: int = 256
    ) -> float:
        hgram, _, _ = np.histogram2d(img1.ravel(), img2.ravel(), bins=bins)
        pxy = hgram / np.sum(hgram)
        px = np.sum(pxy, axis=1)
        py = np.sum(pxy, axis=0)

        px_py = px[:, None] * py[None, :]
        # Entropies
        Hx = -np.sum(px[px > 0] * np.log(px[px > 0]))
        Hy = -np.sum(py[py > 0] * np.log(py[py > 0]))

        nz = pxy > 0
        mi = np.sum(pxy[nz] * np.log(pxy[nz] / px_py[nz]))

        nmi = mi / np.sqrt(Hx * Hy)  # Normalized MI
        return nmi

    def calculate_metric_with_std(
        self, img_path: Union[str, np.ndarray], normalise: bool
    ):
        stack = self._load_video_stack(img_path, normalise)

        mean_image = np.max(stack, axis=0)

        #mean_image = self._normalise_img(mean_image, normType.l1_norm)

        mi_scores = [
            self._mutual_information(
                img, mean_image
            )
            for img in stack
        ]
        mean_score = np.percentile(mi_scores, 1)
        std_score = np.std(mi_scores)
        return mean_score, std_score

    def calculate_metric(
        self, img_path: Union[str, np.ndarray], normalise: bool, normalisationType=None
    ) -> np.ndarray:
        """
        Returns the mean MI score for the input stack
        """
        stack = self._load_video_stack(img_path, normalise)
        mean_img = np.mean(stack, axis=0)

        mi_scores = [self._mutual_information(img, mean_img) for img in stack]

        return np.mean(mi_scores)
