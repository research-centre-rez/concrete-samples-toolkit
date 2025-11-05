import random
import enum
from typing import Union
from tqdm import tqdm
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import torch.nn.functional as F
import torch


@dataclass
class GaborKernel:
    """
    Dataclass for creating gabor filters
    """

    kernel_shape: Union[tuple[int, int], int]
    sigma: float
    theta: float
    lambd: float
    gamma: float
    psi: float
    kernel_type: int
    gabor_kernel: np.ndarray = None  # This will get created after __init__

    def __post_init__(self):
        if isinstance(self.kernel_shape, int) or isinstance(self.kernel_shape, float):
            self.kernel_shape = (int(self.kernel_shape), int(self.kernel_shape))

        self.gabor_kernel = cv.getGaborKernel(
            ksize=self.kernel_shape,
            sigma=self.sigma,
            theta=self.theta,
            lambd=self.lambd,
            gamma=self.gamma,
            psi=self.psi,
            ktype=self.kernel_type,
        )

    def __str__(self) -> str:
        return (
            f"GaborKernel("
            f"shape={self.kernel_shape}, sigma={self.sigma}, theta={self.theta:.2f}, "
            f"lambda={self.lambd}, gamma={self.gamma}, psi={self.psi})"
        )

    def visualise_kernel(self) -> None:
        plt.imshow(self.gabor_kernel, cmap="gray")
        plt.title(
            f"Gabor filter\nθ={self.theta:.2f}, λ={self.lambd}, σ={self.sigma}, γ={self.gamma}"
        )
        plt.axis("off")
        plt.show()


class GaborKernelBank:
    """
    Class used for applying a bank of Gabor filters onto an image and extracting its features. There are three main ways of processing an image with this gabor bank:
        `apply_bank_on_image`: Processes a single image with the bank of filters. This is done solely on the CPU and works only on a single image.
        `apply_bank_on_image_gpu`: Processes a single image with the bank of filters with the use of GPU. The kernels are stacked into a single matrix (and also padded if needed), and then applied on the image in parallel with the use of a GPU. If this is called and no GPU is available, PyTorch will intead do this operation on the CPU.
        `apply_bank_on_images_gpu`: Processes a batch of images with the bank of filters the GPU. The kernels are stacked into a single matrix (and padded if needed), and then applied on the image in parallel with the use of a GPU. If this is called and no GPU is available, PyTorch will intead do this operation on the CPU.
    """

    def __init__(self) -> None:
        self.kernels = []  # a list of GaborKernels
        self.padded_gpu_kernels = None  # gabor kernel matrices that are on the GPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def add_gabor_kernel(self, k_params: GaborKernel) -> None:
        self.kernels.append(k_params)

    def add_random_gabor_kernel(self, add_n_kernels) -> None:
        for _ in range(add_n_kernels):
            ksize = random.choice(range(5, 72, 2))
            kernel_shape = (ksize, ksize)

            sigma = random.uniform(2.0, 0.3 * ksize)
            theta = random.uniform(0, np.pi)
            lambd = random.uniform(4.0, float(ksize))
            gamma = random.uniform(0.3, 1.0)
            psi = random.uniform(0, 2 * np.pi)

            params = GaborKernel(
                kernel_shape=kernel_shape,
                sigma=sigma,
                theta=theta,
                lambd=lambd,
                gamma=gamma,
                psi=psi,
                kernel_type=cv.CV_64F,
            )

            self.kernels.append(params)
            # self.gpu_kernels.append(torch.from_numpy(params.gabor_kernel).to("cuda"))

    def pad_gpu_kernels(self) -> None:
        '''
        The gabor kernels need to be padded in order for us to be able to do the convolution in batches. Thus they are padded with 0s and added to `self.padded_gpu_kernels`.
        '''
        self.padded_gpu_kernels = []
        max_length = max(k.gabor_kernel.shape[0] for k in self.kernels)
        for kernel in self.kernels:
            padding_required = (max_length - kernel.gabor_kernel.shape[0]) // 2
            if padding_required > 0:
                padded = np.pad(
                    kernel.gabor_kernel,
                    (padding_required, padding_required),
                    mode="constant",
                )
                self.padded_gpu_kernels.append(torch.from_numpy(padded).to(self.device))
            else:
                self.padded_gpu_kernels.append(
                    torch.from_numpy(kernel.gabor_kernel).to(self.device)
                )

    def apply_bank_on_image(
        self, src_image: np.ndarray, combine: bool = False
    ) -> np.ndarray | list[np.ndarray]:
        if len(self.kernels) == 0:
            raise ValueError(
                "No Gabor kernels in the bank. Add kernels before applying."
            )

        if len(src_image.shape) == 3:
            gray = cv.cvtColor(src_image, cv.COLOR_BGR2GRAY)
        else:
            gray = src_image

        filtered_images = []

        for kernel in self.kernels:
            filtered = cv.filter2D(gray, cv.CV_32F, kernel.gabor_kernel)

            filtered_images.append(filtered.astype(np.uint8))

        if combine:
            combined = np.maximum.reduce(filtered_images)
            return combined

        return filtered_images

    def apply_bank_on_image_gpu(
        self, src_image: Union[torch.Tensor, np.ndarray], combine: bool = True
    ):
        assert (
            len(self.kernels) != 0
        ), "Kernel bank should contain at least one kernel. Add kernels before applying."

        if isinstance(src_image, np.ndarray):
            if len(src_image.shape) == 3:
                tensor_image = cv.cvtColor(src_image, cv.COLOR_BGR2GRAY)
            else:
                tensor_image = src_image
            tensor_image = torch.from_numpy(tensor_image).float().to(self.device)
        else:
            tensor_image = src_image.float().to(self.device)

        # Need to create a fake batch for PyTorch
        if tensor_image.ndim == 2:
            tensor_image = tensor_image.unsqueeze(0).unsqueeze(0)

        # Need to pad the kernels in order to be able to use torch.stack()
        if self.padded_gpu_kernels is None:
            self.pad_gpu_kernels()

        with torch.no_grad():
            weights = torch.stack(
                [k[None, ...] for k in self.padded_gpu_kernels], dim=0
            )
            filtered_image = (
                F.conv2d(tensor_image, weights, padding="same").max(dim=1).values
            )
            return filtered_image.squeeze(0).detach().cpu().numpy()

    #            if combine:
    #                return filtered_image.max(dim=1).values
    #
    #            individual_kernel_images = filtered_image.squeeze(0).detach().cpu().numpy()
    #
    #            if combine:
    #                combined = individual_kernel_images.max(axis=0)
    #                return combined
    #
    #            return [img.astype(np.uint8) for img in individual_kernel_images]

    def apply_bank_on_images_gpu(
        self, src_images: list[torch.Tensor], combine: bool = False
    ):
        if len(self.kernels) == 0:
            raise ValueError(
                "No Gabor kernels in the bank. Add kernels before applying."
            )

        with torch.no_grad():
            imgs = []
            for img in src_images:
                imgs.append(img.float())

            gray = torch.stack(imgs, dim=0).unsqueeze(1).to(self.device)

            if self.padded_gpu_kernels is None:
                self.pad_gpu_kernels()

            weights = torch.stack(
                [k[None, ...] for k in self.padded_gpu_kernels], dim=0
            )

            filtered = F.conv2d(gray, weights, padding="same")

            #            print(f"Dataset occupies {gray.untyped_storage().nbytes() / 1_000_000} MB")
            #            print(f"Weights occupy {weights.untyped_storage().nbytes() / 1_000_000} MB")
            #            print(
            #                f"After pass through the convolution: {filtered.untyped_storage().nbytes() / 1_000_000} MB"
            #            )

            filtered_np = filtered.detach().cpu().numpy()

            if combine:
                combined = filtered_np.max(axis=1)
                return [img.astype(np.uint8) for img in combined]

            results = []
            for img_idx in range(filtered_np.shape[0]):
                per_kernels = [
                    filtered_np[img_idx, k_idx].astype(np.uint8)
                    for k_idx in range(filtered_np.shape[1])
                ]
                results.append(per_kernels)

            return results

    def get_gabor_kernel(self, index) -> GaborKernel:
        """
        Returns a gabor kernel from the specific index.
        """
        return self.kernels[index]

    def write_out_kernels(self) -> None:
        """
        Writes out each of the gabor kernels into the terminal.
        """
        for index, kernel in enumerate(self.kernels):
            print(f"Kernel index: {index}")
            print(f"\t {kernel}")

    def show_kernels_in_gallery(
        self,
        cols: int,
    ) -> None:
        """
        Display all Gabor kernels in the bank as a gallery of images.

        Args:
            cols (int): Number of columns in the grid display.
        """
        n_kernels = len(self.kernels)
        if n_kernels == 0:
            print("No kernels to display.")
            return

        rows = (n_kernels + cols - 1) // cols  # ceil division for grid layout

        _, axes = plt.subplots(rows, cols)
        axes = axes.flatten() if n_kernels > 1 else [axes]

        for ax, kernel in zip(axes, self.kernels):
            ax.imshow(kernel.gabor_kernel, cmap="gray")
            ax.set_title(
                f"θ={kernel.theta:.2f} λ={kernel.lambd:.1f}, σ={kernel.sigma:.2f}, γ={kernel.gamma:.2f}, ψ={kernel.psi:.2f}",
                fontsize=8,
            )
            ax.axis("on")

        # Hide unused subplots if any
        for ax in axes[n_kernels:]:
            ax.axis("off")

        plt.suptitle("Gabor Kernel Bank", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        plt.show()
