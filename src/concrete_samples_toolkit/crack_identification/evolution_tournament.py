import os
import multiprocessing as mp
import sys
import numpy as np
import cv2 as cv
from scipy.optimize import differential_evolution
import pandas
from tqdm import tqdm
from local_contrast_normalisation import local_contrast_normalisation
from gabor_bank import GaborKernel, GaborKernelBank
import torch.nn.functional as F
import torch


def intersection_over_union(target_mask:torch.Tensor, predicted_mask:torch.Tensor) -> float:
    assert (
        target_mask.shape == predicted_mask.shape
    ), "Both masks should have the same shape"

    intersections = torch.logical_and(target_mask, predicted_mask).sum(dim=(1,2))
    unions = torch.logical_or(target_mask, predicted_mask).sum(dim=(1,2))

    ious = intersections / unions
    return ious.mean().item()


def load_image_mask_pairs(csv_path) -> list[torch.Tensor, torch.Tensor]:
    df = pandas.read_csv(csv_path, header=None)
    CSV_DIR = os.path.dirname(csv_path)
    pairs = []
    for _, row in df.iterrows():
        image_path, mask_path = os.path.join(CSV_DIR, row[0]), os.path.join(
            CSV_DIR, row[1]
        )

        image = cv.imread(image_path, cv.IMREAD_GRAYSCALE).astype(np.uint8)
        mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)
        mask = (mask > 50).astype(np.uint8)
        pairs.append((torch.from_numpy(image).to("cuda"), torch.from_numpy(mask).to("cuda")))
    return pairs


def init_gabor_bank(bank_params) -> GaborKernelBank:
    bank = GaborKernelBank()
    for p in bank_params:
        ksize, sigma, theta, lambd, gamma, psi = p
        ksize = int(round(ksize))
        if ksize % 2 == 0:
            ksize += 1
        bank.add_gabor_kernel(
            GaborKernel(
                kernel_shape=(ksize, ksize),
                sigma=sigma,
                theta=np.deg2rad(theta),
                lambd=lambd,
                gamma=gamma,
                psi=psi,
                kernel_type=cv.CV_32F,
            )
        )
    return bank


def evaluate_individual_bank(params, data_pairs, n_kernels):
    params = np.array(params).reshape(n_kernels, 6)
    total_iou = 0.0

    pred_masks = []

    with torch.no_grad():

        gabor_bank = init_gabor_bank(params)
        for image, _ in data_pairs:
            gabor_filtered = gabor_bank.apply_bank_on_image_gpu(image, combine=True)
            pred_masks.append((gabor_filtered > 125).astype(np.uint8))

    gt_masks = torch.stack([mask for _, mask in data_pairs], dim=0)
    pred_masks = torch.from_numpy(np.array(pred_masks)).to('cuda')


    avg_iou = intersection_over_union(gt_masks, pred_masks)
    return -avg_iou



def optimise_gabor_bank(csv_path, n_kernels=1, max_iter=1, pop_size=1):
    mp.set_start_method("spawn", force=True)
    data_pairs = load_image_mask_pairs(csv_path)

    # bounds repeated for each kernel
    single_bounds = [
        (3, 11),  # Kernel size
        (0.0, 5.0),  # Sigma
        (0, 180),  # Theta (in degrees)
        (0, 20.0),  # Lambda
        (0.2, 1.0),  # Gamma
        (0, 2 * np.pi),  # Psi
    ]
    bounds = single_bounds * n_kernels


    result = differential_evolution(
        evaluate_individual_bank,
        bounds,
        args=(data_pairs, n_kernels),
        maxiter=max_iter,
        popsize=pop_size,
        workers=3,
        updating="deferred",
        disp=True,
        polish=False,
    )

    best_params = np.array(result.x).reshape(n_kernels, 6)
    print("Best Gabor bank parameters:")
    for i, p in enumerate(best_params):
        print(f"Kernel {i+1}: {GaborKernel(*p, kernel_type=cv.CV_32F)}")

    print(f"Best IoU: {-result.fun:.4f}")
    return result


if __name__ == "__main__":
    csv_file = "../../../datasets/anotace/anotace/cracks/image_mask_pairs.csv"
    num_kernels = 16
    max_iterations = 3
    population_size = 10

    optimise_gabor_bank(csv_file, num_kernels, max_iterations, population_size)
