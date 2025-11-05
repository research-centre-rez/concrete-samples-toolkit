import csv
import os
import multiprocessing as mp
import sys
import numpy as np
import cv2 as cv
from scipy.optimize import differential_evolution
import pandas
from scipy.stats import tukey_hsd
from tqdm import tqdm
from local_contrast_normalisation import local_contrast_normalisation
from gabor_bank import GaborKernel, GaborKernelBank
import torch.nn.functional as F
import torch
from datetime import datetime


DATA_PAIRS = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def init_worker(shared_data):
    global DATA_PAIRS
    DATA_PAIRS = shared_data


def write_results_to_csv(data: list[float], save_as: str):
    """
    Dumps the best Gabor bank into a csv file.
    Args:
        data (list[float]): The gabor bank parameters to be written into the csv file
        save_as (str): Name of the csv, should contain the '.csv' extension
    """

    # TODO: If this module becomes part of the toolkit, use the filebuilder functions to build the filename.
    date_format = "%d-%m-%d_%H-%M"
    time = datetime.now().strftime(date_format)

    out_filename = f"{save_as}_{time}.csv"
    print(out_filename)

    with open(out_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["kernel size", "sigma", "theta", "lambda", "gamma", "psi"])
        for datum in data:
            if int(datum[0]) % 2 == 0:
                datum[0] = int(datum[0]) + 1
            writer.writerow(datum)


def intersection_over_union(
    target_mask: torch.Tensor, predicted_mask: torch.Tensor
) -> float:
    """
    Computes mean IoU on the GPU, in batches.
    Args:
        target_mask (torch.Tensor): stack of ground truth masks in shape [n, h, w] where `n` is the number of masks, `h, w` are the dimensions of each mask.
        predicted_mask (torch.Tensor): stack of predicted masks in shape [n, h, w] where `n` is the number of masks, `h, w` are the dimensions of each mask.
    Returns:
        mean IoU of the `n` predicted masks.
    """
    assert (
        target_mask.shape == predicted_mask.shape
    ), "Both masks should have the same shape"

    intersections = torch.logical_and(target_mask, predicted_mask).sum(dim=(1, 2))
    unions = torch.logical_or(target_mask, predicted_mask).sum(dim=(1, 2))

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

        h, w = image.shape
        ds_factor = 2

        image = cv.resize(image, (h // ds_factor, w // ds_factor))
        mask = cv.resize(mask, (h // ds_factor, w // ds_factor))

        mask = (mask > 50).astype(np.uint8)
        pairs.append(
            (torch.from_numpy(image).to(DEVICE), torch.from_numpy(mask).to(DEVICE))
        )
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


def evaluate_individual_bank(params, n_kernels):
    """
    This evaluation function evaluates each individual on the whole dataset by evaluating each image individually. Requires less memory space than evaluation on the whole stack.
    Args:
        n_kernels (list): flattened representation of the kernel bank and the parameters of each kernel
    Returns:
        Negative mean of the IoU across the whole dataset
    """
    global DATA_PAIRS, DEVICE
    params = np.array(params).reshape(n_kernels, 6)

    pred_masks = []

    with torch.no_grad():

        gabor_bank = init_gabor_bank(params)
        for image, _ in DATA_PAIRS:
            gabor_filtered = gabor_bank.apply_bank_on_image_gpu(image, combine=True)
            pred_masks.append((gabor_filtered > 30).astype(np.uint8))

    gt_masks = torch.stack([mask for _, mask in DATA_PAIRS], dim=0)
    pred_masks = torch.from_numpy(np.array(pred_masks)).to(DEVICE)

    avg_iou = intersection_over_union(gt_masks, pred_masks)
    return -avg_iou


def evaluate_individual_bank_on_stack(params, n_kernels: list):
    """
    This evaluation function evaluates each specimen on the whole dataset in a single pass. While it requires more memory space (for both the GPU and RAM), it should be faster as the whole convolution is done in one pass.
    Args:
        n_kernels (list): flattened representation of the kernel bank and the parameters of each kernel
    Returns:
        Negative mean of the IoU across the whole dataset
    """

    # TODO: maybe add batching capability to this function, would still give us the benefit of faster computation while doing stuff sequentially. Shouldn't be too difficult.
    try:
        global DATA_PAIRS, DEVICE
        params = np.array(params).reshape(n_kernels, 6)

        pred_masks = []

        images = [im for im, _ in DATA_PAIRS]

        with torch.no_grad():
            bank = init_gabor_bank(params)
            filtered_images = bank.apply_bank_on_images_gpu(images, combine=True)

            pred_masks = [
                torch.from_numpy((im > 30).astype(np.uint8)) for im in filtered_images
            ]
            pred_masks = torch.stack(pred_masks).to(DEVICE)
            gt_masks = torch.stack([mask.to(DEVICE) for _, mask in DATA_PAIRS]).to(
                DEVICE
            )

            return -intersection_over_union(gt_masks, pred_masks)

    except Exception as e:
        print("Worker crashed:", e)
        return 1e9


def optimise_gabor_bank(csv_path, n_kernels, max_iter, pop_size):
    mp.set_start_method("spawn", force=True)
    data_pairs = load_image_mask_pairs(csv_path)

    # Controlling how many CPU cores are used in the computation
    num_processes = 2
    pool = mp.Pool(
        initializer=init_worker, initargs=(data_pairs,), processes=num_processes
    )

    # bounds repeated for each kernel
    single_bounds = [
        (3, 11),  # Kernel size
        (0.0, 3.0),  # Sigma
        (0, 180),  # Theta (in degrees)
        (0, 10.0),  # Lambda
        (0.1, 1.0),  # Gamma
        (0, 0),  # Psi, for now its set to 0 to save some computation.
    ]
    bounds = single_bounds * n_kernels

    result = differential_evolution(
        evaluate_individual_bank_on_stack,
        bounds,
        args=(n_kernels,),
        maxiter=max_iter,
        popsize=pop_size,
        workers=pool.map,
        updating="deferred",
        disp=True,
        polish=False,
        init="sobol",
        recombination=0.9,
        mutation=(0.5, 1.2),
        tol=1e-10,
    )

    pool.close()
    pool.join()

    best_params = np.array(result.x).reshape(n_kernels, 6)
    print("Best Gabor bank parameters:")
    g_kernel = {}
    for i, p in enumerate(best_params):
        kernel_size, sigma, theta, lambd, gamma, psi = p
        kernel_size = int(kernel_size)
        if kernel_size % 2 == 0:
            kernel_size += 1

        g_kernel = {
            "kernel_shape": kernel_size,
            "sigma": sigma,
            "theta": theta,
            "lambd": lambd,
            "gamma": gamma,
            "psi": psi,
            "kernel_type": cv.CV_32F,
        }

        print(f"Kernel {i+1}: {GaborKernel(**g_kernel)}")

    print(f"Best IoU: {-result.fun:.4f}")

    csv_filename = "../../../csv_files/gabor_evolution_result"
    write_results_to_csv(best_params, csv_filename)

    return result


if __name__ == "__main__":
    csv_file = "../../../datasets/anotace/anotace/cracks/image_mask_pairs.csv"
    num_kernels = 16
    max_iterations = 1
    population_size = 1

    optimise_gabor_bank(csv_file, num_kernels, max_iterations, population_size)
