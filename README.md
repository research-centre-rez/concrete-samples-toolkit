# High-Resolution Registration of Irradiated Concrete Scans

This repository contains the codebase and supporting materials for the master's thesis **"High-resolution Registration of Irradiated Concrete Scans"** by Erik Philippe Leclercq. It contains a complete image processing pipeline for the registration of high-resolution video scans of rotating concrete samples that were extracted from nuclear power plants. 

There is a jupyter notebook available at `src/demo.ipynb` that goes through the video processing pipeline step by step. In order to be able to use it, you will need to create a Python venv, download the required packages, and download the dataset (or parts of it).

## Purpose

The designed pipeline enables: 

- Remote analysis of radioactive concrete samples.
- Automatic video registrationg using state-of-the-art algorithms. 
- Generation of composite images for visual inspection and automated crack detection.

The pipeline is fully configurable and modular. Modules in `src/` contain configuration files and a validation schema that ensure that the processed config file is valid. The config files are well-documented with explanations of what each parameter controls.

## Input data

- 4k video scans of cylindrical concrete samples (~390 degree rotation) that contain two illumination conditions.
- Two illumination conditions: **low-angle** (≤15°) and **front-lighting** (≥82°).

> [!NOTE]
> The pipeline is designed such that it assumes that there is only one angle of light present in the scan. Thus you should use `src/split_videos.py` for splitting the original videos into two separate scans.

## Pipeline steps

1. **Video splitting:** into light-specific segments.
1. **Rotation correction via optical flow:** and other video processing steps.
1. **Video registration** with the specified method from one of the following choices: `['mudic', 'orb', 'lightglue']`.
1. **Composite image creation** with the use of `['max', 'min', 'mean']` image fusion.
1. **Evaluation of composite images** with the normalised grey-level variance and the Brenner's evaluation metrics.

Structure of the pipeline is described in `src/README.md`

## Requirements

- Python 3.11
- GPU with `cuda` support
- (optional) 50GB of disc space for the dataset
- FFmpeg, which can be downloaded [here.](https://ffmpeg.org/download.html). You will also need to download the `libx264-dev` library from [here.](https://packages.debian.org/sid/libx264-dev)  
- git lfs installed, instruction on how to install git lfs [can be found here.](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage)  
- jupyter, can be install from [here.](https://jupyter.org/) 

## Installation

```bash
git clone https://github.com/research-centre-rez/leclercq-thesis.git
cd leclercq-thesis

pip install .

# Create the jupyter notebook demo
jupytext src/demo.py --to ipynb
```

The dataset can be downloaded with the following bash script:

```bash
git clone https://huggingface.co/datasets/research-centre-rez/concrete-samples
cd concrete-samples
git lfs pull
```

> [!WARNING]
> The whole dataset spans around 50GB of disk space. You can download a single video from the repository's page [here.](https://huggingface.co/datasets/research-centre-rez/concrete-samples/blob/main/3A.MP4) If you choose to do so, you will need to update the video path in `src/demo.ipynb` accordingly.

## Output

For each individual video, the pipeline outputs the following:

- The registered stack as a `.npy` file, which essentially contains the whole video decompressed into its individual frames.
- One or more of the specified composite images.
- Logs of all the applied transformations.
- Evaluation of the composite images with the specified metrics.
- In case of registration failure a `src/failures.log` is created that logs what exactly went wrong.

## Reference

If you use this work, please cite:

```bibtex
@mastersthesis{leclercq_thesis_2025,
    author = {Erik Philippe Leclercq},
    title = {High-Resolution Registration of Irradiated Concrete Scans},
    school = {Charles University},
    year = {2025},
}
```

Description of the repository for future developers is given in [contributions.md](/contributions.md).
