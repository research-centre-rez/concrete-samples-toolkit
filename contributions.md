# Contributing to the project

Welcome! This guide explains the internal structure and technical details of this codebase for high-resolution video registration of concrete samples.

## Project Overview

See the [README.md](/README.md) for the basic introduction.

The project has two main directories:

1. `src`: This is where the whole software resides, together with its modules.
2. `tests`: Contains `PyTest` tests that are used for ensuring that adding new feature don't (completely) break the existing functionality.

Futhermore, there are the following files:

1. [`pyproject.toml`](/pyproject.toml): Used for installing the cli program with `Poetry`
2. [`README.md`](/README.md): Simple introduction to the project and how to get it running.
3. [`future_work.md`](/future_work.md): A short list of ideas that could be used for future work on the project.

# Tests

The tests that have been written cover the program's functionality so far. They don't use real data, instead we created mockups that have the same "signature" as our data.

`PyTest` is used for running these tests. If you want to only run a certain test, you can do so with the following `pytest tests/test_filename.py`. 

These tests definitely **do not** cover all possible edge-cases, nor do they cover "invalid" filetypes. As this tool is being mainly developed for only internal uses, the program expects that the data that is passed into it has the valid type and that it contains what we would expect it to contain (i.e. a concrete sample that has been radioactively exposed).

# Concrete samples toolkit

The software consists of several modules that can be either be used in a pipeline manner, or independently. The modules that have been developed so far are:

1. [`crack identification`](/src/concrete_samples_toolkit/crack_identification): Extracting binary masks that show where the cracks are located on samples' surface.
1. [`data preprocessing`](/src/concrete_samples_toolkit/data_preprocessing): Removing dark parts of the raw scans and (if needed) splitting videos into separate parts.
1. [`image evaluation`](/src/concrete_samples_toolkit/image_evaluation/): Evaluates the quality of the video registration.
1. [`image fusion`](/src/concrete_samples_toolkit/image_fusion/): Performs image fusion on the registered stacks.
1. [`utils`](/src/concrete_samples_toolkit/utils): Various utility functions that are used across the project.
1. [`video processing`](/src/concrete_samples_toolkit/video_processing): Applying rotation correction on the videos to ensure better video registration.
1. [`video registration`](/src/concrete_samples_toolkit/video_registration): Video registration of the videos.

> [!NOTE]
> As of right now, there is no way to run all of the steps of the pipeline with just one command. Each step of the pipelines has to be run separately, with its own parameters specified. While this is a bit impractical, it allows for checking that each step has been performed correctly. It also allows for using different options between the steps.

There is no GUI for this program at the moment, most of the parameters are configured with the use of config files. However, there are still some parameters that have to be specified at runtime. These are mainly concerning what the input is, where to save it, and selecting a method (for the cases that the module has multiple options).

Since we are always dealing with a batch of new scans, I tend to run the program with a regex that will match all of the required input files. But the program also works with a single input file. The only exception to this is the [`crack identification`](/src/concrete_samples_toolkit/crack_identification) module, that takes in a single csv file that contains pairs of images.

The default configuration files are located in their respective modules directories.

## Data pre-processing module

The raw scans that we get contain periods of darkness, these are due to the operators turning the LED diodes on/off. Originally, the scans contained both angles of illumination and there was a need to separate these out. Right now, the angles of illumination are separated into two videos during scanning time. The module no longer splits one video into two videos, however it still removes the periods of darkness for easier processing.

The input of this module is a "raw" scan, this is the video that we get from the operators. The outputs is a video where the brief periods of darkness have been removed. The output filetype is an `mp4`. The new videos are saved into a new directory that is specified at runtime. `ffmpeg` is used for creating the new videos.

> [!NOTE]
> Since the new scanning separates the the angle of illumination at scan-time, the `part_` part of the filename can be removed.

This module has no configuration files as there is nothing to configure.

## Video processing module

The registration task is too difficult without a reliable ground truth, as such we utilise rotation correction as the first step of the registration. Besides rotation correction, we also perform video sub-sampling and down-scaling to save some space on the disk (however if one wishes, this is entirely optional). This module offers three methods for performing rotation correction:

1. **Optical flow**: Using Lukas-Kanade optical flow, which is a local optical flow method we perform rotation correction on each frame by the angle that the sample has turned. This is the most reliable method that is available.
1. **Fixed rotation correction**: Each frame is rotated by a fixed amount, this method is not as precise due to the interference of signals between the motor steps and the camera's sampling rate.
1. **None**: No rotation correction, used when one wants to only downscale or sub-sample a video.

> [!NOTE]
> One fully decoded original scan (4k resolution) can take up to 20GB of space on disk / RAM. If you would like to preserve the 4k resolution, sub-sampling might still save you some space.

The input is a pre-processed video from [`data_preprocessing`](/src/concrete_samples_toolkit/data_preprocessing); the output is a video that has had its rotation corrected, and has been subsampled / downscaled.

The module is fully configurable, the default config file is a `JSON5` that contains all of the parameters needed and their descriptions. Before running, the validity of the input config file is verified with `jsonschema`.

The default parameters are the ones that have been used in Leclercq's thesis. 

## Video registration module

The registration module registers videos that are output by the `video_procesing` module and save them as fully decoded `.npy` files. The `.npy` files are of shape $(n,h,w)$, where $n$ is the number of frames in the input video, and $h,w$ are the videos height and width. There are three available methods for registration:

1. **muDIC:** Uses digital image correlation for achieveing registration. Due to the constraints that this method utilises, it is the least precise one for registration.
1. **ORB:** Uses ORB descriptors for finding keypoints in the images and establishing correspondences.
1. **LightGlue:** A transformer-based matcher that utilises SuperPoint for finding keypoints.

> [!NOTE]
> The reason for caching the results into `.npy` files is that even though they take up a lot of space it allows us to be able to perform various experiments on the already registered datapoints. 

We have chosen to use perspective transformation. The reason for this is that in the original dataset that we were given, the samples underwent quite a lot of perspective shifts and we wanted to be able to cope with this.

The module is fully configurable, the default config file is a `JSON5` that contains all of the parameters needed and their descriptions. Before running, the validity of the input config file is verified with `jsonschema`.

muDIC was used purely as a baseline for the thesis, I do not think that it has much use in the current pipeline and it is something that can be safely removed from the module. ORB is much faster than LightGlue, and it shows some potential in the right conditions. Maybe with the new scanning procedure it will be able to perform better. LightGlue is the newest methodology, and it is capable of dealing with with the widest variety of registration thanks to SuperPoint and its reliable keypoint extraction.

The default parameters are the ones that have been used in Leclercq's thesis. Since the new scanning methodology is slightly different, it might be a good idea to revisit these parameters and fine-tune them again.

## Image evaluation module

The role of this module is to evaluate the quality of the registrated stack. This is achieved with the use of two metrics: Brenner's method and Normalised Gray Level Variance (NGLV). As these metrics require a single input image, we perform various image fustion methods for evaluation. We use the sharpness of the fused images as a proxy for evaluating the quality of the registration.

The input of this module are fused images, the output is a series of csv files that show the perceived sharpness of the images for that specific fusion methods. Additionally, you can specify if you wish to use image normalisation or not.

Due to the compression that happens in the fusion, this approach is not as good for evaluating the quality of the registration. It was only used in my thesis as part of the thesis requirements stated that the precision of the registration will be evaluated.

If one would like to quantify the quality of the registered scans, a different approach might be more suitable. In my brief experiemtns it seems like measuring the samples' movement throughout the registered stack is more indicative of the precision of the registration. This movement should not only track the $x,y$ positions also the rool, pitch and yaw of the sample's surface. 

The reason why roll, pitch and yaw should also be tracked is that we are using perspective transformation in the registration module. Perspective transformation was chosen due to the issues that some of the original dataset contained, it might be worth investigating if these are still needed in the current setup of the scanning device.

With all of this combined, this module needs to either be re-made so that it is more accurate or be thrown away completely. 

## Crack identification module

The role of this module is to create a binary crack segmentation mask of the fused images. This is the newest module, therefore it does not have as many features as other modules. 

The input for this module is a registered stack, the output is a series of images that highlight where the cracks are. We create three images for each input stack: binary mask, median image, and the median image with the binary mask overlaid. The purpose of the last image is to show us how accurate the mask is. All the images are saved in the same directory where the registered files reside.

Currently, there are two main methods that are used for crack segmentation. These are Sobel filters and median-normalised difference between two fused images. Neither of these methods has good coverage of the cracks that are present on the sample's surfaces. They are both prone to false positives, the Sobel filter more so than the median-normalised difference. On the other hand, the median-normalised difference tends to only cover really "strong" cracks and not much else, Sobel filters are capable of identifying more of the "weaker" cracks. 

> [!NOTE]
> What we mean by the term "stronger" and "weaker" cracks simply describes how easy the cracks are to spot. Stronger cracks are easy to spot even with human eye, while weaker cracks often appear as faint lines. The thickness of the crack does not necessarily reflect how difficult it is to spot.

The approach we have chosen for identifying cracks is to treat the fused images as a 2D signal, where the cracks should create peaks (higlights due to light refraction) and valleys (shadows) in the signal. As such, using max/min image fusion is not ideal due to them imprinting noise that stems from registration error into the resulting images. Median / mean fused images remove most of the registration, making them ideal candidates for this task. In my small local experiments I have found that further noise suppression by a small Gaussian kernel helped the crack identification. 

When it comes to parameter finetuning of both methodologies, very little has been done. There is a lot of space for optimisation, namely kernel sizes and thresholds for both methods.

As described in [`future_work.md`](/future_work.md), it might be a good idea to do crack identification as a two step process. In that case, the parameters should be tuned such that they include more false positives which are easier to remove in CVAt than manually tracing a crack across the surface's sample. Ultimately, this will be used for training a neural network capable of performing crack segmentation automatically and with higer precision.

## Image fusion module

This module is used for performing image fusion on the registered npy files. The input is a registered `npy` file and the output is a series of images. Specified by the user at runtime. Overall there are several method available, most of which are implemented with he use of `numpy`:

1. **min:** Takes minimum from each from across the whole stack.
1. **max:** Takes maximum from each from across the whole stack.
1. **var:** Variable of each pixel across the stack.
1. **med:** Median of each pixel across the stack.
1. **mean:** Mean of each pixel across the registered stack.
1. **all:** Performs all of the available image fusion.

These methods calculate new pixel values based on the "column" where each pixel resides. For the input stack of shape $(n,h,w)$ we get an image with shape $(h,w)$ and each pixel $x,y$ is calculated based on the values in the 0th axis.

I have experimented with more complicated methods for image fusion, however due to the sheer volume of data we are working with the computations were either taking too long or they took too much space.

## Utils module

The utils module mainly contains functions that are accessed across the whole project.

Lately, a small utility program has been added to this module. Its main role is to identify pairs of before / after exposure for each sample and putting these pairs into a csv. This csv is then later used for quantifying how the samples changed between exposures.

Notably, there is also [visualisers.py](/src/concrete_samples_toolkit/utils/) which contains `imshow`. This is useful for debugging purposes as it will display any series of images in one single graph.
