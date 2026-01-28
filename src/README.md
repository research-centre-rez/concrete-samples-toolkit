# Structure of this directory

All of the code related to the project is placed here. The parts of the video processing pipeline are treated as python modules and each part has a corresponding script that can be used to run that part of the pipeline. The structure is the following:

- `data_preprocessing/`: Video splitting and removal of dark scenes. Can be used with `cst-pre-processing`.
- `video_processing/`: Downsampling of the pre-processed videos. Can be used with `cst-video-processing`.
- `video_registration`: Registration of the processed scans. Can be used with `cst-video-registration`
- `image_fusion`: Fusion of the registered video stacks. Can be used with `cst-image-fusion`.
- `image_evaluation`: Evaluation of the fused images. Can be used with `cst-image-evaluation`
- `crack_identification`: Performing crack segmentation on the registered stacks. Can be used with `cst-image-evaluation`

The core functionality of the code base is documented. The jupyter notebook `demo.ipynb` goes through the whole video processing pipeline step by step. If you wish to use the modules, please read their documentation by passing in the `--h` flag for each module.
