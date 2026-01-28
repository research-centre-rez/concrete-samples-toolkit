# Future work

This document will outline potential future steps that I think are relevant to this project. The scale varies, some of the things might already be referenced in the project's Jira board, [which can be found here](https://cvrez.atlassian.net/jira/core/projects/BET/board?filter=&groupBy=status).

In case of any uncertainties feel free to contact me by [e-mail](rkleclercq@gmail.com), or message me on Microsoft Teams (Erik Leclercq). 

The order of importance / necessity is completely arbitraty. These are just ideas and my own opninions, due to lack of time some of these have not been properly researched.

## Create a database for the scanned samples

Currently, each performed experiment has all of its data dumped into a single directory. The structure of these directories tends to look something like the following: 

```bash
directory_root/
    before_exp/
        sample_x/
            front_face/
                video_filename/
                    images/
                    video_filename_registered.npy
                    video_filename.mp4
            rear_face/
                ...
        sample_y/
            ...
    after_exp/
        sample_x/
            ...
```

This structure allows us to process *all* scans in a given directory simply with the use of regex. However, there is no way to quickly extract information that pertains to a specific sample. For the example given above, if one wanted to know whether `sample_y` has been scanned, one would have to manually search the directory for `sample_y` and ensure that it has been scanned, registered, and that the cracks have been identified. 

An example entry in the database could look something like this:

```json
sample_name: {
    front_face: {
        before_exposure_npy_path: str,
        after_exposure_npy_path: str,
        metrics: {
            before_exposure_area: int,
            after_exposure_area: int
        },
        masks: {
            before_mask_path: str,
            after_mask_path: str
        },
        images: ["mean", "max", "..."]
    },
    rear_face: "similarly as above"
}
```

Once we are doing more exposures for long term tracking, the entry could be changed such that for each exposure the relevant information is stored. This would allow us to easily "browse" through the experiment data without having to manually check everything.

## Performing crack analysis semi-automatically

Due to the nature of the data that we are working with, there are no readily available datasets that can be used for training a neural network for crack segmentation. As such, I think it would be a good idea to perform crack analysis semi-automatically with the use of some baseline extractor + manual refinement of the labelling. The prior extractor has potential to save time for the operators while also it could be more precise than manual labelling.

This new refined labelling could then be collected into a single dataset, that could then be utilised for training a neural network. Besides giving us the opportunity to do training, we will be able to evaluate existing approaches and their perfomance against this dataset. 

This is something that has been partially started, however it hasn't been fully realised yet.

## Dockerise the project

For easier deployment and installation.

## Put the project on the PyPI

Right now everything is more or less ready, it just does not have high enough priority so I didn't get around to doing it.
