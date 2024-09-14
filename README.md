# AmyloidPETNet: Classification of Amyloid Positivity in Brain PET Imaging Using End-to-End Deep Learning

## Overview

This is the repository for running inference using [AmyloidPETNet](https://pubs.rsna.org/doi/10.1148/radiol.231442) with pre-trained weights. AmyloidPETNet is a deep learning model that can make binary amyloid positivity classifications based on minimally processed brain PET scans without companion structural MRI scans. Follow this README to get started. The repoistory onsists of the following: 
- `environment.yaml`: conda environment file for dependency installation
- `model`: subdirectory containing the model weights
- `predict.py`: main inference script 
- `DeepPET`: local python module 

## Installation

To install the dependencies of AmyloidPETNet, run the following command: 
```
conda env create -f environment.yml
```
This environment setup had been tested on a Linux cluster with AMD64 CPUs and NVIDIA GPUs. We welcome PRs for environment setups on other platforms.

## Running our model

AmyloidPETNet expects input images of the [NIfTI](https://nifti.nimh.nih.gov) format (`.nii` or `.nii.gz`). Depending on the amyloid tracer, each amyloid brain scan consists of multiple frames of various durations. AmyloidPETNet is compatible with 5-minute frames acquired after the tracer binding steady state was reached. For more details, please refer to [our Radiology manuscript](https://pubs.rsna.org/doi/10.1148/radiol.231442).

To make predictions with our model, run the following command: 

```
python predict.py 
    --odir $MODEL_DIR  
    --dataset $DATASET_CSV
    --cdir $TMP_DIR
    --vdir $VIS_DIR
```
* `$MODEL_DIR`: directory containing the model weights, i.e. `model.pth`. `$MODEL_DIR` defaults to `./model`. 
* `$DATASET_CSV`: path to a `.csv` file with a column named `img_path` (case-sensitive) that contains the paths to input images, each image being a 3D frame. 
* `$TMP_DIR`: directory for storing temporary cached files of the preprocessing pipeline. This can be any directory that you have write access to, but please note that during clean-up the script will remove all files with a `.pt` suffix. `$TMP_DIR` defaults to `/tmp`.
* `$VIS_DIR`: directory for storing the processed images. For more details, please refer to the [trouble shooting](#trouble-shooting) section. To skip storing the processed images, remove this flag. 

For each frame, the script will output the logit, defined as $
\text{logit}(p) = \ln\left(\frac{p}{1 - p}\right)
$, where $p$ is the probability that the corresponding frame is amyloid positive. Throughout the manuscript, we assumed a probability threshold of 0.5 for amyloid positivity, which corresponds to a logit of 0.0. The outputs will be written to a `.csv` file in `$MODEL_DIR` with logits stored under a column named `y_score`. 

## Trouble shooting

To help with trouble shooting, the user can optionally save the processed images as a series of 2D `.png` files that span the axial, coronal, and sagittal views. For an example of this, please refer to `tmp/example`, which contains images of a processed frame from the Centiloid project.

## Licenses and acknowledgements

This project is licensed under the LICENSE file in the root directory of the project.

The initial code of this repository has been initiated by the [Python Machine Learning Research Project Template](https://github.com/CLAIRE-Labo/python-ml-research-template)
with the LICENSE.ml-template file.

Additional LICENSE files may be present in subdirectories of the project.