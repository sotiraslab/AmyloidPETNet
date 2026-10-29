![image](https://github.com/user-attachments/assets/3d583677-d240-4e33-b935-1ebf015f23d9)

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

## Docker (automated workflow)

This repository includes a Dockerized workflow for running inference and training in a reproducible environment.

- `Dockerfile`: builds the project image from `environment.yml` using `micromamba`
- `docker-compose.yml`: defines the runtime container, cache mount, and output mount
- `.vscode/tasks.json`: one-click VS Code tasks for Windows+WSL users

### 1) Prerequisites

- Docker Engine with Docker Compose v2
- For Windows users: Docker Desktop + WSL2 integration enabled
- For Linux users: Docker and Compose available in your shell

### 2) Build image (automated)

From VS Code:

1. Open Command Palette
2. Run `Tasks: Run Task`
3. Choose `Docker: Build AmyloidPETNet image (WSL)`

This task runs:

```
wsl bash -lc "docker compose build amyloidpetnet"
```

Equivalent command-line build from repository root:

```bash
docker compose build amyloidpetnet
```

### 3) Run prediction (automated)

Use task `Docker: Predict (WSL)`.

It will prompt you for:

- dataset csv path relative to repo (example: `data/predict.csv`)
- model directory relative to repo (example: `model`)
- visualization output subdirectory inside `outputs/` (example: `vis`)

The task writes visualizations to `outputs/<your_vis_dir>` and uses `/tmp` inside the container for temporary cache files.

Equivalent command-line prediction from repository root:

```bash
docker compose run --rm amyloidpetnet predict.py --odir /app/model --dataset /app/data/predict.csv --cdir /tmp --vdir /outputs/vis
```

## NiChart integration notes

This repository includes a NiChart-compatible inference wrapper:

- `scripts/nichart_predict_wrapper.py`

The wrapper addresses two integration needs:

1. NiChart provides a directory of images, while `predict.py` expects a CSV with an `img_path` column.
2. `predict.py` writes the output CSV into the model directory (`--odir`), which may be ephemeral in containerized deployments.

The wrapper behavior is:

1. Reads a flat input directory of `.nii` or `.nii.gz` files.
2. Builds a temporary CSV with `img_path` entries.
3. Runs `predict.py` using the existing CLI.
4. Copies the output CSV from the model directory into a user-specified output directory.

Example container command:

```bash
python /app/scripts/nichart_predict_wrapper.py \
    --input-dir /input/nifti \
    --output-dir /output/predictions \
    --model-dir /app/model \
    --cache-dir /tmp
```

Draft NiChart definition files are provided in:

- `scripts/nichart/amyloidpetnet_tool.yaml`
- `scripts/nichart/amyloidpetnet_pipeline.yaml`

These are intended as starting points and may need small key-name edits to match your NiChart schema version.

### 4) Run training (automated)

Use task `Docker: Train (WSL)`.

It will prompt you for:

- training csv path relative to repo
- validation csv path relative to repo
- output subdirectory inside `outputs/`

Training outputs are saved under `outputs/<your_train_output_dir>`.

Equivalent command-line training from repository root:

```bash
docker compose run --rm amyloidpetnet train.py --train /app/data/train.csv --val /app/data/val.csv --cdir /tmp --odir /outputs/train-run
```

### 5) Notes on paths

- The tasks map the repo root to `/app` in container.
- CSVs should use image paths that are accessible from inside container.
- Easiest approach: keep data under this repo (for example `data/`) and reference `/app/...` paths.

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

## Training AmyloidPETNet on other datasets

If you would like to train AmyloidPETNet from scratch on your own dataset, run the following command

```
python train.py 
    --train $PATH_TO_TRAIN_DATA
    --val $PATH_TO_VAL_DATA
    --cdir $TMP_DIR
```
* `$PATH_TO_TRAIN_DATA`: path to a `.csv` file with the following columns:
    * `img_path`: paths to input images, each image being a 3D frame. 
    * `suvr_positivity`: binary labels of amyloid positivity.
* `$PATH_TO_VAL_DATA`: same as above but for validation data
* `$TMP_DIR`: directory for storing temporary cached files of the preprocessing pipeline. This can be any directory that you have write access to, but please note that during clean-up the script will remove all files with a `.pt` suffix. `$TMP_DIR` defaults to `/tmp`.

## Trouble shooting

To help with trouble shooting, the user can optionally save the processed images as a series of 2D `.png` files that span the axial, coronal, and sagittal views. For an example of this, please refer to `tmp/example`, which contains images of a processed frame from the Centiloid project.

## Licenses and acknowledgements

This project is licensed under the LICENSE file in the root directory of the project.
