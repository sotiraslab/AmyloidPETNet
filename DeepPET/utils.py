import numpy as np
from skimage.segmentation import slic
import nibabel as nib
import torch
import os
import shutil 

torch_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def float2uint8(img_np):

    img_np = (img_np - np.min(img_np)) / (np.max(img_np) - np.min(img_np)) * 255

    return img_np.astype(np.uint8)


def load_nifti(nii_path, multichannel=False):

    img_nii = nib.load(nii_path)
    img_np = img_nii.get_fdata()

    if len(img_np.shape) == 4:
        if multichannel==False:
            img_np = img_np[:, :, :, 0]

    return img_np


def mint_nifti(img_np, affine_nib, header_nib):

    return nib.Nifti1Image(
        dataobj=img_np,
        affine=affine_nib,
        header=header_nib,
    )


def clean_cache(cdir):

    # clear cache
    pt_files = os.listdir(cdir)
    filtered_files = [file for file in pt_files if file.endswith(".pt")]
    print(f"removing {len(filtered_files)} files")
    for file in filtered_files:
        path_to_file = os.path.join(cdir, file)
        os.remove(path_to_file)
    print(f"removed {len(filtered_files)} files")
    print(f"clean-up complete")
