import os
import shutil
from monai.data.dataset import PersistentDataset
from nibabel import orientations
import numpy as np
from numpy.testing._private.utils import IgnoreException
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from DeepPET.stratified_group_split import StratifiedGroupKFold
from skimage import measure, morphology
from skimage.io import imsave
from typing import Any, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union
from monai.config import KeysCollection
from monai.data import Dataset, DataLoader, PersistentDataset
from monai.transforms import (
    LoadImaged,
    Orientationd,
    AsChannelFirstd,
    AsChannelLastd,
    Compose,
    ScaleIntensityRangePercentilesd,
    NormalizeIntensityd,
    RandAffined,
    RandFlipd,
    CropForegroundd,
    ToTensord,
    Spacingd,
    SaveImaged,
    transform,
    RandomizableTransform,
    InvertibleTransform,
)



class SuppressBackgroundd(transform.MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False,
    ) -> None:

        super().__init__(keys, allow_missing_keys)
        self.suppressor = SuppressBackground()

    def __call__(
        self, data: Mapping[Hashable, np.ndarray]
    ) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.suppressor(d[key])
        return d


class SuppressBackground(transform.Transform):
    def __init__(
        self,
    ) -> None:
        pass

    def __call__(self, img_np: np.ndarray) -> np.ndarray:

        if len(img_np.shape) > 3:
            img_np = img_np[0, :, :, :]

        #! replace NaN with 0
        img_np[np.isnan(img_np)] = 0

        img_np = img_np.clip(min=0)
        img_hist, hist_bins = np.histogram(img_np, bins=256)

        foreground_mask = img_np > hist_bins[1]
        foreground_mask = morphology.binary_closing(foreground_mask)
        foreground_mask = self.get_largest_component(foreground_mask)

        img_np = img_np * foreground_mask
        img_np = np.expand_dims(img_np, axis=0)

        return img_np
    
    def get_foreground_mask(self, img_np):

        img_np = img_np.clip(min=0)
        img_hist, hist_bins = np.histogram(img_np, bins=256)

        foreground_mask = img_np > hist_bins[1]
        foreground_mask = morphology.binary_closing(foreground_mask)
        foreground_mask = self.get_largest_component(foreground_mask)

        return foreground_mask

    def get_largest_component(self, segmentation):

        labels = measure.label(segmentation)
        largestCC = labels == np.argmax(
            np.bincount(labels.flat, weights=segmentation.flat)
        )
        return largestCC


class ExpandChanneld(transform.MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False,
    ) -> None:

        super().__init__(keys, allow_missing_keys)
        self.expander = ExpandChannel()

    def __call__(
        self, data: Mapping[Hashable, np.ndarray]
    ) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.expander(d[key])
        return d


class ExpandChannel(transform.Transform):
    def __init__(
        self,
    ) -> None:
        pass

    def __call__(self, img_np: np.ndarray) -> np.ndarray:

        if len(img_np.shape) == 3:
            # add a channel dimension
            img_np = np.expand_dims(img_np, axis=3)

        return img_np


class SliceAxiald(
    transform.MapTransform,
    transform.RandomizableTransform,
):
    def __init__(
        self,
        keys: KeysCollection,
        prob: float = 0.5,
        allow_missing_keys: bool = False,
    ) -> None:

        transform.MapTransform.__init__(self, keys, allow_missing_keys)
        transform.RandomizableTransform.__init__(self, prob)
        self.slicer = SliceAxial()

    def randomize(self):
        super().randomize(None)
        self._slice = int(self.R.uniform(low=1, high=10))

    def __call__(
        self, data: Mapping[Hashable, np.ndarray]
    ) -> Dict[Hashable, np.ndarray]:

        self.randomize()

        d = dict(data)
        for key in self.key_iterator(d):
            if self._do_transform:
                d[key] = self.slicer(d[key], slice=self._slice)
        return d


class SliceAxial(transform.Transform):
    def __init__(self) -> None:
        pass

    def __call__(self, img, slice):

        return img[:, :, :, :-slice]


class DebugBefored(transform.MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False,
    ) -> None:

        transform.MapTransform.__init__(self, keys, allow_missing_keys)

    def __call__(
        self, data: Mapping[Hashable, np.ndarray]
    ) -> Dict[Hashable, np.ndarray]:

        d = dict(data)
        for key in self.key_iterator(d):
            print(f"debugger (before random): {d[key]}")
        return d


class DebugAfterd(transform.MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False,
    ) -> None:

        transform.MapTransform.__init__(self, keys, allow_missing_keys)

    def __call__(
        self, data: Mapping[Hashable, np.ndarray]
    ) -> Dict[Hashable, np.ndarray]:

        d = dict(data)
        for key in self.key_iterator(d):
            print(f"debugger (after random): {d[key]}")
        return d

class DebugShaped(transform.MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False,
    ) -> None:

        transform.MapTransform.__init__(self, keys, allow_missing_keys)

    def __call__(
        self, data: Mapping[Hashable, np.ndarray]
    ) -> Dict[Hashable, np.ndarray]:

        d = dict(data)
        print("hello")
        for key in self.key_iterator(d):
            print(f"DebugShape: {d[key].shape}")
        return d




class DeepPETDataGenerator:
    def __init__(self, subjects=None, tracers=None, fpaths=None, labels=None):

        self.random_state = 2024
        self.fpaths = np.array(fpaths)
        self.subjects = subjects
        self.tracers = tracers
        self.labels = labels

        # training transforms
        self.transform_lst = [
            DebugBefored(keys=["debug"]),
            LoadImaged(keys=["img"]),
            # TODO: raise GitHub issue on orientation issue
            # Orientationd(keys=["img"], axcodes="RAS"),
            # expand channel because OASIS3 has 3 channels only
            ExpandChanneld(keys=["img"]),
            AsChannelFirstd(keys=["img"], channel_dim=-1),
            # set background pixels to 0
            SuppressBackgroundd(keys=["img"]),
            # crop out pixels with 0 values
            CropForegroundd(keys=["img"], source_key="img"),
            # resample pixel dimensions to (2mm, 2mm, 2mm)
            Spacingd(
                keys=["img"],
                pixdim=(2, 2, 2),
                diagonal=True,
                mode="bilinear",
                padding_mode="zeros",
            ),
            SliceAxiald(keys=["img"]),
            # random flipping along coronal axis because of brain symmetry
            RandFlipd(keys=["img"], prob=0.5, spatial_axis=0),
            # random rotations in all dimensions
            RandAffined(
                keys=["img"],
                prob=0.5,
                rotate_range=(0.10, 0.10, 0.10),
                as_tensor_output=True,
                padding_mode="zeros",
            ),
            DebugAfterd(keys=["debug"]),
            # clip intensity at 5 and 95 percentile
            ScaleIntensityRangePercentilesd(
                keys=["img"], lower=5, upper=95, b_min=0, b_max=100, clip=True
            ),
            # normalize to reduce the impact of outliers
            NormalizeIntensityd(keys=["img"], nonzero=False),
            ToTensord(keys=["img"]),
        ]

        # prediction transforms
        self.prediction_transform_lst = [
            LoadImaged(keys=["img"]),
            # expand channel because OASIS3 has 3 channels only
            ExpandChanneld(keys=["img"]),
            AsChannelFirstd(keys=["img"], channel_dim=-1),
            # set background pixels to 0
            SuppressBackgroundd(keys=["img"]),
            # crop out pixels with 0 values
            CropForegroundd(keys=["img"], source_key="img"),
            # resample pixel dimensions to (2mm, 2mm, 2mm)
            Spacingd(
                keys=["img"],
                pixdim=(2, 2, 2),
                diagonal=True,
                mode="bilinear",
                padding_mode="zeros",
            ),
            # clip intensity at 5 and 95 percentile
            ScaleIntensityRangePercentilesd(
                keys=["img"], lower=5, upper=95, b_min=0, b_max=100, clip=True
            ),
            # normalize to reduce the impact of outliers d
            NormalizeIntensityd(keys=["img"], nonzero=False),
            ToTensord(keys=["img"]),
        ]

        # LIME image transform
        self.lime_image_transform = Compose([
            LoadImaged(keys=["img"]),
            # expand channel because OASIS3 has 3 channels only
            ExpandChanneld(keys=["img"]),
            AsChannelFirstd(keys=["img"], channel_dim=-1),
            # resample pixel dimensions to (2mm, 2mm, 2mm)
            Spacingd(
                keys=["img"],
                pixdim=(2, 2, 2),
                mode="bilinear",
                padding_mode="zeros",
            ),
            # clip intensity at 5 and 95 percentile
            ScaleIntensityRangePercentilesd(
                keys=["img"], lower=5, upper=95, b_min=0, b_max=100, clip=True
            ),
            # normalize to reduce the impact of outliers d
            NormalizeIntensityd(keys=["img"], nonzero=False),
            ToTensord(keys=["img"]),
        ])

        # LIME segmentation transforms
        self.lime_segmentation_transform = Compose([
            LoadImaged(keys=["img"]),
            # expand channel because OASIS3 has 3 channels only
            ExpandChanneld(keys=["img"]),
            AsChannelFirstd(keys=["img"], channel_dim=-1),
            # # resample pixel dimensions to (2mm, 2mm, 2mm)
            Spacingd(
                keys=["img"],
                pixdim=(2, 2, 2),
                diagonal=True,
                mode="bilinear",
                padding_mode="zeros",
            ),
            ToTensord(keys=["img"]),
        ])

        self.dict_transforms = Compose(self.transform_lst)
        self.prediction_transforms = Compose(self.prediction_transform_lst)

        pass

    def get_split_idx(self, test_size=0.20):
        '''
        !IMPORTANT:
        StraitifiedGroupKFold can also be used to stratify based on tracers (self.tracers) for secondary analysis 
        '''

        train_test_split = StratifiedGroupKFold(n_splits=int(1/test_size), random_state=self.random_state, shuffle=True)

        # split into train & val and test folds
        train_idx, test_idx = next(train_test_split.split(X=self.fpaths, y=self.labels, groups=self.subjects))
        print(f"\ntest-train split...")
        print(f"{len(self.fpaths)} total images")
        print(f"{len(test_idx)} testing images")
        print(f"{len(train_idx)} training images")

        self.train_idx = train_idx
        self.test_idx = test_idx

        return train_idx, test_idx


    def create_dataset(self, cache_dir, idx=list(), mode="training"):

        # cache directory for PersistentDataset
        self.cache_dir = cache_dir

        if len(idx) == 0:
            idx = np.arange(start=0, stop=len(self.fpaths))

        if mode == "training":
            # create dataset for training 
            data_lst = [{"img": fpath, "debug": fpath, "label": label} for fpath, label in zip(self.fpaths[idx], self.labels[idx])]
            transform_chain = self.dict_transforms
        elif mode == "validation":
            # create dataset for validation
            data_lst = [{"img": fpath, "debug": fpath, "label": label} for fpath, label in zip(self.fpaths[idx], self.labels[idx])]
            transform_chain = self.prediction_transforms
        elif mode == "prediction":
            # create dataset for prediction
            print("dataset mode: prediction")
            data_lst = [{"img": fpath} for fpath in self.fpaths[idx]]
            return PersistentDataset(data=data_lst, transform=Compose(self.prediction_transform_lst), cache_dir=self.cache_dir)
        elif mode == "lime image":
            # create dataset for prediction
            data_lst = [{"img": fpath} for fpath in self.fpaths[idx]]
            transform_chain = self.lime_image_transform
        elif mode == "lime segmentation":
            # create dataset for prediction
            data_lst = [{"img": fpath} for fpath in self.fpaths[idx]]
            transform_chain = self.lime_segmentation_transform

        return PersistentDataset(data=data_lst, transform=transform_chain, cache_dir=self.cache_dir)

    def split_train_test(self, cache_dir, test_size=0.1):

        self.cache_dir = cache_dir

        train_idx, test_idx = self.get_split_idx(test_size=test_size)

        train_ds = self.create_dataset(idx=train_idx, cache_dir=self.cache_dir, mode="training")
        # val_ds = self.create_dataset(idx=val_idx, cache_dir=self.cache_dir, mode="validation")
        test_ds = self.create_dataset(idx=test_idx, cache_dir=self.cache_dir, mode="prediction")

        return train_ds, test_ds

    def save_dataframes(self, df, odir):

        if not os.path.isdir(odir):
            os.mkdir(odir)

        train_df = df.iloc[self.train_idx, :]
        train_df.to_csv(os.path.join(odir, "train.csv"))

        val_df = df.iloc[self.val_idx, :]
        val_df.to_csv(os.path.join(odir, "val.csv"))

        test_df = df.iloc[self.test_idx, :]
        test_df.to_csv(os.path.join(odir, "test.csv"))

    def preprocess_for_visualization(self, fpaths, transform_lst=None):
        """
        preprocess a list of images [fpaths] with the generator's preprocessing pipeline or a custom list of transforms [transform_lst]
        """

        if transform_lst != None:
            preprocess_pl = Compose(transform_lst)
        else:
            transform_lst = self.prediction_transform_lst.copy()
            preprocess_pl = Compose(transform_lst)

        check_ds = Dataset(
            data=[{"img": fpath} for fpath in fpaths], transform=preprocess_pl
        )
        check_loader = DataLoader(check_ds, batch_size=1)

        img_lst = []

        for i, batch in enumerate(check_loader):

            img_lst.append(batch["img"][0, 0, :, :, :].numpy())

        return img_lst

    def preprocess_for_prediction(self, fpaths):
        """
        create a data loader with images pre-processed for prediction
        """

        preprocess_pl = Compose(self.prediction_transform_lst)

        check_ds = Dataset(
            data=[{"img": fpath} for fpath in fpaths], transform=preprocess_pl
        )

        return DataLoader(check_ds, batch_size=1)

    def save_3d(self, img_np, map_np, odir):

        if os.path.exists(odir):
            shutil.rmtree(odir)
        os.makedirs(odir, exist_ok=True)

        for k in np.arange(img_np.shape[0]):
            plt.imshow(img_np[k, :, :], cmap="gray")
            plt.imshow(map_np[k, :, :], cmap="jet", alpha=0.5)
            plt.colorbar()
            plt.clim(0, 255)
            plt.savefig(os.path.join(odir, f"saggital_{k}.png"))
            plt.close()

        for k in np.arange(img_np.shape[1]):
            plt.imshow(img_np[:, k, :], cmap="gray")
            plt.imshow(map_np[:, k, :], cmap="jet", alpha=0.5)
            plt.colorbar()
            plt.clim(0, 255)
            plt.savefig(os.path.join(odir, f"coronal_{k}.png"))
            plt.close()

        for k in np.arange(img_np.shape[2]):
            plt.imshow(img_np[:, :, k], cmap="gray")
            plt.imshow(map_np[:, :, k], cmap="jet", alpha=0.5)
            plt.colorbar()
            plt.clim(0, 255)
            plt.savefig(os.path.join(odir, f"axial_{k}.png"))
            plt.close()

        return None
