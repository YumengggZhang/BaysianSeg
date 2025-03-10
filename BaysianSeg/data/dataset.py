import os
from glob import glob
from monai.data import CacheDataset, PatchDataset
from data.transform import (
    volume_transform,
    slice_transform_train,
    slice_transform_valid,
    transform_train,
    transform_valid,
    FilterSliced,
)
from torch.utils.data import Dataset
from monai.transforms import Compose


def build_Prostate(image_set, args):
    assert os.path.exists(
        args.dataset_dir
    ), f"provided data path {args.dataset_dir} does not exist"

    # file_paths = glob(os.path.join(args.dataset_dir, "RUNMC", image_set, "*.nii.gz"))
    file_paths = glob(os.path.join(args.dataset_dir, image_set, "*.nii.gz"))

    image_paths, label_paths = [], []
    for path in file_paths:
        if path.split("\\")[-1][9:12] in ["seg", "Seg"]:
            label_paths.append(path)
        else:
            image_paths.append(path)

    image_paths, label_paths = sorted(image_paths), sorted(label_paths)

    path_dicts = [
        {"image": image_path, "label": label_path}
        for image_path, label_path in zip(image_paths, label_paths)
    ]

    # split train and val set
    if image_set == "train":
        slice_transform = slice_transform_train
    elif image_set == "val":
        slice_transform = slice_transform_valid

    dataset = CacheDataset(
        data=path_dicts, transform=volume_transform, cache_rate=1.0, num_workers=1
    )

    slice_sampler = FilterSliced(
        ["image", "label"], source_key="label", samples_per_image=12
    )

    slice_dataset = PatchDataset(dataset, slice_sampler, 12, slice_transform)
    
    return slice_dataset


def build_ImageCAS(image_set, args):
    assert os.path.exists(
        args.slice_dataset_dir
    ), f"provided data path {args.slice_dataset_dir} does not exist"

    # file_paths = glob(os.path.join(args.dataset_dir, "RUNMC", image_set, "*.nii.gz"))
    file_paths = glob(os.path.join(args.slice_dataset_dir, image_set, "*.nii.gz"))

    image_paths, label_paths = [], []
    for path in file_paths:
        if path.split("\\")[-1][-10:-7] == 'lbl':
            label_paths.append(path)
        else:
            image_paths.append(path)

    image_paths, label_paths = sorted(image_paths), sorted(label_paths)

    path_dicts = [
        {"image": image_path, "label": label_path}
        for image_path, label_path in zip(image_paths, label_paths)
    ]

    # split train and val set
    if image_set == "train":
        slice_transform = slice_transform_train
    elif image_set == "val":
        slice_transform = slice_transfor

    # path_dicts = path_dicts[:10]

    dataset = Slices2DDataset(
        data=path_dicts, transform=slice_transform)

    return dataset

def build_ImageCAS_3d(image_set, args):
    assert os.path.exists(args.dataset_dir), f"provided data path {args.dataset_dir} does not exist"

    # file_paths = glob(os.path.join(args.dataset_dir, "RUNMC", image_set, "*.nii.gz"))
    file_paths = glob(os.path.join(args.dataset_dir, image_set, "*.nii.gz"))

    image_paths, label_paths = [], []
    for path in file_paths:
        if 'Segmentation' in path:
            label_paths.append(path)
        else:
            image_paths.append(path)

    image_paths, label_paths = sorted(image_paths), sorted(label_paths)

    path_dicts = [
        {"image": image_path, "label": label_path}
        for image_path, label_path in zip(image_paths, label_paths)
    ]

    # split train and val set
    if image_set == "train":
        slice_transform = transform_train
    elif image_set == "val":
        slice_transform = transform_valid

    # path_dicts = path_dicts[:10]

    dataset = CasDataset(
        data=path_dicts, transform=slice_transform)

    return dataset


class Slices2DDataset(Dataset):
    def __init__(self, data, transform=None):
        """
        data_dicts: list of {"image": <path>, "label": <path>}
        slice_transform: optional transform pipeline (e.g. LoadImaged, etc.)
        """
        self.data_dicts = data
        self.slice_transform = transform

    def __len__(self):
        return len(self.data_dicts)

    def __getitem__(self, idx):
        d = self.data_dicts[idx]
        # If you want to load the data and transform it:
        if self.slice_transform is not None:
            d = self.slice_transform(d)
        return d

class CasDataset(Dataset):
    def __init__(self, data, transform=None):
        """
        data_dicts: list of {"image": <path>, "label": <path>}
        slice_transform: optional transform pipeline (e.g. LoadImaged, etc.)
        """
        self.data_dicts = data
        self.transform = transform

    def __len__(self):
        return len(self.data_dicts)

    def __getitem__(self, idx):
        d = self.data_dicts[idx]
        # If you want to load the data and transform it:
        if self.transform is not None:
            d = self.transform(d)
        return d