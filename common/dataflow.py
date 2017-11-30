
import os
from functools import partial

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold

import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler

from .torch_common_utils.imgaug import RandomAffine, RandomChoice, RandomFlip, RandomCrop
from .torch_common_utils.dataflow import TransformedDataset, OnGPUDataLoader
from .torch_common_utils.deserialization import restore_object

root_path = os.path.abspath("..")
INPUT_PATH = os.path.abspath(os.path.join(root_path, 'input'))

TRAIN_JSON_PATH = os.path.join(INPUT_PATH, 'train.json')
TEST_JSON_PATH = os.path.join(INPUT_PATH, 'test.json')


def preprocess(df):
    df['inc_angle'] = pd.to_numeric(df['inc_angle'], errors='coerce')
    return df


def get_train_df():
    df = pd.read_json(TRAIN_JSON_PATH)
    df = preprocess(df)
    return df


def get_test_df():
    df = pd.read_json(TEST_JSON_PATH)
    df = preprocess(df)
    return df


def get_image(index, df):
    return np.array([df.loc[index, 'band_1'], df.loc[index, 'band_2']], dtype=np.float).reshape(2, 75, 75)


def get_inc_angle(index, df):
    return df.loc[index, 'inc_angle']


def get_target(index, df):
    return df.loc[index, 'is_iceberg']


class IcebergDataset(Dataset):

    def __init__(self, data_type='Train'):
        assert data_type in ['Train', 'Test']
        super(Dataset, self).__init__()

        self.df = get_train_df() if data_type == 'Train' else get_test_df()
        self.size = len(self.df)
        self.data_cols = ['band_1', 'band_2', 'inc_angle']

        if data_type == 'Train':
            self.get_y = self._get_target
        else:
            self.get_y = self._get_index

    def __len__(self):
        return self.size

    def _get_target(self, index):
        return self.df.loc[index, 'is_iceberg']

    def _get_index(self, index):
        return index

    def __getitem__(self, index):

        if index < 0 or index >= self.size:
            raise IndexError()

        b1, b2, a = self.df.loc[index, self.data_cols]
        x = np.zeros((75 * 75, 2), dtype=np.float32)
        x[:, 0] = b1
        x[:, 1] = b2
        x = x.reshape((75, 75, 2))
        return (x, a), self.get_y(index)


class _ToTensor(object):
    def __call__(self, x):
        return torch.from_numpy(x.transpose([2, 0, 1]))


CUSTOM_TRANSFORMS = {
    "RandomCrop": RandomCrop,
    "RandomChoice": RandomChoice,
    "RandomAffine": RandomAffine,
    "RandomFlip": RandomFlip,
    "_ToTensor": _ToTensor
}


def get_data_transforms(json_str):
    return restore_object(json_str, custom_objects=CUSTOM_TRANSFORMS, verbose_debug=False)


def x_transform(x, aug_fn):
    x, a = x
    x = aug_fn(x)
    return x, a


def y_transform(y):
    return y


def get_trainval_batches(train_aug_str, test_aug_str, fold_index, n_splits, batch_size, num_workers, seed=None):
    
    trainval_ds = IcebergDataset('Train')
    
    train_aug = get_data_transforms(train_aug_str)
    test_aug = get_data_transforms(test_aug_str)
    
    train_aug_ds = TransformedDataset(trainval_ds, 
                                      x_transforms=partial(x_transform, aug_fn=train_aug), 
                                      y_transforms=y_transform)

    val_aug_ds = TransformedDataset(trainval_ds, 
                                    x_transforms=partial(x_transform, aug_fn=test_aug), 
                                    y_transforms=y_transform)
    
    x_array = []
    y_array = []    
    for i, ((_, _), y) in enumerate(trainval_ds):
        x_array.append(i) 
        y_array.append(y)
    
    # Stratified split:
    train_indices = None
    val_indices = None
    skf = StratifiedKFold(n_splits=n_splits, random_state=seed)
    for i, (train_indices, val_indices) in enumerate(skf.split(x_array, y_array)):
        if i == fold_index:
            break

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_batches = OnGPUDataLoader(train_aug_ds,
                                    batch_size=batch_size,
                                    sampler=train_sampler,
                                    num_workers=num_workers,
                                    drop_last=True,
                                    pin_memory=True)

    val_batches = OnGPUDataLoader(val_aug_ds,
                                  batch_size=batch_size,
                                  sampler=val_sampler,
                                  num_workers=num_workers,
                                  drop_last=True,
                                  pin_memory=True)
    
    return train_batches, val_batches        
