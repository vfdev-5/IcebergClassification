
import os
from functools import partial

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold

import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler

from .torch_common_utils.imgaug import RandomAffine, RandomChoice, RandomFlip, RandomCrop, RandomApply
from .torch_common_utils.dataflow import TransformedDataset, OnGPUDataLoader
from .torch_common_utils.deserialization import restore_object

root_path = os.path.abspath("..")
INPUT_PATH = os.path.abspath(os.path.join(root_path, 'input'))

TRAIN_JSON_PATH = os.path.join(INPUT_PATH, 'train.json')
TEST_JSON_PATH = os.path.join(INPUT_PATH, 'test.json')
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_PATH, 'sample_submission.csv')


def preprocess(df):
    df['inc_angle'] = pd.to_numeric(df['inc_angle'], errors='coerce')
    # Simplest fix NaN
    df.loc[df['inc_angle'].isnull(), 'inc_angle'] = 39.26
    return df


_TRAIN_DF = None
_TEST_DF = None


def get_train_df():
    global _TRAIN_DF
    if _TRAIN_DF is None:
        df = pd.read_json(TRAIN_JSON_PATH)
        df = preprocess(df)
        _TRAIN_DF = df
    df = _TRAIN_DF
    return df


def get_test_df():
    global _TEST_DF
    if _TEST_DF is None:
        df = pd.read_json(TEST_JSON_PATH)
        df = preprocess(df)
        _TEST_DF = df
    df = _TEST_DF
    return df


def get_image(index, df):
    b1, b2 = df.loc[index, ['band_1', 'band_2']]
    x = np.zeros((75 * 75, 2), dtype=np.float32)
    x[:, 0] = b1
    x[:, 1] = b2
    x = x.reshape((75, 75, 2))
    return x


def get_image_by_id(image_id, df):
    index = df[df['id'] == image_id].index[0]
    return get_image(index, df)


def get_inc_angle(index, df):
    return df.loc[index, 'inc_angle']


def get_target(index, df):
    return int(df.loc[index, 'is_iceberg'])


class IcebergDataset(Dataset):

    def __init__(self, data_type='Train', limit_n_samples=None):
        assert data_type in ['Train', 'Test']
        super(Dataset, self).__init__()

        self.df = get_train_df() if data_type == 'Train' else get_test_df()

        if limit_n_samples:
            self.df = self.df[:limit_n_samples]

        self.size = len(self.df)
        self.data_cols = ['band_1', 'band_2', 'inc_angle']

        if data_type == 'Train':
            self.get_y = self._get_target
        else:
            self.get_y = self._get_id

    def __len__(self):
        return self.size

    def _get_target(self, index):
        return int(self.df.loc[index, 'is_iceberg'])

    def _get_id(self, index):
        return self.df.loc[index, 'id']

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


class ToFiveBands(object):
    def __call__(self, x):
        xs = np.expand_dims(x[:, :, 1] - x[:, :, 0], axis=-1)
        xa = np.expand_dims(x[:, :, 1] + x[:, :, 0], axis=-1)
        xm = np.expand_dims(x[:, :, 1] * x[:, :, 0], axis=-1)
        x5b = np.concatenate((x, xs, xa, xm), axis=-1)
        return x5b


class ToThreeBands(object):
    def __call__(self, x):
        xa = np.expand_dims(x[:, :, 1] + x[:, :, 0], axis=-1)
        x3b = np.concatenate((x, xa), axis=-1)
        return x3b


class _Normalize(object):
    def __call__(self, x):
        for i in range(x.size(0)):
            meanv = torch.mean(x[i, :, :])
            minv = torch.min(x[i, :, :])
            maxv = torch.max(x[i, :, :])
            x[i, :, :].add_(-meanv)
            x[i, :, :].mul_(1.0 / (maxv - minv))
        return x


CUSTOM_TRANSFORMS = {
    "RandomCrop": RandomCrop,
    "RandomChoice": RandomChoice,
    "RandomAffine": RandomAffine,
    "RandomFlip": RandomFlip,
    "RandomApply": RandomApply,
    "_ToTensor": _ToTensor,
    "_Normalize": _Normalize,
    "ToFiveBands": ToFiveBands,
    "ToThreeBands": ToThreeBands,
}


def get_data_transforms(json_str):
    return restore_object(json_str, custom_objects=CUSTOM_TRANSFORMS, verbose_debug=False)


def normalize_inc_angle(a):
    return (a - 30.0) / (50.0 - 30.0) - 0.5


def x_transform(x, aug_fn):
    x, a = x
    x = x.astype(np.float32)
    x = aug_fn(x)
    a = normalize_inc_angle(a)
    a = a.astype(np.float32)
    return x, a


def y_transform(y):
    return y


def y_transform2(y):
    return torch.Tensor([y])


def get_trainval_batches(train_aug_str, test_aug_str, fold_index, n_splits, batch_size, num_workers,
                         seed=None, limit_n_samples=None):

    trainval_ds = IcebergDataset('Train', limit_n_samples=limit_n_samples)

    train_aug = get_data_transforms(train_aug_str)
    val_aug = get_data_transforms(test_aug_str)

    train_aug_ds = TransformedDataset(trainval_ds,
                                      x_transforms=partial(x_transform, aug_fn=train_aug),
                                      y_transforms=y_transform)

    val_aug_ds = TransformedDataset(trainval_ds,
                                    x_transforms=partial(x_transform, aug_fn=val_aug),
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


def get_trainval_batches_single_class(train_aug_str, test_aug_str, fold_index, n_splits, batch_size, num_workers,
                                      seed=None, limit_n_samples=None):

    trainval_ds = IcebergDataset('Train', limit_n_samples=limit_n_samples)

    train_aug = get_data_transforms(train_aug_str)
    val_aug = get_data_transforms(test_aug_str)

    train_aug_ds = TransformedDataset(trainval_ds,
                                      x_transforms=partial(x_transform, aug_fn=train_aug),
                                      y_transforms=y_transform2)

    val_aug_ds = TransformedDataset(trainval_ds,
                                    x_transforms=partial(x_transform, aug_fn=val_aug),
                                    y_transforms=y_transform2)

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


def get_test_batches(test_aug_str, batch_size, num_workers,
                     seed=None, limit_n_samples=None):

    test_ds = IcebergDataset('Test', limit_n_samples=limit_n_samples)

    test_aug = get_data_transforms(test_aug_str)

    test_aug_ds = TransformedDataset(test_ds,
                                     x_transforms=partial(x_transform, aug_fn=test_aug))

    test_batches = OnGPUDataLoader(test_aug_ds,
                                   batch_size=batch_size,
                                   shuffle=False,
                                   num_workers=num_workers,
                                   drop_last=False,
                                   pin_memory=True)
    return test_batches


def get_test_batches_single_class(test_aug_str, batch_size, num_workers,
                                  seed=None, limit_n_samples=None):

    test_ds = IcebergDataset('Test', limit_n_samples=limit_n_samples)

    test_aug = get_data_transforms(test_aug_str)

    test_aug_ds = TransformedDataset(test_ds,
                                     x_transforms=partial(x_transform, aug_fn=test_aug),
                                     y_transforms=y_transform2)

    test_batches = OnGPUDataLoader(test_aug_ds,
                                   batch_size=batch_size,
                                   shuffle=False,
                                   num_workers=num_workers,
                                   drop_last=False,
                                   pin_memory=True)
    return test_batches