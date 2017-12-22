
import os
from functools import partial

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold

import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler

from .torch_common_utils.imgaug import RandomAffine, RandomChoice, RandomFlip, RandomCrop, RandomApply, CenterCrop
from .torch_common_utils.dataflow import TransformedDataset, OnGPUDataLoader
from .torch_common_utils.deserialization import restore_object
from .imgproc_utils import smart_crop

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


def get_raw_train_df():
    df = pd.read_json(TRAIN_JSON_PATH)
    return df


def is_generated(a):
    return a * 1e4 - int(a * 1e4) > 0.00001


def get_train_df(with_size_field=False):
    global _TRAIN_DF
    if _TRAIN_DF is None:
        df = pd.read_json(TRAIN_JSON_PATH)
        df = preprocess(df)
        if with_size_field:
            common_path = os.path.join(root_path, 'common')
            assert os.path.exists(os.path.join(common_path, 'big_small_ships.npz')), "File 'big_small_ships.npz' is not found"
            assert os.path.exists(os.path.join(common_path, 'big_small_icebergs.npz')), "File 'big_small_icebergs.npz' is not found"
            big_small_ships = np.load(os.path.join(common_path, 'big_small_ships.npz'))
            big_small_icebergs = np.load(os.path.join(common_path, 'big_small_icebergs.npz'))
            df.loc[big_small_ships['index'], "is_small"] = big_small_ships['y_classes']
            df.loc[big_small_icebergs['index'], "is_small"] = big_small_icebergs['y_classes']
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


def get_index(image_id, df):
    return df[df['id'] == image_id].index[0]


def get_image_by_id(image_id, df):
    index = get_index(image_id, df)
    return get_image(index, df)


def get_inc_angle(index, df):
    return df.loc[index, 'inc_angle']


def get_target(index, df):
    return int(df.loc[index, 'is_iceberg'])


class IcebergDataset(Dataset):

    def __init__(self, data_type='Train',
                 limit_n_samples=None,
                 normalized_inc_angle=True,
                 return_object_size_hint=False,
                 smart_crop_size=None):
        assert data_type in ['Train', 'Test']
        if data_type == "Test":
            assert not return_object_size_hint
        super(Dataset, self).__init__()

        self.normalized_inc_angle = normalized_inc_angle
        self.smart_crop_size = smart_crop_size
        self.return_object_size_hint = return_object_size_hint

        self.df = get_train_df(with_size_field=True) if data_type == 'Train' else get_test_df()

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

    def _get_size_hint(self, index):
        return int(self.df.loc[index, 'is_small'])

    def _get_id(self, index):
        return self.df.loc[index, 'id']

    @staticmethod
    def norm_b1(x, a):
        return 0.5 * (x + (0.455748230884 * (a - 2 * 30) - 3.3218050865))

    @staticmethod
    def norm_b2(x, a):
        return 0.5 * (x + (0.280327982183 * (a - 2 * 30) - 15.6226798323))

    def __getitem__(self, index):

        if index < 0 or index >= self.size:
            raise IndexError()

        b1, b2, a = self.df.loc[index, self.data_cols]
        x = np.zeros((75 * 75, 2), dtype=np.float32)
        if self.normalized_inc_angle:
            x[:, 0] = self.norm_b1(b1, a)
            x[:, 1] = self.norm_b2(b2, a)
        x = x.reshape((75, 75, 2))
        if self.smart_crop_size is not None:
            x = smart_crop(x, self.smart_crop_size)

        if self.return_object_size_hint:
            return (x, a, self._get_size_hint(index)), self.get_y(index)
        else:
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
    "CenterCrop": CenterCrop,
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


def x_transform_with_meta(x, aug_fn):
    x, a = x
    x = x.astype(np.float32)
    a = a.astype(np.float32)

    x_meta = [a, ]
    for i in range(x.shape[2]):
        x_meta.extend([x[:, :, i].min(), x[:, :, i].mean(), x[:, :, i].max()])
    x_meta = np.array(x_meta, dtype=np.float32)

    x = aug_fn(x)
    return x, x_meta


def id_transform(y):
    return y


def to_tensor(y):
    return torch.Tensor([y])


def get_trainval_batches(train_aug_str, test_aug_str, fold_index, n_splits, batch_size, num_workers,
                         seed=None, limit_n_samples=None):

    trainval_ds = IcebergDataset('Train', limit_n_samples=limit_n_samples)

    train_aug = get_data_transforms(train_aug_str)
    val_aug = get_data_transforms(test_aug_str)

    train_aug_ds = TransformedDataset(trainval_ds,
                                      x_transforms=partial(x_transform, aug_fn=train_aug),
                                      y_transforms=id_transform)

    val_aug_ds = TransformedDataset(trainval_ds,
                                    x_transforms=partial(x_transform, aug_fn=val_aug),
                                    y_transforms=id_transform)

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

    trainval_ds = IcebergDataset('Train',
                                 normalized_inc_angle=True,
                                 limit_n_samples=limit_n_samples)

    train_aug = get_data_transforms(train_aug_str)
    val_aug = get_data_transforms(test_aug_str)

    train_aug_ds = TransformedDataset(trainval_ds,
                                      x_transforms=partial(x_transform, aug_fn=train_aug),
                                      y_transforms=to_tensor)

    val_aug_ds = TransformedDataset(trainval_ds,
                                    x_transforms=partial(x_transform, aug_fn=val_aug),
                                    y_transforms=to_tensor)

    # Integrate size to Kfold stratified split
    _trainval_ds = IcebergDataset('Train', limit_n_samples=limit_n_samples, return_object_size_hint=True)
    x_array = []
    y_array = []
    new_classes = {
        (0, 0): 0,
        (0, 1): 1,
        (1, 0): 2,
        (1, 1): 3,
    }
    for i, ((_, _, is_small), y) in enumerate(_trainval_ds):
        x_array.append(i)
        y = (int(y), int(is_small))
        y_array.append(new_classes[y])

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


def get_cropped_trainval_batches(train_aug_str, test_aug_str, fold_index, n_splits, batch_size, num_workers,
                                 seed=None, limit_n_samples=None):

    trainval_ds = IcebergDataset('Train',
                                 limit_n_samples=limit_n_samples,
                                 normalized_inc_angle=True,
                                 smart_crop_size=48)

    train_aug = get_data_transforms(train_aug_str)
    val_aug = get_data_transforms(test_aug_str)

    train_aug_ds = TransformedDataset(trainval_ds,
                                      x_transforms=partial(x_transform, aug_fn=train_aug),
                                      y_transforms=to_tensor)

    val_aug_ds = TransformedDataset(trainval_ds,
                                    x_transforms=partial(x_transform, aug_fn=val_aug),
                                    y_transforms=to_tensor)

    # Integrate size to Kfold stratified split
    _trainval_ds = IcebergDataset('Train',
                                  limit_n_samples=limit_n_samples,
                                  return_object_size_hint=True)
    x_array = []
    y_array = []
    new_classes = {
        (0, 0): 0,
        (0, 1): 1,
        (1, 0): 2,
        (1, 1): 3,
    }
    for i, ((_, _, is_small), y) in enumerate(_trainval_ds):
        x_array.append(i)
        y = (int(y), int(is_small))
        y_array.append(new_classes[y])

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


def get_trainval_batches_with_metadata(train_aug_str, test_aug_str, fold_index, n_splits, batch_size, num_workers,
                                       seed=None, limit_n_samples=None):

    trainval_ds = IcebergDataset('Train',
                                 normalized_inc_angle=True,
                                 limit_n_samples=limit_n_samples)

    train_aug = get_data_transforms(train_aug_str)
    val_aug = get_data_transforms(test_aug_str)

    train_aug_ds = TransformedDataset(trainval_ds,
                                      x_transforms=partial(x_transform_with_meta, aug_fn=train_aug),
                                      y_transforms=to_tensor)

    val_aug_ds = TransformedDataset(trainval_ds,
                                    x_transforms=partial(x_transform_with_meta, aug_fn=val_aug),
                                    y_transforms=to_tensor)


    # Integrate size to Kfold stratified split
    _trainval_ds = IcebergDataset('Train', limit_n_samples=limit_n_samples, return_object_size_hint=True)
    x_array = []
    y_array = []
    new_classes = {
        (0, 0): 0,
        (0, 1): 1,
        (1, 0): 2,
        (1, 1): 3,
    }
    for i, ((_, _, is_small), y) in enumerate(_trainval_ds):
        x_array.append(i)
        y = (int(y), int(is_small))
        y_array.append(new_classes[y])

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


def get_crop48_trainval_batches_with_metadata(train_aug_str, test_aug_str, fold_index, n_splits, batch_size, num_workers,
                                              seed=None, limit_n_samples=None):

    trainval_ds = IcebergDataset('Train',
                                 limit_n_samples=limit_n_samples,
                                 normalized_inc_angle=True,
                                 smart_crop_size=48)

    train_aug = get_data_transforms(train_aug_str)
    val_aug = get_data_transforms(test_aug_str)

    train_aug_ds = TransformedDataset(trainval_ds,
                                      x_transforms=partial(x_transform_with_meta, aug_fn=train_aug),
                                      y_transforms=to_tensor)

    val_aug_ds = TransformedDataset(trainval_ds,
                                    x_transforms=partial(x_transform_with_meta, aug_fn=val_aug),
                                    y_transforms=to_tensor)


    # Integrate size to Kfold stratified split
    _trainval_ds = IcebergDataset('Train', limit_n_samples=limit_n_samples, return_object_size_hint=True)
    x_array = []
    y_array = []
    new_classes = {
        (0, 0): 0,
        (0, 1): 1,
        (1, 0): 2,
        (1, 1): 3,
    }
    for i, ((_, _, is_small), y) in enumerate(_trainval_ds):
        x_array.append(i)
        y = (int(y), int(is_small))
        y_array.append(new_classes[y])

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


# def get_resampled_trainval_batches_single_class(train_aug_str, test_aug_str, fold_index, n_splits, batch_size, num_workers,
#                                                 seed=None, limit_n_samples=None):
#
#     trainval_ds = IcebergDataset('Train', limit_n_samples=limit_n_samples)
#
#     train_aug = get_data_transforms(train_aug_str)
#     val_aug = get_data_transforms(test_aug_str)
#
#     train_aug_ds = TransformedDataset(trainval_ds,
#                                       x_transforms=id_transform,
#                                       y_transforms=to_tensor)
#
#     val_aug_ds = TransformedDataset(trainval_ds,
#                                     x_transforms=partial(x_transform, aug_fn=val_aug),
#                                     y_transforms=to_tensor)
#
#     resampled_train_aug_ds = ResampledDataset(train_aug_ds, n=5,
#                                               x_transforms=partial(x_transform, aug_fn=train_aug))
#
#     x_array = []
#     y_array = []
#     for i, ((_, _), y) in enumerate(trainval_ds):
#         x_array.append(i)
#         y_array.append(y)
#
#     # Stratified split:
#     train_indices = None
#     val_indices = None
#     skf = StratifiedKFold(n_splits=n_splits, random_state=seed)
#     for i, (train_indices, val_indices) in enumerate(skf.split(x_array, y_array)):
#         if i == fold_index:
#             break
#
#     train_sampler = SubsetRandomSampler(train_indices)
#     val_sampler = SubsetRandomSampler(val_indices)
#
#     train_batches = OnGPUDataLoader(train_aug_ds,
#                                     batch_size=batch_size,
#                                     sampler=train_sampler,
#                                     num_workers=num_workers,
#                                     drop_last=True,
#                                     pin_memory=True)
#
#     val_batches = OnGPUDataLoader(val_aug_ds,
#                                   batch_size=batch_size,
#                                   sampler=val_sampler,
#                                   num_workers=num_workers,
#                                   drop_last=True,
#                                   pin_memory=True)
#
#     return train_batches, val_batches


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

    test_ds = IcebergDataset('Test', normalized_inc_angle=True, limit_n_samples=limit_n_samples)

    test_aug = get_data_transforms(test_aug_str)

    test_aug_ds = TransformedDataset(test_ds,
                                     x_transforms=partial(x_transform, aug_fn=test_aug),
                                     y_transforms=id_transform)

    test_batches = OnGPUDataLoader(test_aug_ds,
                                   batch_size=batch_size,
                                   shuffle=False,
                                   num_workers=num_workers,
                                   drop_last=False,
                                   pin_memory=True)
    return test_batches


def get_test_batches_with_metadata(test_aug_str, batch_size, num_workers,
                                   seed=None, limit_n_samples=None):

    test_ds = IcebergDataset('Test', limit_n_samples=limit_n_samples)

    test_aug = get_data_transforms(test_aug_str)

    test_aug_ds = TransformedDataset(test_ds,
                                     x_transforms=partial(x_transform_with_meta, aug_fn=test_aug),
                                     y_transforms=id_transform)

    test_batches = OnGPUDataLoader(test_aug_ds,
                                   batch_size=batch_size,
                                   shuffle=False,
                                   num_workers=num_workers,
                                   drop_last=False,
                                   pin_memory=True)
    return test_batches


def get_crop48_test_batches_with_metadata(test_aug_str, batch_size, num_workers,
                                          seed=None, limit_n_samples=None):

    test_ds = IcebergDataset('Test',
                             limit_n_samples=limit_n_samples,
                             normalized_inc_angle=True,
                             smart_crop_size=48)

    test_aug = get_data_transforms(test_aug_str)

    test_aug_ds = TransformedDataset(test_ds,
                                     x_transforms=partial(x_transform_with_meta, aug_fn=test_aug),
                                     y_transforms=id_transform)

    test_batches = OnGPUDataLoader(test_aug_ds,
                                   batch_size=batch_size,
                                   shuffle=False,
                                   num_workers=num_workers,
                                   drop_last=False,
                                   pin_memory=True)
    return test_batches