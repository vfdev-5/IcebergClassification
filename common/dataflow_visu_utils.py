
import numpy as np
import matplotlib.pylab as plt

import torch

from .torch_common_utils.dataflow_visu_utils import _to_ndarray, _to_str, scale_percentile


def display_image(ax, img, id_num, band_id, inc_angle, target=None, **kwargs):

    if len(img.shape) == 3 and img.shape[2] == 2:
        img3b = np.zeros(img.shape[:2] + (3, ), dtype=img.dtype)
        img3b[:, :, 0] = img[:, :, 0]
        img3b[:, :, 1] = img[:, :, 1]
        img = scale_percentile(img3b, q_min=0.0, q_max=100.0)
    elif len(img.shape) == 3 and img.shape[2] > 3:
        imgNb = np.zeros((img.shape[0], img.shape[1] * img.shape[2]), dtype=img.dtype)
        for i in range(img.shape[2]):
            imgNb[:, i*img.shape[1]:(i+1)*img.shape[1]] = scale_percentile(img[:, :, i], q_min=0.0, q_max=100.0)
        img = imgNb

    im = ax.imshow(img, **kwargs)
    ax.text(10, 4, '%s %s %.5f' % (id_num, band_id, inc_angle), color='w', backgroundcolor='m', alpha=0.7)
    if target is not None:
        ax.text(10, 14, 'y={}'.format(target), color='k', backgroundcolor='w', alpha=0.7)
    ax.axis('on')
    return im


def display_dataset(ds, max_datapoints=15, n_cols=5, figsize=(12, 6), show_info=False):

    for i, ((x, a), y) in enumerate(ds):

        if i % n_cols == 0:
            plt.figure(figsize=figsize)

        if show_info:
            shape = x.size() if torch.is_tensor(x) else x.shape
            print("x: ", type(x), shape, x[:, :, 0].min(), x[:, :, 0].max(),
                  x[:, :, 1].min(), x[:, :, 1].max())

            if torch.is_tensor(y):
                shape = y.size()
            elif isinstance(y, np.ndarray):
                shape = y.shape
            else:
                shape = 1
            print("y: ", type(y), shape)

        x = _to_ndarray(x)
        y = _to_str(y)

        ax = plt.subplot(1, n_cols, (i % n_cols) + 1)
        display_image(ax, x, i, 'b12', a)
        plt.title("Class {}".format(y))

        max_datapoints -= 1
        if max_datapoints == 0:
            break


def display_data_augmentations(ds, aug_ds, max_datapoints=15, n_cols=5, figsize=(12, 6)):

    for i, (((x1, a1), y1), ((x2, a2), y2)) in enumerate(zip(ds, aug_ds)):

        if i % n_cols == 0:
            plt.figure(figsize=figsize)

        x1 = _to_ndarray(x1)
        x2 = _to_ndarray(x2)
        y1 = _to_str(y1)
        y2 = _to_str(y2)

        ax = plt.subplot(2, n_cols, (i % n_cols) + 1)
        display_image(ax, x1, i, 'b12', a1)
        plt.title("Orig. Class {}".format(y1))

        ax = plt.subplot(2, n_cols, (i % n_cols) + 1 + n_cols)
        display_image(ax, x2, i, 'b12', a2)
        plt.title("Aug. Class {}".format(y2))

        max_datapoints -= 1
        if max_datapoints == 0:
            break


def display_batches(batches_ds, max_batches=3, n_cols=5, figsize=(16, 6), suptitle_prefix=""):

    for i, ((batch_x, batch_a), batch_y) in enumerate(batches_ds):

        plt.figure(figsize=figsize)
        plt.suptitle(suptitle_prefix + "Batch %i" % i)
        for j in range(len(batch_x)):
            if j > 0 and j % n_cols == 0:
                plt.figure(figsize=figsize)

            x = batch_x[j, ...]
            y = batch_y[j, ...] if isinstance(batch_y, np.ndarray) else batch_y[j]

            x = _to_ndarray(x)
            y = _to_str(y)

            ax = plt.subplot(1, n_cols, (j % n_cols) + 1)
            display_image(ax, x, -1, 'b12', batch_a[j])
            plt.title("Class {}".format(y))

        max_batches -= 1
        if max_batches == 0:
            break