
import os
import sys
import argparse

from functools import partial

import numpy as np
import cv2

from tqdm import trange

root_path = os.path.abspath("..")
if root_path not in sys.path:
    sys.path.append(root_path)

from common.dataflow import get_train_df, IcebergDataset, get_test_df
from common.dataflow import get_image
from common.imgproc_utils import *


norm_b1 = IcebergDataset.norm_b1
norm_b2 = IcebergDataset.norm_b2


def get_norm_band(b, f=norm_b1):
    return f(np.array(b[0]), b[1]).tolist()


def preprocess(df):
    """
    Add useful features:
        'inv_max_b21',
        'inv_max_avg_b1',
        'inv_max_avg_b2',
        'obj_n_cnvx_def',
        'inv_obj_f1',
        'inv_obj_area',
        'inv_obj_cnvx_def_min',
        'inv_obj_cnvx_def_mean',
        'inv_obj_cnvx_def_max',
        'obj_m2',
        'obj_m3',
        'sea_median',

    """

    print("\n Normalize bands and add band 3")
    df.loc[:, "nband_1"] = df[['band_1', 'inc_angle']].apply(partial(get_norm_band, f=norm_b1), axis=1)
    df.loc[:, "nband_2"] = df[['band_2', 'inc_angle']].apply(partial(get_norm_band, f=norm_b2), axis=1)

    print("\n Compute min, avg, max for all bands")
    df.loc[:, 'avg_b1'] = df['nband_1'].apply(np.mean)
    df.loc[:, 'max_b1'] = df['nband_1'].apply(np.max)
    df.loc[:, 'avg_b2'] = df['nband_2'].apply(np.mean)
    df.loc[:, 'max_b2'] = df['nband_2'].apply(np.max)

    print("\n Compute inv_max_avg_b1, inv_max_avg_b2, max_b21")
    df.loc[:, 'inv_max_avg_b1'] = df[['max_b1', 'avg_b1']].apply(lambda x: 1.0/(x[0] - x[1] + 1e-10), axis=1)
    df.loc[:, 'inv_max_avg_b2'] = df[['max_b2', 'avg_b2']].apply(lambda x: 1.0/(x[0] - x[1] + 1e-10), axis=1)
    df.loc[:, 'inv_max_b21'] = df[['max_b2', 'max_b1']].apply(lambda x: 1.0/(x[0] - x[1] + 1e-10), axis=1)    

    print("\n Compute object properties")
    for i in trange(len(df)):
        index = df.index[i]
        img = get_image(index, df, bands=['nband_1', 'nband_2'])        
        mask = segment_object(img)

        mask = morpho_close(mask, ksize=3)
        _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        biggest_contour = contours[np.argmax([len(c) for c in contours])]

        defects = cv2.convexityDefects(biggest_contour,
                                       cv2.convexHull(biggest_contour, returnPoints=False))
        if defects is not None:
            defects = (np.min(defects[:, 0, -1]),
                       np.max(defects[:, 0, -1]),
                       np.mean(defects[:, 0, -1]),
                       len(defects[:, 0, -1]))
        else:
            defects = (0, 0, 0, 0)

        df.loc[i, 'obj_length'] = cv2.arcLength(biggest_contour, closed=True)
        df.loc[i, 'obj_area'] = cv2.contourArea(biggest_contour)
        df.loc[i, 'inv_obj_cnvx_def_min'] = 1.0 / (defects[0] + 1e-10) if defects[0] > 0 else 0
        df.loc[i, 'inv_obj_cnvx_def_max'] = 1.0 / (defects[1] + 1e-10) if defects[1] > 0 else 0
        df.loc[i, 'inv_obj_cnvx_def_mean'] = 1.0 / (defects[2] + 1e-10) if defects[2] > 0 else 0
        df.loc[i, 'obj_n_cnvx_def'] = defects[3]

        mm = cv2.HuMoments(cv2.moments(mask))
        df.loc[i, 'obj_m2'] = mm[2][0]
        df.loc[i, 'obj_m3'] = mm[3][0]

        proc = img[:, :, 0] + img[:, :, 1]
        proc = smooth(proc, sigmaX=0.7)
        sea_patches = get_sea_8_corners(proc, size=18)
        
        sea_median = []
        for p in sea_patches:
            sea_median.append(np.median(p))
        sea_median = np.mean(sea_median)        
        df.loc[i, 'sea_median'] = 1.0/(sea_median + 1e-10)

        img_b1_fft = fft(img[:, :, 0])
        img_b2_fft = fft(img[:, :, 1])
        df.loc[i, 'b1_fft_max'] = img_b1_fft.max()
        df.loc[i, 'b2_fft_max'] = img_b2_fft.max()    
        t1 = img_b1_fft.mean() + 2*img_b1_fft.std()
        t2 = img_b2_fft.mean() + 2*img_b2_fft.std()    
        df.loc[i, 'b1_fft_mean_2std_count'] = 1.0 / (np.sum(img_b1_fft > t1) + 1e-10)
        df.loc[i, 'b2_fft_mean_2std_count'] = 1.0 / (np.sum(img_b2_fft > t2) + 1e-10)


    df.loc[:, 'inv_obj_f1'] = (df['obj_area']) / (df['obj_length'] * df['obj_length'] + 1e-10)
    df.loc[:, 'inv_obj_area'] = df['obj_area'].apply(lambda x: 1.0 / (x + 1e-10))
    return df


if __name__ == "__main__":

    print("\nStart dataset processing\n")
    parser = argparse.ArgumentParser(description="Process dataset")
    parser.add_argument('dataset_type', type=str, choices={'train', 'test'}, help="Dataset type: train or test")
    parser.add_argument('output_filename', type=str, help="Filename to store processed dataset")
    parser.add_argument('--ignore_fake_data', action='store_true',
                        help='Ignore fake data of the test dataset')

    args = parser.parse_args()
    dataset_type = args.dataset_type
    output_filename = args.output_filename
    assert not os.path.exists(output_filename), "Output filename '%s' already exists" % output_filename

    if dataset_type == "train":
        df = get_train_df()
    else:
        df = get_test_df(ignore_synth_data=args.ignore_fake_data)

    processed_df = preprocess(df)
    print("\nSave to file")
    processed_df.to_csv(output_filename, index=False)



