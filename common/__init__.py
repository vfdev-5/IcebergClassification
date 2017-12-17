
import os
import json
from glob import glob

import pandas as pd

from .dataflow import SAMPLE_SUBMISSION_PATH


def read_config(filepath):
    with open(filepath, 'r') as handler:
        config = json.load(handler)
    return config


def to_readable_str(config):
    config_str = ""
    for k, v in config.items():
        config_str += "{}: {}\n".format(k, v)
    return config_str


def find_weights_filepath(path, prefix=""):
    best_model_filenames = glob(os.path.join(path, '%s*.pth*' % prefix))
    assert len(best_model_filenames) > 0 and os.path.exists(best_model_filenames[0]), \
        "Failed to find any weights at '%s'" % path
    return best_model_filenames[0]


def write_submission(output_filename, y_probas, proba_thresholds=None):
    submission_df = pd.read_csv(SAMPLE_SUBMISSION_PATH)
    y_is_iceberg_proba = y_probas[:, 1] if y_probas.shape[1] == 2 else y_probas[:, 0]
    if proba_thresholds is not None and \
        isinstance(proba_thresholds, (list, tuple)) and len(proba_thresholds) == 2:
        y_is_iceberg_proba[y_is_iceberg_proba < proba_thresholds[0]] = proba_thresholds[0]
        y_is_iceberg_proba[y_is_iceberg_proba > proba_thresholds[1]] = proba_thresholds[1]
    submission_df['is_iceberg'] = y_is_iceberg_proba
    submission_df.to_csv(output_filename, index=False)
