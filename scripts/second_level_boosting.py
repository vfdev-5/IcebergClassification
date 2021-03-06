import os
import sys
import argparse
from glob import glob

import numpy as np

root_path = os.path.abspath("..")
if root_path not in sys.path:
    sys.path.append(root_path)

from common import write_submission
from common import read_config, to_readable_str

if __name__ == "__main__":

    print("\nSecond level boosting on multiple probas of multiple models predictions\n")

    parser = argparse.ArgumentParser(description="Write submission as average proba of multiple models predictions")
    parser.add_argument('config_filepath', metavar='<config_filepath>', help="Config file")

    args = parser.parse_args()
    config_filepath = args.config_filepath
    assert os.path.exists(config_filepath), "Config file is not found at %s" % config_filepath

    config = read_config(config_filepath)

    # N * K trained models, N different CNNs trained on K identical folds
    #
    # For a fold:
    # - we have train/validation datasets
    # - make N predictions on train/validation datasets
    # - concatenate N predictions from train dataset
    # - concatenate N predictions from validation dataset
    # - train gb trees on train dataset
    # - compute logloss on validation dataset
    #
    # Compute mean logloss on all folds
    #






    filenames = glob(os.path.join(output_path, "*.npz"))
    assert len(filenames) > 0, "No npz predictions at '%s'" % output_path

    proba_thresholds = args.proba_thresholds
    if proba_thresholds is not None:
        assert isinstance(proba_thresholds, (tuple, list)) and len(proba_thresholds) == 2
        print("Apply proba_thresholds={}".format(proba_thresholds))

    y_probas_list = []
    data_ids_list = []
    for f in filenames:
        ret = np.load(f)
        y_probas_list.append(ret['y_probas'])
        data_ids_list.append(ret['data_ids'])

    y_probas = np.mean(y_probas_list, axis=0)

    submission_filename = os.path.join("..", "results",
                                       "submission__avg_models__" +
                                       os.path.basename(output_path) + ".csv")
    write_submission(submission_filename, y_probas)
