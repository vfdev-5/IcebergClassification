
import os
import sys
import argparse
from glob import glob

import numpy as np

root_path = os.path.abspath("..")
if root_path not in sys.path:
    sys.path.append(root_path)


from common import write_submission


def gmean(x, axis):
    n = x.shape[axis]
    return np.power(np.prod(x, axis=axis), 1.0/n)


if __name__ == "__main__":

    print("\nWrite submission as average proba of multiple models predictions\n")

    parser = argparse.ArgumentParser(description="Write submission as average proba of multiple models predictions")
    parser.add_argument('path', metavar='<path>', help="Path to npz files with predictions")
    parser.add_argument('--proba_thresholds', type=float, nargs=2, help="Probability min/max thresholds, e.g 0.03 0.97")
    parser.add_argument('--gmean', action='store_true', help='Apply gmean instead of mean')
    args = parser.parse_args()
    output_path = args.path

    assert os.path.exists(output_path)

    filenames = glob(os.path.join(output_path, "**/*.npz"), recursive=True)
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

    if args.gmean:
        y_probas_list = np.array(y_probas_list)
        y_probas = gmean(y_probas_list, axis=0)
        prefix_str = "gmean"
    else:
        y_probas = np.mean(y_probas_list, axis=0)
        prefix_str = "avg"

    submission_filename = os.path.join("..", "results",
                                       "submission__%s_models__" % prefix_str +
                                       os.path.basename(output_path) + ".csv")
    write_submission(submission_filename, y_probas)
    print("Written submission: %s" % submission_filename)
