
import os
import sys
import argparse
from glob import glob

import numpy as np

root_path = os.path.abspath("..")
if root_path not in sys.path:
    sys.path.append(root_path)


from common import write_submission


if __name__ == "__main__":

    print("\nWrite submission as average proba of multiple models predictions\n")

    parser = argparse.ArgumentParser(description="Write submission as average proba of multiple models predictions")
    parser.add_argument('path', metavar='<path>', help="Path to npz files with predictions")
    args = parser.parse_args()
    output_path = args.path

    assert os.path.exists(output_path)

    filenames = glob(os.path.join(output_path, "*.npz"))
    assert len(filenames) > 0, "No npz predictions at '%s'" % output_path

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
