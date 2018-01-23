
import os
import sys
import argparse
from datetime import datetime

import numpy as np
import pandas as pd

import xgboost as xgb

root_path = os.path.abspath("..")
if root_path not in sys.path:
    sys.path.append(root_path)

from common import read_config, to_readable_str
from common.torch_common_utils.training_utils import write_conf_log
from common import write_submission


def get_test_df(filename):
    test_df = pd.read_csv(filename)
    return test_df


def save_y_probas(filename, data_ids, y_probas):
    np.savez_compressed(filename, data_ids=data_ids, y_probas=y_probas)


if __name__ == "__main__":

    print("\nMake predictions\n")

    parser = argparse.ArgumentParser(description="Make predictions with trained xgb model")
    parser.add_argument('config_filepath', metavar='<config_filepath>', help="Config file")
    parser.add_argument('--do_not_write_submission', action='store_true',
                        help='Do not write a submission for every prediction')

    args = parser.parse_args()
    config_filepath = args.config_filepath
    assert os.path.exists(config_filepath), "Config file is not found at %s" % config_filepath

    config = read_config(config_filepath)
    assert 'models' in config
    assert "dataset" in config

    need_write_submission = not args.do_not_write_submission

    # Define global TTA config if defined
    n_tta = 1
    mean_fn_str = None

    now = datetime.now()
    config_filename = os.path.basename(config_filepath)
    config_filename = config_filename.replace(".json", "")
    output_path = os.path.abspath(os.path.join("..", "output",
                                               "%s_predictions_%s" %
                                               (now.strftime("%Y%m%d_%H%M"), config_filename)))
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    print("\n")
    print(to_readable_str(config))
    print("\n")

    config_str = ""
    for k, v in config.items():
        config_str += "{}: {}\n".format(k, v)
    write_conf_log(output_path, config_str)


    test_df = get_test_df(config["dataset"])

    for i, model_conf in enumerate(config['models']):

        model = xgb.Booster(model_file=model_conf["path"])
        model_name = model_conf["name"]
        print("\n --- Model %s" % model_name)

        features = model_conf["features"]
        assert len(set(features) - set(test_df.columns)) == 0, \
            "There are some model features that are not in test_df: {}".format(set(features) - set(test_df.columns))

        dtest = xgb.DMatrix(test_df[features])
        y_probas = model.predict(dtest)
        y_probas = np.expand_dims(y_probas, axis=-1)
        data_ids = test_df['id']

        filename = os.path.join(output_path, "y_probas_%i_%s.npz" % (i, model_name))
        save_y_probas(filename, data_ids, y_probas)
        if need_write_submission:
            filename = os.path.join(output_path, "y_probas_%i_%s.csv" % (i, model_name))
            write_submission(filename, y_probas)
