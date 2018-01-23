

import os
import sys
import argparse
from datetime import datetime
from operator import itemgetter
from functools import partial

import numpy as np
import pandas as pd
import xgboost as xgb


root_path = os.path.abspath("..")
if root_path not in sys.path:
    sys.path.append(root_path)

from common import read_config
from common.torch_common_utils.training_utils import write_conf_log, _write_log


def get_train_df(filename):
    train_df = pd.read_csv(filename)
    return train_df


def search_best_params(dtrainval, n_runs, n_folds, num_boost_round, early_stopping_rounds, hyper_params, seed):

    np.random.seed(seed)
    for i in range(n_runs):

        eta, max_depth, subsample, colsample_bytree = generate_params(hyper_params)
        i += 1
        seed += i
        print("\n{} : XGBoost params. ETA: {}, MAX_DEPTH: {}, SUBSAMPLE: {}, COLSAMPLE_BY_TREE: {}"
              .format(i, eta, max_depth, subsample, colsample_bytree))

        params["eta"] = eta
        params["max_depth"] = max_depth
        params["subsample"] = subsample
        params["colsample_bytree"] = colsample_bytree
        params["seed"] = seed

        cvresult = xgb.cv(params, dtrain=dtrainval,
                          seed=params["seed"],
                          num_boost_round=num_boost_round,
                          early_stopping_rounds=early_stopping_rounds,
                          nfold=n_folds,
                          stratified=True,
                          verbose_eval=early_stopping_rounds)

        min_test_eval_metric_mean = cvresult["test-%s-mean" % params["eval_metric"]].min()
        if best_params["test-%s-mean" % params["eval_metric"]] > min_test_eval_metric_mean:
            best_params["test-%s-mean" % params["eval_metric"]] = min_test_eval_metric_mean
            best_params["train-%s-mean" % params["eval_metric"]] = cvresult[
                "train-%s-mean" % params["eval_metric"]].min()
            best_params["params"] = params
            best_params["num_boost_round"] = len(cvresult)
            print("Best cv result: ", cvresult.loc[cvresult.index[-1], :])
            print("Best params: ", params)

    return best_params


def generate_params(params):
    eta_min, eta_max = params['eta'] if 'eta' in params else [0.005, 0.00005]
    max_depth_min, max_depth_max = params['max_depth'] if 'max_depth' in params else [2, 5]
    subsample_min, subsample_max = params['subsample'] if 'subsample' in params else [0.4, 0.98]
    colsample_bytree_min, colsample_bytree_max = params['colsample_bytree'] \
        if 'colsample_bytree' in params else [0.4, 0.98]

    eta = np.random.uniform(eta_min, eta_max)
    max_depth = np.random.randint(max_depth_min, max_depth_max)
    subsample = np.random.uniform(subsample_min, subsample_max)
    colsample_bytree = np.random.uniform(colsample_bytree_min, colsample_bytree_max)
    return eta, max_depth, subsample, colsample_bytree


if __name__ == "__main__":

    print("\nStart train XGB model\n")
    parser = argparse.ArgumentParser(description="Train XGB model")
    parser.add_argument("config_filepath", metavar="<config_filepath>", help="Config file")

    args = parser.parse_args()
    config_filepath = args.config_filepath
    assert os.path.exists(config_filepath), "Config file is not found at %s" % config_filepath

    config = read_config(config_filepath)

    now = datetime.now()
    config_filename = os.path.basename(config_filepath)
    config_filename = config_filename.replace(".json", "")
    output_path = os.path.abspath(os.path.join("..", "output",
                                               "%s_training_%s" % (now.strftime("%Y%m%d_%H%M"), config_filename)))
    logs_path = output_path
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)

    assert "dataset" in config
    train_df = get_train_df(config["dataset"])

    assert "features" in config
    features = train_df[config["features"]]
    labels = train_df["is_iceberg"]

    seed = config["seed"]
    dtrainval = xgb.DMatrix(features, label=labels)
    n_folds = config["n_folds"]

    params = {
        "objective": "binary:logistic",
        "booster": "gbtree",
        "eval_metric": "logloss",
        "eta": None,
        "tree_method": "auto",
        "max_depth": None,
        "subsample": None,
        "colsample_bytree": None,
        "silent": 1,
        "seed": None,
    }

    best_params = {
        "test-%s-mean" % params["eval_metric"]: 1e10,
        "params": {},
        "num_boost_round": 0
    }

    config_str = ""
    for k, v in config.items():
        config_str += "{}: {}\n".format(k, v)

    write_conf_log(logs_path, config_str)

    # Search best params
    if "best_params" not in config:
        best_params = search_best_params(dtrainval,
                                         n_runs=config["n_runs"],
                                         n_folds=n_folds,
                                         num_boost_round=config["num_boost_round"],
                                         early_stopping_rounds=config["early_stopping_rounds"],
                                         hyper_params=config['hyper_params'],
                                         seed=seed)
    else:
        best_params = config["best_params"]

    _write_log(os.path.join(logs_path, "best_params.log"), "{}".format(best_params))

    # Train model
    dtrain = xgb.DMatrix(features, label=labels)
    model = xgb.train(best_params["params"],
                      dtrain,
                      num_boost_round=best_params["num_boost_round"],
                      evals=[(dtrain, "train")],
                      verbose_eval=500)

    print("Features scores")
    print(sorted(list(model.get_fscore().items()), key=itemgetter(1), reverse=True))

    # Save model
    model_filename = os.path.join(logs_path, "model.xgb")
    model.save_model(model_filename)
