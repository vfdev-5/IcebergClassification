
import os
import sys
import argparse
from datetime import datetime

import numpy as np

root_path = os.path.abspath("..")
if root_path not in sys.path:
    sys.path.append(root_path)

from common import find_weights_filepath, read_config, to_readable_str
from common.models import *
from common.dataflow import *
from common.torch_common_utils.deserialization import restore_object, CustomObjectEval
from common.torch_common_utils.training_utils import load_checkpoint

from common.torch_common_utils.testing_utils import predict
from common import write_submission


def save_y_probas(filename, data_ids, y_probas):
    if torch.is_tensor(y_probas):
        if y_probas.cuda:
            y_probas = y_probas.cpu()
        y_probas = y_probas.numpy()
    np.savez_compressed(filename, data_ids=data_ids, y_probas=y_probas)


def setup_tta(config):
    n_tta = config['n_rounds']
    mean_fn_str = config['merge_fn']
    assert mean_fn_str in ["mean", "gmean"]
    if mean_fn_str == "mean":
        merge_fn_1 = lambda acc, x: acc + x
        merge_fn_2 = lambda acc, n: acc.div_(n)
    else:
        merge_fn_1 = lambda acc, x: acc * x
        merge_fn_2 = lambda acc, n: acc.pow_(1.0 / n)
    return n_tta, merge_fn_1, merge_fn_2, mean_fn_str


if __name__ == "__main__":

    print("\nMake predictions\n")

    parser = argparse.ArgumentParser(description="Make predictions with trained model")
    parser.add_argument('config_filepath', metavar='<config_filepath>', help="Config file")
    parser.add_argument('--do_not_write_submission', action='store_true',
                        help='Do not write a submission for every prediction')

    args = parser.parse_args()
    config_filepath = args.config_filepath
    assert os.path.exists(config_filepath), "Config file is not found at %s" % config_filepath

    config = read_config(config_filepath)
    assert 'models' in config

    need_write_submission = not args.do_not_write_submission

    # Define global TTA config if defined
    n_tta = 1
    mean_fn_str = None

    global_vars = globals()
    if 'to_proba_fn' in config:
        to_proba_glob_fn = eval(config['to_proba_fn'], global_vars)

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

    if "get_test_batches_fn" in config:
        get_test_batches_fn = eval(config["get_test_batches_fn"], globals())
    else:
        get_test_batches_fn = get_test_batches

    custom_objects = CustomObjectEval(globals=globals())

    for i, model_conf in enumerate(config['models']):

        model = restore_object(model_conf['name'], custom_objects=custom_objects, verbose_debug=False)
        model_name = model.__class__.__name__
        print("\n --- Model %s" % model_name)
        model = model.cuda()
        path = find_weights_filepath(model_conf['weights_path'])
        load_checkpoint(path, model)

        test_aug_str = None
        if 'test_aug' in model_conf:
            test_aug_str = model_conf['test_aug']
        elif 'test_aug' in config:
            test_aug_str = config['test_aug']

        assert test_aug_str is not None, "Config .json should define 'test_aug' model-wide or config-wide"

        if 'to_proba_fn' in model_conf:
            to_proba_fn = eval(model_conf['to_proba_fn'], global_vars)
        else:
            to_proba_fn = to_proba_glob_fn

        if 'TTA' in model_conf:
            n_tta, merge_fn_1, merge_fn_2, mean_fn_str = setup_tta(model_conf['TTA'])
        elif 'TTA' in config:
            n_tta, merge_fn_1, merge_fn_2, mean_fn_str = setup_tta(config['TTA'])

        y_probas_acc = None
        data_ids_tta = []
        for r in range(n_tta):
            if n_tta > 1:
                print("- TTA round : %i / %i" % (r + 1, n_tta))

            test_batches = get_test_batches_fn(test_aug_str, config['batch_size'],
                                               config['num_workers'], config['seed'])

            ret = predict(model, test_batches, to_proba_fn)
            if ret is None:
                exit(1)

            del test_batches

            y_probas, data_ids = ret
            if y_probas_acc is None:
                y_probas_acc = y_probas
            else:
                assert merge_fn_1 is not None and n_tta > 1
                y_probas_acc = merge_fn_1(y_probas_acc, y_probas)
            data_ids_tta.append(data_ids)

        if n_tta > 1:
            print("- Merge TTA predictions")
            # Check data_ids_tta. They should all indentical
            data_ids = data_ids_tta[0]
            for next_data_ids in data_ids_tta[1:]:
                assert (data_ids == next_data_ids).all()

            y_probas = merge_fn_2(y_probas_acc, n_tta)
        else:
            y_probas = y_probas_acc
            data_ids = data_ids_tta[0]

        filename = os.path.join(output_path, "y_probas_%i_%s.npz" % (i, model_name))
        save_y_probas(filename, data_ids, y_probas)
        if need_write_submission:
            filename = os.path.join(output_path, "y_probas_%i_%s.csv" % (i, model_name))
            write_submission(filename, y_probas)

        del y_probas, data_ids, y_probas_acc
        del model
