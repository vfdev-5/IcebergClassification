
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
from common.dataflow import get_test_batches
from common.torch_common_utils.deserialization import restore_object, CustomObjectEval
from common.torch_common_utils.training_utils import load_checkpoint

from common.torch_common_utils.testing_utils import predict


def save_y_probas(filename, data_ids, y_probas):
    if torch.is_tensor(y_probas):
        if y_probas.cuda:
            y_probas = y_probas.cpu()
        y_probas = y_probas.numpy()
    np.savez_compressed(filename, data_ids=data_ids, y_probas=y_probas)


if __name__ == "__main__":

    print("\nMake predictions\n")

    parser = argparse.ArgumentParser(description="Make predictions with trained model")
    parser.add_argument('config_filepath', metavar='<config_filepath>', help="Config file")

    args = parser.parse_args()
    config_filepath = args.config_filepath
    assert os.path.exists(config_filepath), "Config file is not found at %s" % config_filepath

    config = read_config(config_filepath)
    assert 'models' in config

    n_tta = 1
    mean_fn_str = None
    if 'TTA' in config:
        n_tta = config['TTA']['n_rounds']
        mean_fn_str = config['TTA']['merge_fn']
        assert mean_fn_str in ["mean", "gmean"]
        if mean_fn_str == "mean":
            merge_fn_1 = lambda acc, x: acc + x
            merge_fn_2 = lambda acc, n: acc.div_(n)
        else:
            merge_fn_1 = lambda acc, x: acc * x
            merge_fn_2 = lambda acc, n: acc.pow_(1.0 / n)

    assert 'to_proba_fn' in config
    to_proba_fn = eval(config['to_proba_fn'], globals())

    now = datetime.now()
    config_filename = os.path.basename(config_filepath)
    config_filename = config_filename.replace(".json", "")
    output_path = os.path.abspath(os.path.join("..", "output",
                                               "predictions_%s_%s" %
                                               (config_filename,
                                                now.strftime("%Y%m%d_%H%M"))))
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    print("\n")
    print(to_readable_str(config))
    print("\n")

    custom_objects = CustomObjectEval(globals=globals())

    for i, s in enumerate(config['models']):
        model = restore_object(s['name'], custom_objects=custom_objects, verbose_debug=False)
        model_name = model.__class__.__name__
        print("\n --- Model %s" % model_name)
        model = model.cuda()
        path = find_weights_filepath(s['weights_path'])
        load_checkpoint(path, model)

        y_probas_acc = None
        data_ids_tta = []
        for r in range(n_tta):
            if n_tta > 1:
                print("- TTA round : %i / %i" % (r + 1, n_tta))

            test_batches = get_test_batches(config['test_aug'], config['batch_size'],
                                            config['num_workers'], config['seed'])

            ret = predict(model, test_batches, to_proba_fn)
            if ret is None:
                exit(1)

            del test_batches

            y_probas, data_ids = ret
            if y_probas_acc is None:
                y_probas_acc = y_probas
            else:
                y_probas_acc = merge_fn_1(y_probas_acc, y_probas)
            data_ids_tta.append(data_ids)

        if n_tta > 1:
            print("- Merge TTA predictions")
            # Check data_ids_tta. They should all indentical
            data_ids = data_ids_tta[0]
            for next_data_ids in data_ids_tta[1:]:
                assert (data_ids == next_data_ids).all()

            y_probas = merge_fn_2(y_probas_acc, n_tta)
            filename = os.path.join(output_path, "y_probas_%i_%s.npz" % (i, model_name))
            save_y_probas(filename, data_ids, y_probas)
        else:
            y_probas = y_probas_acc
            data_ids = data_ids_tta[0]

        del y_probas, data_ids, y_probas_acc
        del model
