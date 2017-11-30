

import os
import sys
import argparse
import json
from datetime import datetime

root_path = os.path.abspath("..")
if root_path not in sys.path:
    sys.path.append(root_path)

from torch.nn import CrossEntropyLoss
from torch.nn.functional import softmax
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau

from common.dataflow import get_trainval_batches
from common.models import *
from common.torch_common_utils.deserialization import restore_object, CustomObjectEval

from common.torch_common_utils.training_utils import train_one_epoch, validate
from common.torch_common_utils.training_utils import write_conf_log, write_csv_log, _write_log, \
    save_checkpoint, optimizer_to_str
from common.torch_common_utils.training_utils import EarlyStopping
from common.torch_common_utils.training_utils import accuracy


def accuracy_logits(y_logits, y_true):
    y_pred = softmax(y_logits).data
    return accuracy(y_pred, y_true)


def read_config(filepath):
    with open(filepath, 'r') as handler:
        config = json.load(handler)
    return config


def to_readable_str(config):
    config_str = ""
    for k, v in config.items():
        config_str += "{}: {}\n".format(k, v)
    return config_str


def train(model, criterion, optimizer, batches, lr_schedulers, early_stopping, logs_path, config):

    train_batches, val_batches = batches
    best_acc = 0
    config_str = ""
    for k, v in config.items():
        config_str += "{}: {}\n".format(k, v)

    write_conf_log(logs_path, config_str)
    _write_log(os.path.join(logs_path, "model.log"), "{}".format(model.__dict__))

    write_csv_log(logs_path, "epoch,train_loss,train_acc,val_loss,val_acc")

    for epoch in range(0, config['n_epochs']):

        for scheduler in lr_schedulers:
            if isinstance(scheduler, _LRScheduler):
                scheduler.step()

        # train for one epoch
        ret = train_one_epoch(model, train_batches, criterion, optimizer, epoch,
                              config['n_epochs'],
                              avg_metrics=[accuracy_logits, ])
        if ret is None:
            return False
        train_loss, train_acc = ret

        # evaluate on validation set
        if epoch % config['validate_every_epoch'] == 0:
            ret = validate(model, val_batches, criterion, avg_metrics=[accuracy_logits, ])
            if ret is None:
                return False
            val_loss, val_acc = ret

            write_csv_log(logs_path, "%i,%f,%f,%f,%f" % (epoch, train_loss, train_acc, val_loss, val_acc))

            for scheduler in lr_schedulers:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(val_acc)

            if early_stopping(val_acc):
                break

            # remember best accuracy and save checkpoint
            if val_acc > best_acc:
                best_acc = max(val_acc, best_acc)
                save_checkpoint(logs_path, 'val_acc', {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'val_acc': val_acc,
                    'optimizer': optimizer.state_dict()})
    return True

if __name__ == "__main__":

    print("\nStart train model\n")
    parser = argparse.ArgumentParser(description="Train model")
    parser.add_argument('config_filepath', metavar='<config_filepath>', help="Config file")

    args = parser.parse_args()
    config_filepath = args.config_filepath
    assert os.path.exists(config_filepath), "Config file is not found at %s" % config_filepath

    config = read_config(config_filepath)

    model_name = list(config['model'].keys())[0]

    now = datetime.now()
    output_path = os.path.abspath(os.path.join("..", "output",
                                               "logs_%s_%s" % (model_name, now.strftime("%Y%m%d_%H%M"))))

    fold_indices = config['fold_index']
    if not isinstance(fold_indices, (tuple, list)):
        fold_indices = [fold_indices]

    for fold_index in fold_indices:
        print("\n\n -------- Train on fold %i / %i ---------- \n\n" % (fold_index + 1, len(fold_indices)))
        print(to_readable_str(config))
        print("\n")

        # setup model before optimizer
        custom_objects = CustomObjectEval(globals=globals())
        model = restore_object(config['model'], custom_objects=custom_objects, verbose_debug=False)
        model = model.cuda()

        # Setup optimizer
        custom_objects = CustomObjectEval(globals=globals())
        optimizer = restore_object(config['optimizer'], custom_objects=custom_objects, verbose_debug=False)
        print(optimizer_to_str(optimizer))

        params_to_insert = {'optimizer': '_opt'}
        custom_objects = {"_opt": optimizer}

        lr_schedulers_conf = config['lr_schedulers']

        lr_schedulers = []
        if not isinstance(lr_schedulers_conf, (tuple, list)):
            lr_schedulers_conf = [lr_schedulers_conf]

        for s in lr_schedulers:
            scheduler = restore_object(s, params_to_insert=params_to_insert,
                                       custom_objects=custom_objects,
                                       verbose_debug=False)
            lr_schedulers.append(scheduler)

        early_stopping = None
        if 'early_stopping' in config:
            early_stopping = EarlyStopping(**config['early_stopping'])

        criterion = CrossEntropyLoss().cuda()
        train_batches, val_batches = get_trainval_batches(config['train_aug'],
                                                          config['test_aug'],
                                                          n_splits=config['n_splits'],
                                                          fold_index=fold_index,
                                                          batch_size=config['batch_size'],
                                                          num_workers=config['num_workers'],
                                                          seed=config['seed'])

        logs_path = os.path.join(output_path, "fold_%i" % fold_index)
        if not os.path.exists(logs_path):
            os.makedirs(logs_path)

        if not train(model, criterion, optimizer, [train_batches, val_batches],
                     lr_schedulers, early_stopping, logs_path, config):
            break
