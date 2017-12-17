

import os
import sys
import argparse
from datetime import datetime

root_path = os.path.abspath("..")
if root_path not in sys.path:
    sys.path.append(root_path)

from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.nn.functional import softmax, sigmoid
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau

from common import read_config, to_readable_str
from common.dataflow import *
from common.models import *
from common.torch_common_utils.deserialization import restore_object, CustomObjectEval

from common.torch_common_utils.training_utils import train_one_epoch, validate
from common.torch_common_utils.training_utils import write_conf_log, write_csv_log, _write_log, \
    save_checkpoint, optimizer_to_str
from common.torch_common_utils.training_utils import EarlyStopping
from common.torch_common_utils.training_utils import accuracy
from common.torch_common_utils.nn_utils import print_trainable_parameters


def accuracy_logits(y_logits, y_true, to_proba_fn=softmax):
    y_pred = to_proba_fn(y_logits).data
    return accuracy(y_pred, y_true)


def train(model, criterion, optimizer, batches, lr_schedulers, early_stopping, logs_path, config):

    train_batches, val_batches = batches

    config_str = ""
    for k, v in config.items():
        config_str += "{}: {}\n".format(k, v)

    write_conf_log(logs_path, config_str)
    _write_log(os.path.join(logs_path, "model.log"), "{}".format(model.__dict__))

    write_csv_log(logs_path, "epoch,train_loss,train_acc,val_loss,val_acc")

    mode = "min"
    best_val_loss = 10e5 if mode == "min" else 0
    if mode == "min":
        comp_fn = lambda current, best: current < best
    else:
        comp_fn = lambda current, best: current > best

    if "to_proba_fn" in config:
        to_proba_fn = eval(config["to_proba_fn"], globals())
    else:
        to_proba_fn = partial(softmax, dim=1)

    for epoch in range(0, config['n_epochs']):

        if epoch > 0 and epoch % 10 == 0:
            print(optimizer_to_str(optimizer))

        for scheduler in lr_schedulers:
            if isinstance(scheduler, _LRScheduler):
                scheduler.step()

        # train for one epoch
        ret = train_one_epoch(model, train_batches, criterion, optimizer, epoch,
                              config['n_epochs'],
                              avg_metrics=[partial(accuracy_logits, to_proba_fn=to_proba_fn)])
        if ret is None:
            return False
        train_loss, train_acc = ret

        # evaluate on validation set
        if (epoch + 1) % config['validate_every_epoch'] == 0:
            ret = validate(model, val_batches, criterion,
                           avg_metrics=[partial(accuracy_logits, to_proba_fn=to_proba_fn)])
            if ret is None:
                return False
            val_loss, val_acc = ret

            write_csv_log(logs_path, "%i,%f,%f,%f,%f" % (epoch, train_loss, train_acc, val_loss, val_acc))

            for scheduler in lr_schedulers:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(val_loss)

            if early_stopping(val_loss):
                break

            # remember best logloss and save checkpoint
            if comp_fn(val_loss, best_val_loss):
                best_val_loss = val_loss
                save_checkpoint(logs_path, 'val_loss', {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'val_loss': val_loss,
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

    now = datetime.now()
    config_filename = os.path.basename(config_filepath)
    config_filename = config_filename.replace(".json", "")
    output_path = os.path.abspath(os.path.join("..", "output",
                                               "%s_training_%s" % (now.strftime("%Y%m%d_%H%M"), config_filename)))

    fold_indices = config['fold_index']
    if not isinstance(fold_indices, (tuple, list)):
        fold_indices = [fold_indices]

    if "get_trainval_batches_fn" in config:
        get_trainval_batches_fn = eval(config["get_trainval_batches_fn"], globals())
    else:
        get_trainval_batches_fn = get_trainval_batches

    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])

    for fold_index in fold_indices:
        print("\n\n -------- Train on fold %i / %i ---------- \n\n" % (fold_index + 1, len(fold_indices)))
        print(to_readable_str(config))
        print("\n")

        # setup model before optimizer
        custom_objects = CustomObjectEval(globals=globals())
        model = restore_object(config['model'], custom_objects=custom_objects, verbose_debug=False)
        model = model.cuda()

        print_trainable_parameters(model)

        # Setup optimizer
        # Put model into custom_objects
        model in custom_objects
        optimizer = restore_object(config['optimizer'], custom_objects=custom_objects, verbose_debug=False)
        print(optimizer_to_str(optimizer))

        criterion = restore_object(config['criterion'], custom_objects=custom_objects, verbose_debug=False)
        criterion = criterion.cuda()

        params_to_insert = {'optimizer': '_opt'}
        custom_objects = {"_opt": optimizer}
        lr_schedulers_conf = config['lr_schedulers']

        lr_schedulers = []
        if not isinstance(lr_schedulers_conf, (tuple, list)):
            lr_schedulers_conf = [lr_schedulers_conf]

        for s in lr_schedulers_conf:
            scheduler = restore_object(s, params_to_insert=params_to_insert,
                                       custom_objects=custom_objects,
                                       verbose_debug=False)
            lr_schedulers.append(scheduler)

        early_stopping = None
        if 'early_stopping' in config:
            early_stopping = EarlyStopping(**config['early_stopping'])

        train_batches, val_batches = get_trainval_batches_fn(config['train_aug'],
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
