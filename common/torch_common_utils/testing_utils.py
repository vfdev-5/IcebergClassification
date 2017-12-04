
import numpy as np

import torch

from . import get_tqdm
from .nn_utils import apply_variable
from .dataflow import to_cuda


def predict(model, test_batches, to_proba_fn):

    # Switch to evaluate mode
    model.eval()
    # Initialization
    collate_fn = test_batches.collate_fn
    x, _ = test_batches.dataset[0]
    x = collate_fn([x])
    x = to_cuda(x)
    x = apply_variable(x, volatile=True)
    y = model(*x)
    n_classes = y.size(1)
    n_samples = len(test_batches.dataset)
    del x, y

    data_ids = np.empty((n_samples), dtype=np.object)
    y_probas = torch.zeros((n_samples, n_classes)).pin_memory().cuda(async=True)
    start_index = 0
    try:
        with get_tqdm(total=len(test_batches)) as pbar:
            for i, (batch_x, batch_ids) in enumerate(test_batches):
                if torch.is_tensor(batch_ids):
                    batch_size = batch_ids.size(0)
                    if batch_ids.cuda:
                        batch_ids = batch_ids.cpu()
                    batch_ids = batch_ids.numpy()
                else:
                    batch_size = len(batch_ids)

                data_ids[start_index:start_index + batch_size] = batch_ids
                batch_x = apply_variable(batch_x, volatile=True)

                # compute output and measure loss
                batch_y_logits = model(*batch_x)
                batch_y_probas = to_proba_fn(batch_y_logits).data
                y_probas[start_index:start_index + batch_size, :] = batch_y_probas

                # measure average metrics
                prefix_str = "Prediction: "
                pbar.set_description_str(prefix_str, refresh=False)
                pbar.update(1)
                start_index += batch_size

        return y_probas, data_ids
    except KeyboardInterrupt:
        return None, None