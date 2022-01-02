import logging
import sys
from typing import Dict, List

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import Callback
from tqdm import tqdm

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler(stream=sys.stdout))


def calculate_vog(grads) -> torch.Tensor:
    mean_grad = torch.sum(grads, 0) / len(grads)
    vog: torch.Tensor = torch.mean(torch.sum((grads - mean_grad) ** 2, 0) / len(grads))
    return vog


# def calculate_vog_maps(
#     example_idx_to_grads: Dict[int, List[torch.Tensor]],
#     example_idx_to_class_labels: Dict[int, int],
#     vog_cal: str
# ) -> Dict[int, torch.Tensor]:
#     # Analysis of Gradients
#     # TODO: Add normalization
#     example_index_to_vog: Dict[int, torch.Tensor] = {}
#     class_label_to_vogs: Dict[int, List[torch.Tensor]] = {}
#
#     for global_example_index in example_idx_to_grads.keys():
#         class_label: int = example_idx_to_class_labels[global_example_index]
#         # TODO: Ask about this mean calc and why we sum over all snapshots, but only divide for the subset
#         # mean_grad = np.sum(np.array(self.vog[global_example_index]), axis=0) / len(temp_grad)
#         vog = calculate_vog(torch.vstack(example_idx_to_grads[global_example_index]))
#         example_index_to_vog[global_example_index] = vog
#         class_label_to_vogs.setdefault(class_label, []).append(vog)
#
#     class_label_to_vog_mu = {class_label: torch.mean(torch.stack(vogs, dim=0)) for class_label, vogs in class_label_to_vogs.items()}
#     class_label_to_vog_std = {class_label: torch.std(torch.stack(vogs, dim=0)) for class_label, vogs in class_label_to_vogs.items()}
#
#     if vog_cal == 'normalize':
#         log.info("Calculating normalized vog")
#         example_index_to_normalized_vog = {}
#         for idx, grad in example_index_to_vog.items():
#             class_label = example_idx_to_class_labels[idx]
#             mu = class_label_to_vog_mu[class_label]
#             std = class_label_to_vog_std[class_label]
#             example_index_to_normalized_vog[idx] = (grad - mu)/std
#         return example_index_to_normalized_vog
#     elif vog_cal == 'abs_normalize':
#         log.info("Calculating abs normalized vog")
#         example_index_to_abs_normalized_vog = {}
#         for idx, grad in example_index_to_vog.items():
#             class_label = example_idx_to_class_labels[idx]
#             mu = class_label_to_vog_mu[class_label]
#             std = class_label_to_vog_std[class_label]
#             example_index_to_abs_normalized_vog[idx] = (grad - mu)/std
#         return example_index_to_abs_normalized_vog
#     else:
#         log.info("Calculating non-normalized vog")
#         return example_index_to_vog


def dataset_with_indices(cls):
    """
    Modifies the given Dataset class to return a tuple data, target, index
    instead of just data, target.

    Found here: https://discuss.pytorch.org/t/how-to-retrieve-the-sample-indices-of-a-mini-batch/7948/19
    """

    def __getitem__(self, index):
        data, target = cls.__getitem__(self, index)
        return data, target, index

    return type(cls.__name__, (cls,), {"__getitem__": __getitem__,})


class GradientSnapshotCallback(Callback):
    def __init__(self, snapshot_interval: int):
        self.snapshot_interval = snapshot_interval
        self._x_idx_to_grads: Dict[int, List[torch.Tensor]] = {}
        self._x_idx_to_y: Dict[int, int] = {}

    def on_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if trainer.current_epoch % self.snapshot_interval == 0:
            train_dataloader = pl_module.train_dataloader()
            for x, y, global_example_indexes in tqdm(
                train_dataloader,
                desc="Snapshotting: ",
                file=sys.stdout,
                position=0,
                leave=True,
            ):
                x.requires_grad = True
                logits = pl_module.forward(x)
                loss = F.nll_loss(logits, y)
                loss.backward()
                for global_index, grad, class_label in zip(
                    global_example_indexes.tolist(), x.grad, y
                ):
                    self._x_idx_to_grads.setdefault(global_index, []).append(grad)
                    self._x_idx_to_y[global_index] = class_label

    def get_example_index_to_vog(self) -> Dict[int, torch.Tensor]:
        log.info("Calculating non-normalized vog")
        example_index_to_vog: Dict[int, torch.Tensor] = {}

        for global_example_index, grads in self._x_idx_to_grads.items():
            # TODO: Ask about this mean calc and why we sum over all snapshots, but only divide for the subset
            # mean_grad = np.sum(np.array(self.vog[global_example_index]), axis=0) / len(temp_grad)
            vog = calculate_vog(torch.vstack(grads))
            example_index_to_vog[global_example_index] = vog

        return example_index_to_vog


