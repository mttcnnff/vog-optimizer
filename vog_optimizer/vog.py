import sys
from typing import Dict, Tuple, List, Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import Callback
from tqdm import tqdm


def calculate_vog(grads) -> torch.Tensor:
    mean_grad = torch.sum(grads, 0) / len(grads)
    vog: torch.Tensor = torch.mean(torch.sum((grads - mean_grad) ** 2, 0) / len(grads))
    return vog


def calculate_vog_maps(example_idx_to_grads: Dict[int, List[torch.Tensor]], example_idx_to_class_labels: Dict[int, int]) -> Tuple[Dict[int, torch.Tensor], Dict[int, List[torch.Tensor]]]:
    example_idx_to_vog: Dict[int, torch.Tensor] = {}
    class_label_to_vogs: Dict[int, List[torch.Tensor]] = {}

    for example_idx in example_idx_to_grads.keys():
        grads = example_idx_to_grads[example_idx]
        class_label = example_idx_to_class_labels[example_idx]
        vog = calculate_vog(torch.vstack(grads))

        example_idx_to_vog[example_idx] = vog
        class_label_to_vogs.setdefault(class_label, []).append(vog)

    return example_idx_to_vog, example_idx_to_grads


def dataset_with_indices(cls):
    """
    Modifies the given Dataset class to return a tuple data, target, index
    instead of just data, target.

    Found here: https://discuss.pytorch.org/t/how-to-retrieve-the-sample-indices-of-a-mini-batch/7948/19
    """
    def __getitem__(self, index):
        data, target = cls.__getitem__(self, index)
        return data, target, index

    return type(cls.__name__, (cls,), {
        '__getitem__': __getitem__,
    })


class GradientSnapshotCallback(Callback):

    def __init__(self, snapshot_interval: int):
        self.snapshot_interval = snapshot_interval
        self._x_idx_to_grads: Dict[int, List[torch.Tensor]] = {}
        self._x_idx_to_y: Dict[int, int] = {}

        self._x_idx_to_vog: Optional[Dict[int, torch.Tensor]] = None
        self._y_to_vogs: Optional[Dict[int, List[torch.Tensor]]] = None

    def on_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if trainer.current_epoch % self.snapshot_interval == 0:
            train_dataloader = pl_module.train_dataloader()
            for x, y, global_example_indexes in tqdm(train_dataloader, desc="Snapshotting: ", file=sys.stdout, position=0, leave=True):
                x.requires_grad = True
                logits = pl_module.forward(x)
                loss = F.nll_loss(logits, y)
                loss.backward()
                for global_index, grad, class_label in zip(global_example_indexes.tolist(), x.grad, y):
                    self._x_idx_to_grads.setdefault(global_index, []).append(grad)
                    self._x_idx_to_y[global_index] = class_label

    def _analyze_gradients(self):
        # Analysis of Gradients
        # TODO: Add normalization
        example_index_to_vog: Dict[int, torch.Tensor] = {}
        class_label_to_vogs: Dict[int, List[torch.Tensor]] = {}

        for global_example_index in self._x_idx_to_grads.keys():
            temp_grad: torch.Tensor = torch.vstack(self._x_idx_to_grads[global_example_index])
            class_label: int = self._x_idx_to_y[global_example_index]
            # TODO: Ask about this mean calc and why we sum over all snapshots, but only divide for the subset
            # mean_grad = np.sum(np.array(self.vog[global_example_index]), axis=0) / len(temp_grad)
            vog = calculate_vog(temp_grad)

            example_index_to_vog[global_example_index] = vog
            class_label_to_vogs.setdefault(class_label, []).append(vog)


        return example_index_to_vog, class_label_to_vogs


    @property
    def example_index_to_vog(self) -> Dict[int, torch.Tensor]:
        if self._x_idx_to_vog is None:
            self._x_idx_to_vog, self._y_to_vogs = self._analyze_gradients()
        return self._x_idx_to_vog

    @property
    def class_label_to_vog(self) -> Dict[int, List[torch.Tensor]]:
        if self._y_to_vogs is None:
            self._x_idx_to_vog, self._y_to_vogs = self._analyze_gradients()
        return self._y_to_vogs

