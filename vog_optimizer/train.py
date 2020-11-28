from typing import Optional

import pytorch_lightning as pl

from vog_optimizer.models import MnistModel


def train_mnist(max_epochs: int = 3, gpus: Optional[int] = None):
    mnist_model = MnistModel()
    trainer = pl.Trainer(gpus=gpus, max_epochs=max_epochs, progress_bar_refresh_rate=20)
    trainer.fit(mnist_model)

