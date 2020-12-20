import numpy as np
import pytorch_lightning as pl
from matplotlib import pyplot as plt
from torchvision.datasets import MNIST

from vog_optimizer.models.mnist import MnistModel
from vog_optimizer.vog import GradientSnapshotCallback
from vog_optimizer.utils import build_image_grid

if __name__ == '__main__':
    mnist_model = MnistModel()
    gradient_snapshot_tracker = GradientSnapshotCallback(2)

    mnist_trainer = pl.Trainer(
        gpus=None,
        max_epochs=10,
        progress_bar_refresh_rate=20,
        callbacks=[gradient_snapshot_tracker]
    )

    mnist_trainer.fit(mnist_model)

    sorted_vogs = gradient_snapshot_tracker.analyze_gradients()

    dataset = MNIST('./', train=True)
    lowest_5_vogs_images = [dataset[item[0]][0] for item in sorted_vogs[:5]]
    highest_5_vogs_images = [dataset[item[0]][0] for item in sorted_vogs[-5:]]
    img_grid = build_image_grid([lowest_5_vogs_images, highest_5_vogs_images])

    plt.figure(1)
    plt.imshow(img_grid)
    plt.show()





