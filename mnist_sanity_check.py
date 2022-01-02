import pytorch_lightning as pl
from matplotlib import pyplot as plt
from torchvision.datasets import MNIST

from vog_optimizer.models.mnist import MnistModel
from vog_optimizer.utils import build_image_grid
from vog_optimizer.vog import GradientSnapshotCallback

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

    sorted_vogs = sorted(gradient_snapshot_tracker.get_example_index_to_vog().items(), key=lambda item: item[1])

    dataset = MNIST('./', train=True)

    lowest_5_idxs = [item[0] for item in sorted_vogs[:5]]
    highest_5_idxs = [item[0] for item in sorted_vogs[-5:]]
    print(f"Lowest 5 vogs: {lowest_5_idxs}")
    print(f"Highest 5 vogs: {highest_5_idxs}")

    lowest_5_vogs_images = [dataset[idx][0] for idx in lowest_5_idxs]
    highest_5_vogs_images = [dataset[idx][0] for idx in highest_5_idxs]
    img_grid = build_image_grid([lowest_5_vogs_images, highest_5_vogs_images])

    plt.figure(1)
    plt.imshow(img_grid)
    plt.show()





