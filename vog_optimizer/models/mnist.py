import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from torchvision import transforms
from torchvision.datasets import MNIST

import pytorch_lightning as pl
from torchmetrics.functional import accuracy


class MNIST_with_index(MNIST):
    def __getitem__(self, index):
        data, target = super().__getitem__(index)
        return data, target, index


class MnistModel(pl.LightningModule):
    def __init__(self, data_dir='./data', learning_rate=2e-4):
        super(MnistModel, self).__init__()

        MEAN = 0.1307
        SDEV = 0.3081
        HEIGHT = 28
        WIDTH = 28
        CLASS_COUNT = 10

        self.data_dir = data_dir
        self.learning_rate = learning_rate
        self.batch_size = 32

        self.mnist_train = None
        self.mnist_test = None
        self.mnist_val = None

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN, SDEV),
        ])

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(HEIGHT * WIDTH, CLASS_COUNT),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor):
        x = self.model(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, _batch_idx):
        x, y, _idx = batch
        logits = self(x)
        return F.nll_loss(logits, y)

    def validation_step(self, batch, _batch_idx):
        x, y, _idx = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        prediction = torch.argmax(logits, dim=1)
        acc = accuracy(prediction, y)

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def prepare_data(self) -> None:
        MNIST_with_index(self.data_dir, train=True, download=True)
        MNIST_with_index(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.mnist_train, self.mnist_val = random_split(self.get_train_dataset(), [55000, 5000])

        if stage == 'test':
            self.mnist_test = self.get_test_dataset()

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.mnist_train, batch_size=self.batch_size, num_workers=4)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=1)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.mnist_test, batch_size=self.batch_size)

    def get_train_dataset(self, with_transforms=True):
        return MNIST_with_index(self.data_dir, train=True, transform=self.transform if with_transforms else None)

    def get_test_dataset(self, with_transforms=True):
        return MNIST_with_index(self.data_dir, train=False, transform=self.transform if with_transforms else None)
