from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from config import config


train_data = MNIST(
    root="data",
    train=True,
    transform=ToTensor(),
    download=True
)

test_data = MNIST(
    root="data",
    train=False,
    transform=ToTensor(),
    download=True
)

data_loaders = {
    "train": DataLoader(
        train_data,
        batch_size=config['training_batch_size'],
        shuffle=True,
        num_workers=1
    ),
    "test": DataLoader(
        test_data,
        batch_size=config['testing_batch_size'],
        shuffle=True,
        num_workers=1
    ),
}
