import helpers.utils as utils
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
import params


def generate_datasets_and_dataloaders(device):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    train_dataset = datasets.MNIST(
        root="./data/",
        train=True,
        transform=transform,
        download=True,
    )

    test_dataset = datasets.MNIST(
        root="./data/",
        train=False,
        transform=transform,
        download=True,
    )

    (
        labeled_dataset,
        labels,
        unlabeled_dataset,
    ) = utils.separate_into_labeled_and_unlabeled_datasets(
        train_dataset, params.labeled_size, device
    )

    # Apply the transform to the labeled dataset
    transformed_labeled_data = []
    for sample in labeled_dataset:
        transformed_sample = transform(
            torch.reshape(sample.cpu(), [28, 28]).cpu().numpy()
        )  # Apply the transform to the tensor
        transformed_labeled_data.append(transformed_sample.to(device))

    transformed_unlabeled_data = []
    for sample in unlabeled_dataset:
        transformed_sample = transform(
            torch.reshape(sample.cpu(), [28, 28]).cpu().numpy()
        )
        transformed_unlabeled_data.append(transformed_sample.to(device))

    transformed_unlabeled_dataset = TensorDataset(
        torch.stack(transformed_unlabeled_data)
    )
    unlabeled_dataloader = DataLoader(
        transformed_unlabeled_dataset,
        params.batch_size,
        shuffle=True,
    )

    transformed_labeled_data = torch.stack(transformed_labeled_data, dim=0)
    return (
        test_dataset,
        transformed_labeled_data,
        labels,
        unlabeled_dataloader,
    )
