import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def get_dataloaders(batch_size=128):
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.RandomErasing(
            p=0.25,
            scale=(0.02, 0.2),
            ratio=(0.3, 3.3)
        ),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    train_full = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=train_transform,
    )

    val_full = datasets.CIFAR10(
        root="./data",
        train=True,
        download=False,
        transform=eval_transform,
    )

    test_dataset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=eval_transform,
    )

    generator = torch.Generator().manual_seed(42)

    train_dataset, _ = random_split(
        train_full,
        [45000, 5000],
        generator=generator,
    )

    _, val_dataset = random_split(
        val_full,
        [45000, 5000],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
if __name__ == '__main__':
    # Initialisation des loaders
    train_loader, val_loader, test_loader = get_dataloaders()

    # Récupération en toute sécurité d'un lot d'images
    images, labels = next(iter(train_loader))

    # Affichages des dimensions
    print("Dimensions des images :", images.shape)
    print("Dimensions des labels :", labels.shape)
    print("Taille Dataset Entraînement :", len(train_loader.dataset))
    print("Taille Dataset Validation :", len(val_loader.dataset))
    print("Taille Dataset Test :", len(test_loader.dataset))