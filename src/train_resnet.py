import torch
import torch.nn as nn
import torch.optim as optim

from dataset import get_dataloaders
from resnet_model import CIFARResNet18


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    correct = 0
    total = len(loader.dataset)

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        correct += (logits.argmax(dim=1) == labels).sum().item()

    epoch_loss = running_loss / len(loader)
    accuracy = correct / total

    return epoch_loss, accuracy


def evaluate(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = len(loader.dataset)

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            running_loss += loss.item()
            correct += (logits.argmax(dim=1) == labels).sum().item()

    loss = running_loss / len(loader)
    accuracy = correct / total

    return loss, accuracy


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    train_loader, val_loader, _ = get_dataloaders(
        batch_size=128
    )

    model = CIFARResNet18().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam([
        {
            "params": model.model.layer1.parameters(),
            "lr": 1e-4
        },
        {
            "params": model.model.layer2.parameters(),
            "lr": 1e-4
        },
        {
            "params": model.model.layer3.parameters(),
            "lr": 1e-4
        },
        {
            "params": model.model.layer4.parameters(),
            "lr": 1e-4
        },
        {
            "params": model.model.fc.parameters(),
            "lr": 1e-3
        },
    ])
    epochs = 15

    best_val_acc = 0.0

    for epoch in range(epochs):

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        val_loss, val_acc = evaluate(
            model,
            val_loader,
            criterion,
            device
        )

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {train_loss:.4f} "
            f"Train Acc: {train_acc:.2%} "
            f"Val Loss: {val_loss:.4f} "
            f"Val Acc: {val_acc:.2%}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                },
                "best_resnet18_cifar32_discriminative.pth"
            )

    print(
        f"\nBest validation accuracy: "
        f"{best_val_acc:.2%}"
    )


if __name__ == "__main__":
    main()