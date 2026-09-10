import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloaders
from model import CIFARClassifier

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0
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
        correct += (torch.max(logits, 1)[1] == labels).sum().item()
    epoch_loss = running_loss / len(loader)
    train_acc = correct / total
    return epoch_loss, train_acc
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0
    correct = 0
    total = len(loader.dataset)
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            running_loss += loss.item()
            correct += (torch.max(logits, 1)[1] == labels).sum().item()
    avg_loss = running_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy
def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    train_loader, val_loader, test_loader = get_dataloaders()
    model = CIFARClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-3
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=5,
        gamma=0.1
    )
    epochs = 20
    best_val_acc = 0.0
    for epoch in range(epochs):
        train_loss,train_acc = train_one_epoch(
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
            f"Val Acc: {val_acc:.2%}",
            f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}"
        )
        scheduler.step()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
            }, "best_model.pth")
    checkpoint = torch.load("best_model.pth")
    model.load_state_dict(checkpoint["model_state_dict"])

    print(
        f"Loaded best model from epoch {checkpoint['epoch'] + 1} "
        f"with validation accuracy {checkpoint['val_acc']:.2%}"
    )

if __name__ == "__main__":
    main()