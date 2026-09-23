import json
from pathlib import Path

import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from dataset import get_dataloaders
from resnet_model import CIFARResNet18



CHECKPOINT_PATH = "best_resnet18_cifar32_discriminative.pth"

OUTPUT_DIR = Path("../outputs/evaluation")

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

def evaluate_model(model, loader, device):
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            predictions = logits.argmax(dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_labels, all_predictions


def calculate_metrics(labels, predictions):
    accuracy = accuracy_score(labels, predictions)

    report = classification_report(
        labels,
        predictions,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    return accuracy, report



def save_confusion_matrix(labels, predictions, filename):
    matrix = confusion_matrix(labels, predictions)

    fig, ax = plt.subplots(figsize=(10, 10))

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=CLASS_NAMES,
    )

    display.plot(
        ax=ax,
        xticks_rotation=45,
        values_format="d",
    )

    ax.set_title("CIFAR-10 Confusion Matrix")

    plt.tight_layout()

    output_path = OUTPUT_DIR / filename
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved confusion matrix: {output_path}")

def print_metrics(name, accuracy, report):
    print(f"{name} RESULTS")
    print(f"Accuracy: {accuracy:.4%}")

    print("\nPer-class metrics:")

    print(
        f"{'Class':<15}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'F1':>12}"
    )

    print("-" * 51)

    for class_name in CLASS_NAMES:

        metrics = report[class_name]

        print(
            f"{class_name:<15}"
            f"{metrics['precision']:>12.4f}"
            f"{metrics['recall']:>12.4f}"
            f"{metrics['f1-score']:>12.4f}"
        )

    print("\nOverall:")

    print(
        f"Macro Precision: "
        f"{report['macro avg']['precision']:.4f}"
    )

    print(
        f"Macro Recall:    "
        f"{report['macro avg']['recall']:.4f}"
    )

    print(
        f"Macro F1:        "
        f"{report['macro avg']['f1-score']:.4f}"
    )

    print(
        f"Weighted F1:     "
        f"{report['weighted avg']['f1-score']:.4f}"
    )


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=128
    )
    model = CIFARResNet18().to(device)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print(
        f"Loaded checkpoint from epoch "
        f"{checkpoint['epoch'] + 1}"
    )

    print(
        f"Checkpoint validation accuracy: "
        f"{checkpoint['val_acc']:.4%}"
    )

    print("\nEvaluating validation set...")

    val_labels, val_predictions = evaluate_model(
        model,
        val_loader,
        device,
    )

    val_accuracy, val_report = calculate_metrics(
        val_labels,
        val_predictions,
    )

    print_metrics(
        "VALIDATION",
        val_accuracy,
        val_report,
    )

    save_confusion_matrix(
        val_labels,
        val_predictions,
        "confusion_matrix_validation.png",
    )


    print("\nEvaluating test set...")

    test_labels, test_predictions = evaluate_model(
        model,
        test_loader,
        device,
    )

    test_accuracy, test_report = calculate_metrics(
        test_labels,
        test_predictions,
    )

    print_metrics(
        "TEST",
        test_accuracy,
        test_report,
    )

    save_confusion_matrix(
        test_labels,
        test_predictions,
        "confusion_matrix_test.png",
    )


    results = {
        "checkpoint": CHECKPOINT_PATH,
        "checkpoint_epoch": checkpoint["epoch"] + 1,
        "checkpoint_validation_accuracy": checkpoint["val_acc"],
        "validation": {
            "accuracy": val_accuracy,
            "macro_precision": val_report["macro avg"]["precision"],
            "macro_recall": val_report["macro avg"]["recall"],
            "macro_f1": val_report["macro avg"]["f1-score"],
            "weighted_f1": val_report["weighted avg"]["f1-score"],
        },
        "test": {
            "accuracy": test_accuracy,
            "macro_precision": test_report["macro avg"]["precision"],
            "macro_recall": test_report["macro avg"]["recall"],
            "macro_f1": test_report["macro avg"]["f1-score"],
            "weighted_f1": test_report["weighted avg"]["f1-score"],
        },
    }

    metrics_path = OUTPUT_DIR / "metrics.json"

    with open(metrics_path, "w") as file:
        json.dump(
            results,
            file,
            indent=4,
        )

    print(f"\nSaved metrics: {metrics_path}")

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()