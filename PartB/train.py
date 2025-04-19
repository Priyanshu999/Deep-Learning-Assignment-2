# ========================== Imports & Installations ==========================


# Core Libraries
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.datasets import INaturalist
import pytorch_lightning as pl
import wandb
import matplotlib.pyplot as plt
import numpy as np
import random
import torchvision.models as models

# import wandb
# import matplotlib.pyplot as plt
# import numpy as np
import seaborn as sns
from io import BytesIO
from PIL import Image

# ============================ Argument Parser ================================


def parse_args():
    parser = argparse.ArgumentParser(description="CNN Hyperparameter Configuration")

    # parser.add_argument("-nf", "--num_filters", type=int, default=32)
    # parser.add_argument(
    #     "-fo",
    #     "--filter_organisation",
    #     type=str,
    #     choices=["same", "double", "half"],
    #     default="double",
    # )
    # parser.add_argument("-dr", "--dropout_rate", type=float, default=0)
    parser.add_argument("-e", "--epochs", type=int, default=5)
    # parser.add_argument("-fs", "--filter_size", type=int, default=5)
    # parser.add_argument(
    #     "-neu",
    #     "--dense_neurons",
    #     type=int,
    #     help="Number of neurons in dense layer",
    #     default=512,
    # )
    # parser.add_argument(
    #     "-a",
    #     "--activation",
    #     type=str,
    #     choices=["ReLU", "GELU", "LeakyReLU", "SiLU", "Mish"],
    #     default="GELU",
    # )
    # parser.add_argument("-da", "--data_augmentation", type=str, default="True")
    # parser.add_argument("-bn", "--batch_norm", type=str, default="True")

    return parser.parse_args()


args = parse_args()


# ======================== Set Seeds for Reproducibility ======================

torch.manual_seed(2)
random.seed(2)
np.random.seed(2)

# =============================== Device Setup ================================


def set_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


device = set_device()
print("Currently Using ::", device)

# ============================== Dataset Split ================================

data_path = "/content/drive/MyDrive/nature_12k/inaturalist_12K/train"
output_path = "train_val"

splitfolders.ratio(input=data_path, output=output_path, seed=42, ratio=(0.8, 0.2))

# ============================ Data Preparation ===============================


def configure_loaders(augment_data):
    config = {
        "input_size": 224,
        "scale_range": (0.08, 1.0),
        "norm_mean": [0.485, 0.456, 0.406],
        "norm_std": [0.229, 0.224, 0.225],
        "loader_params": {
            "batch_size": 64,
            "num_workers": 4,
            "pin_memory": True,
            "persistent_workers": True,
        },
    }

    def create_base_pipeline():
        return [
            transforms.RandomResizedCrop(
                config["input_size"], scale=config["scale_range"]
            ),
            transforms.ToTensor(),
        ]

    augmentation_modules = (
        [transforms.RandomHorizontalFlip(p=0.5), transforms.RandomRotation(degrees=30)]
        if augment_data == True
        else []
    )

    pipeline = create_base_pipeline()
    pipeline[1:1] = augmentation_modules
    pipeline.append(transforms.Normalize(config["norm_mean"], config["norm_std"]))

    test_pipe = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(config["norm_mean"], config["norm_std"]),
        ]
    )

    data_paths = {
        "train": "train_val/train",
        "validation": "train_val/val",
        "test": "/content/drive/MyDrive/nature_12k/inaturalist_12K/val",
    }

    train_ds = ImageFolder(data_paths["train"], transforms.Compose(pipeline))
    val_ds = ImageFolder(data_paths["validation"], test_pipe)
    test_ds = ImageFolder(data_paths["test"], test_pipe)

    def create_loader(dataset, shuffle=False):
        return DataLoader(dataset, shuffle=shuffle, **config["loader_params"])

    return (
        create_loader(train_ds, shuffle=True),
        create_loader(val_ds),
        create_loader(test_ds),
    )

def freeze_layers(net, mode, n):
    """
    Freeze specified layers of a neural network.

    Args:
      net (torch.nn.Module): model to modify
      mode (str): one of "start", "middle", "end", "freeze_all"
      n (int): how many layers to freeze (or around center if mode="middle")

    Raises:
      ValueError: if n is out of bounds or mode is invalid
    """
    children = list(net.named_children())
    L = len(children)
    if not (0 <= n < L):
        raise ValueError(f"n must be in [0, {L-1}], got {n}")

    # Determine which layer indices to freeze
    if mode == "start":
        to_freeze = set(range(n))
        msg = f"Frozen first {n} layer(s)"
    elif mode == "end":
        to_freeze = set(range(L - n, L))
        msg = f"Frozen last {n} layer(s)"
    elif mode == "middle":
        mid = L // 2
        lo = max(0, mid - n)
        hi = min(L, mid + n)
        to_freeze = set(range(lo, hi))
        msg = f"Frozen middle layers {lo}–{hi - 1}"
    elif mode == "freeze_all":
        # everything except the last layer
        to_freeze = set(range(L - 1))
        msg = "Frozen all but the last layer"
    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    # Apply freezing
    for idx, (_, lyr) in enumerate(children):
        if idx in to_freeze:
            for p in lyr.parameters():
                p.requires_grad = False

    print(msg)


def _activation_registry(activation_name):
    """Non-linear response function selector"""
    registry = {
        "ReLU": nn.ReLU(),  # Standard rectification
        "GELU": nn.GELU(),  # Gaussian error linear unit
        "SiLU": nn.SiLU(),  # Sigmoid-weighted linear unit
        "Mish": nn.Mish(),  # Self-regularized non-linearity
        "LeakyReLU": nn.LeakyReLU(),  # Negative slope preservation
    }

    # Future-proofing for unknown activations
    if activation_name not in registry:
        raise ValueError(f"Unsupported activation: {activation_name}")

    return registry[activation_name]



def train(num_cycles, network, train_loader, val_loader, logging_mode, strategy, k):
    """Orchestrate model training with stability enhancements"""
    freeze_layers(network, strategy, k)
    # Optimization configuration
    loss_metric = nn.CrossEntropyLoss()
    optimization_policy = {
        'lr': 1e-4,
        'betas': (0.9, 0.999),
        'grad_clip': 5.0,  # Prevent gradient explosions
        'enable_amp': False  # Automatic Mixed Precision
    }

    # Parameter update engine
    optim = torch.optim.Adam(
        network.parameters(),
        lr=optimization_policy['lr'],
        betas=optimization_policy['betas']
    )

    # Training state tracking
    phase_metrics = {
        'train': {'correct': 0, 'total': 0, 'loss': 0.0},
        'val': {'correct': 0, 'total': 0, 'loss': 0.0}
    }

    # Learning rate warmup scheduler (no actual scaling)
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optim, lr_lambda=lambda epoch: 1.0
    )

    for cycle in range(num_cycles):
        # Phase 1: Parameter Update
        network.train()
        phase_metrics['train'] = {k: 0 for k in phase_metrics['train']}

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # Hardware acceleration protocol
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward propagation
            predictions = network(inputs)
            batch_loss = loss_metric(predictions, targets)

            # Backward propagation with safety measures
            optim.zero_grad()
            batch_loss.backward()

            # Gradient normalization safeguard
            torch.nn.utils.clip_grad_norm_(
                network.parameters(),
                optimization_policy['grad_clip']
            )

            # Parameter update
            optim.step()

            # Metric aggregation
            phase_metrics['train']['loss'] += batch_loss.item()
            _, predicted_labels = torch.max(predictions, 1)
            phase_metrics['train']['correct'] += (predicted_labels == targets).sum().item()
            phase_metrics['train']['total'] += targets.size(0)

            # Progress monitoring
            if (batch_idx+1) % 25 == 0:
                print(f'Epoch [{cycle+1}/{num_cycles}], Batch [{batch_idx+1}/{len(train_loader)}]')

        # Phase 1 metrics calculation
        train_acc = 100.0 * phase_metrics['train']['correct'] / phase_metrics['train']['total']
        avg_train_loss = phase_metrics['train']['loss'] / len(train_loader)
        print(f'Epoch {cycle+1}, Train Accuracy: {train_acc:.2f}%, Avg Loss: {avg_train_loss:.4f}')

        # Phase 2: Model Validation
        network.eval()
        phase_metrics['val'] = {k: 0 for k in phase_metrics['val']}

        with torch.no_grad():
            for val_inputs, val_targets in val_loader:
                val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
                val_predictions = network(val_inputs)
                val_loss = loss_metric(val_predictions, val_targets)

                # Prediction consensus
                _, val_predicted = torch.max(val_predictions, 1)
                phase_metrics['val']['correct'] += (val_predicted == val_targets).sum().item()
                phase_metrics['val']['total'] += val_targets.size(0)
                phase_metrics['val']['loss'] += val_loss.item()

        # Phase 2 metrics calculation
        val_acc = 100.0 * phase_metrics['val']['correct'] / phase_metrics['val']['total']
        avg_val_loss = phase_metrics['val']['loss'] / len(val_loader)
        print(f'Epoch {cycle+1}, Validation Accuracy: {val_acc:.2f}%, Avg Loss: {avg_val_loss:.4f}')

        # External logging interface
        if logging_mode == "wandb":
            _log_training_artifacts(
                cycle+1, avg_train_loss, train_acc, avg_val_loss, val_acc
            )

    # Final model capability score
    return val_acc

def _log_training_artifacts(cycle, train_loss, train_acc, val_loss, val_acc):
    """Record training trajectory for analysis"""
    wandb.log({
        'Epoch': cycle,
        'Training Loss': train_loss,
        'Training Accuracy': train_acc,
        'Validation Loss': val_loss,
        'Validation Accuracy': val_acc
    })

def load_model(device):
    model = models.googlenet(pretrained=True)
    last_layer_in_features = model.fc.in_features
    model.fc = nn.Linear(last_layer_in_features, 10)
    model = model.to(device)
    return model


classes = [
    "Amphibia",
    "Animalia",
    "Arachnida",
    "Aves",
    "Fungi",
    "Insecta",
    "Mammalia",
    "Mollusca",
    "Plantae",
    "Reptilia",
]


def test_model(network, data_loader):
    """Execute model evaluation with diagnostic analytics"""
    # Evaluation protocol configuration
    eval_profile = {
        "loss_function": nn.CrossEntropyLoss(),
        "sample_capture_interval": 200,  # Diagnostic imaging frequency
        "precision_mode": "fp32",  # Evaluation precision
        "enable_metrics": True,  # Comprehensive reporting
    }

    # Performance tracking
    performance_stats = {
        "correct": 0,
        "total": 0,
        "loss": 0.0,
        "diagnostic_images": [],
        "prediction_records": [],
    }

    # Hardware optimization
    compute_device = next(network.parameters()).device

    with torch.inference_mode():
        sample_counter = 0
        for batch_inputs, batch_labels in data_loader:
            # Data standardization protocol
            batch_inputs = batch_inputs.to(compute_device)
            batch_labels = batch_labels.to(compute_device)

            # Model inference
            predictions = network(batch_inputs)

            # Loss computation
            batch_loss = eval_profile["loss_function"](predictions, batch_labels)
            performance_stats["loss"] += batch_loss.item()

            # Prediction analysis
            _, predicted_classes = torch.max(predictions, 1)
            performance_stats["correct"] += (
                (predicted_classes == batch_labels).sum().item()
            )
            performance_stats["total"] += batch_labels.size(0)

            # Diagnostic image capture
            if eval_profile["enable_metrics"]:
                for idx in range(batch_inputs.size(0)):
                    sample_counter += 1
                    if sample_counter % eval_profile["sample_capture_interval"] in (
                        1,
                        2,
                        3,
                    ):
                        if sample_counter % eval_profile["sample_capture_interval"] in (
                            1,
                            2,
                            3,
                        ):
                            performance_stats["diagnostic_images"].append(
                                batch_inputs[idx]
                            )
                            performance_stats["prediction_records"].append(
                                (
                                    batch_labels[idx].item(),
                                    predicted_classes[idx].item(),
                                )
                            )
                            print(
                                f"Class Verification: Actual: {classes[batch_labels[idx]]}, Predicted: {classes[predicted_classes[idx]]}"
                            )

        # Final metric computation
        accuracy = 100.0 * performance_stats["correct"] / performance_stats["total"]
        avg_loss = performance_stats["loss"] / len(data_loader)

        # Result certification
        print(f"Model Diagnostics :: Accuracy: {accuracy:.2f}% | Loss: {avg_loss:.4f}")
        print(f'Total Samples Analyzed: {performance_stats["total"]}')

    return (
        performance_stats["diagnostic_images"],
        performance_stats["prediction_records"],
    )


# test_images, test_labels = test_model(model, test_dl)


sns.set_style("white")


def display_images_with_predictions(
    test_images, test_labels, classes, num_rows=10, num_cols=3, log_to_wandb=False
):
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 4, num_rows * 3))
    axes = axes.flatten()
    total = num_rows * num_cols

    for i in range(total):
        ax = axes[i]
        img = np.transpose(test_images[i].cpu().numpy(), (1, 2, 0))
        img_min, img_max = img.min(), img.max()
        img = (img - img_min) / (img_max - img_min + 1e-5)

        ax.imshow(img)
        ax.axis("off")

        true_label = test_labels[i][0]
        pred_label = test_labels[i][1]
        correct = true_label == pred_label

        emoji = "✓" if correct else "✗"
        label_color = "green" if correct else "red"

        ax.set_title(
            f"{emoji} True: {classes[true_label]}\nPred: {classes[pred_label]}",
            fontsize=9,
            color=label_color,
            loc="center",
            pad=10,
        )

        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_color(label_color)
            spine.set_linestyle("--")

    plt.subplots_adjust(hspace=0.6, wspace=0.2)
    plt.tight_layout()

    if log_to_wandb:
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        image = Image.open(buf)  # Convert BytesIO to PIL Image
        wandb.log({"Predictions": wandb.Image(image)})
        buf.close()
        plt.close(fig)
        plt.show()

    
    else:
        plt.show()

model = load_model(device)
train_loader , val_loader , test_loader = configure_loaders(True)
epochs=args.epochs
strategy='start'
k=5
train(epochs,model,train_loader,val_loader,"print_on", strategy, k)
test_images, test_labels = test_model(model, test_loader)
display_images_with_predictions(test_images, test_labels, classes, log_to_wandb=False)
# display_images_with_predictions(test_images, test_labels, classes, log_to_wandb=False)