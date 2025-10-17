import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.utils.class_weight import compute_class_weight

# Set device to CUDA if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def create_cosine_annealing_scheduler(initial_lr=0.001, min_lr=1e-6, warmup_epochs=5, total_epochs=50):
    """
    Create a cosine annealing learning rate scheduler with warmup.
    
    Args:
        initial_lr: Initial learning rate.
        min_lr: Minimum learning rate.
        warmup_epochs: Number of warmup epochs.
        total_epochs: Total number of epochs.
    
    Returns:
        Learning rate scheduler function.
    """
    def cosine_annealing_scheduler(epoch):
        if epoch < warmup_epochs:
            # Linear warmup
            lr = initial_lr * (epoch + 1) / warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            lr = min_lr + (initial_lr - min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        
        return lr
    
    return cosine_annealing_scheduler


def visualize_lr_schedule(initial_lr=0.001, min_lr=1e-6, warmup_epochs=5, total_epochs=50, save_path=None):
    """
    Visualize the cosine annealing learning rate schedule.
    
    Args:
        initial_lr: Initial learning rate.
        min_lr: Minimum learning rate.
        warmup_epochs: Number of warmup epochs.
        total_epochs: Total number of epochs.
        save_path: Path to save the plot (optional).
    """
    import matplotlib.pyplot as plt
    
    scheduler = create_cosine_annealing_scheduler(initial_lr, min_lr, warmup_epochs, total_epochs)
    
    epochs = range(total_epochs)
    learning_rates = [scheduler(epoch) for epoch in epochs]
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, learning_rates, 'b-', linewidth=2, label='Learning Rate')
    plt.axvline(x=warmup_epochs, color='r', linestyle='--', alpha=0.7, label=f'Warmup End (Epoch {warmup_epochs})')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Cosine Annealing Learning Rate Schedule')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Learning rate schedule saved to {save_path}")
    
    plt.show()


def save_classification_metrics(y_true, y_pred, class_names, save_dir="confusion_output"):
    """
    Save classification metrics (precision, recall, f1-score, support) to a text file.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        class_names: List of class names.
        save_dir: Directory to save the metrics file.
    """
    from sklearn.metrics import classification_report
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate classification report
    report = classification_report(
        y_true, y_pred, 
        target_names=class_names, 
        digits=4, 
        zero_division=0
    )
    
    # Save to txt file
    metrics_path = os.path.join(save_dir, "classification_metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("Classification Metrics Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(report)
        f.write("\n" + "=" * 50 + "\n")
        f.write(f"Generated on: {os.path.basename(save_dir)}\n")
    
    print(f"[INFO] Classification metrics saved to: {metrics_path}")
    return metrics_path


def load_model(model_weights_path: str = None, num_classes: int = 7):
    """
    Load MobileNetV2 model (PyTorch) and weights if available.
    """
    from torchvision.models import mobilenet_v2
    model = mobilenet_v2(weights="IMAGENET1K_V1")
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)

    if model_weights_path:
        if os.path.exists(model_weights_path):
            model.load_state_dict(torch.load(model_weights_path, map_location=device))
            print(f"[INFO] Loaded weights from: {model_weights_path}")
        else:
            print(f"[WARN] Weights not found at: {model_weights_path}. Using ImageNet weights.")
    
    model.to(device)
    return model


def get_class_weights(y_train):
    """Calculate class weights for imbalanced dataset."""
    # Convert one-hot encoded to class indices
    y_train_indices = np.argmax(y_train, axis=1)
    
    # Calculate class weights
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train_indices),
        y=y_train_indices
    )
    
    # Convert to dictionary
    class_weight_dict = dict(zip(range(len(class_weights)), class_weights))
    
    return class_weight_dict


def load_data(partition_id: int, num_partitions: int = 10, img_size: int = 224, batch_size: int = 32):
    """
    Load data using torchvision ImageFolder and DataLoader for PyTorch.
    """
    base_dir = f"my_flower_app/dataset/partition_{partition_id}"
    data_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    
    def get_loader(split):
        split_dir = os.path.join(base_dir, split)
        dataset = datasets.ImageFolder(split_dir, transform=data_transforms)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        print(f"[DEBUG] {split} - Classes: {dataset.classes}")
        return loader, dataset.classes

    train_loader, class_names = get_loader("train")
    val_loader, _ = get_loader("val")
    test_loader, _ = get_loader("test")
    
    return train_loader, val_loader, test_loader, class_names
