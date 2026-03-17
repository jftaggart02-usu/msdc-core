"""Training script for the SteeringNet neural network.

Features:
- Automatic train/validation split
- Model checkpointing (every 5 epochs and best model)
- Training history logging (JSON format)
- GPU/CPU support with automatic detection
- Resume training from checkpoint

Usage example:
```
python train_steering_net.py \
--dataset-dir /path/to/dataset \
--output-dir ./checkpoints \
--batch-size 32 \
--num-epochs 50 \
--learning-rate 0.0001 \
--color-space YCbCr
```

Args:
    --dataset-dir: Path to dataset directory containing 'images/' and 'labels.csv'
    --output-dir: Directory to save model checkpoints and training history
    --batch-size: Batch size for training
    --num-epochs: Number of training epochs
    --learning-rate: Learning rate for optimizer
    --val-split: Fraction of data to use for validation (default: 0.2)
    --image-height: Image height (default: 66, as used in DAVE-2)
    --image-width: Image width (default: 200, as used in DAVE-2)
    --color-space: Color space for images (default: RGB, DAVE-2 uses YUV/YCbCr)
    --num-workers: Number of worker processes for data loading (default: 4)
    --checkpoint: Path to checkpoint to resume training from (default: None)
"""

import argparse
import json
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from msdc_core.control.steering_dataset import SteeringDataset
from msdc_core.control.steering_net import SteeringNet


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    """Train the model for one epoch.

    Args:
        model: The neural network model
        dataloader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on (cpu/cuda)

    Returns:
        Average loss for the epoch
    """
    model.train()
    running_loss = 0.0
    num_batches = 0

    for images, steering_angles in tqdm(dataloader, desc="Training", leave=False):
        images = images.to(device)
        steering_angles = steering_angles.to(device).unsqueeze(1)  # Shape: (batch_size, 1)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, steering_angles)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    return running_loss / num_batches


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Validate the model.

    Args:
        model: The neural network model
        dataloader: DataLoader for validation data
        criterion: Loss function
        device: Device to validate on (cpu/cuda)

    Returns:
        Average validation loss
    """
    model.eval()
    running_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for images, steering_angles in tqdm(dataloader, desc="Validating", leave=False):
            images = images.to(device)
            steering_angles = steering_angles.to(device).unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, steering_angles)

            running_loss += loss.item()
            num_batches += 1

    return running_loss / num_batches


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    train_loss: float,
    val_loss: float,
    checkpoint_path: Path,
) -> None:
    """Save model checkpoint.

    Args:
        model: The neural network model
        optimizer: Optimizer
        epoch: Current epoch number
        train_loss: Training loss
        val_loss: Validation loss
        checkpoint_path: Path to save the checkpoint
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


def main() -> None:
    """Main function to train the SteeringNet model."""
    parser = argparse.ArgumentParser(description="Train SteeringNet")
    parser.add_argument(
        "--dataset-dir", type=str, required=True, help="Path to dataset directory containing images/ and labels.csv"
    )
    parser.add_argument("--output-dir", type=str, default="./checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--num-epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--val-split", type=float, default=0.2, help="Fraction of data to use for validation")
    parser.add_argument("--image-height", type=int, default=66, help="Image height (DAVE-2 uses 66)")
    parser.add_argument("--image-width", type=int, default=200, help="Image width (DAVE-2 uses 200)")
    parser.add_argument(
        "--color-space", type=str, default="RGB", choices=["RGB", "YCbCr"], help="Color space for images (DAVE-2 uses YUV/YCbCr)"
    )
    parser.add_argument("--num-workers", type=int, default=4, help="Number of worker processes for data loading")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to resume training from")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save training configuration
    config_path = output_dir / "training_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=4)
    print(f"Training configuration saved to {config_path}")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    print(f"Loading dataset from {args.dataset_dir}")
    full_dataset = SteeringDataset(
        dataset_dir=args.dataset_dir, image_size=(args.image_height, args.image_width), color_space=args.color_space
    )

    # Split into train and validation sets
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    # Initialize model
    model = SteeringNet().to(device)

    # Initialize a dummy input to compute lazy layers
    with torch.no_grad():
        # PyTorch uses the convention (batch_size, channels, height, width) for image tensors
        dummy_input = torch.randn(1, 3, args.image_height, args.image_width).to(device)
        model(dummy_input)

    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Load checkpoint if provided
    start_epoch = 0
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resumed from checkpoint: {args.checkpoint}")
        print(f"Starting from epoch {start_epoch}")

    # Training loop
    best_val_loss = float("inf")
    training_history = []

    print("\nStarting training...")
    for epoch in range(start_epoch, args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")

        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss = validate(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        # Save training history
        training_history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch + 1}.pt"
            save_checkpoint(model, optimizer, epoch, train_loss, val_loss, checkpoint_path)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = output_dir / "best_model.pt"
            save_checkpoint(model, optimizer, epoch, train_loss, val_loss, best_model_path)
            print(f"New best model saved with validation loss: {val_loss:.6f}")

    # Save final model
    final_model_path = output_dir / "final_model.pt"
    save_checkpoint(model, optimizer, args.num_epochs - 1, train_loss, val_loss, final_model_path)
    print(f"\nTraining complete! Final model saved to {final_model_path}")

    # Save training history
    history_path = output_dir / "training_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(training_history, f, indent=4)
    print(f"Training history saved to {history_path}")


if __name__ == "__main__":
    main()
