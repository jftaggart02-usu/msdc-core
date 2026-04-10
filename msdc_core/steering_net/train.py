"""This module implements a training loop for the steering net."""

import json

import torch
from torch.optim import Adam
from torch.nn import MSELoss
from torch.utils.data import DataLoader, random_split
from msdc_core.steering_net.steering_dataset import SteeringDataset
from msdc_core.steering_net.steering_net import SteeringNet


def train(
    dataset_dir: str,
    model_checkpoint_dir: str,
    num_epochs: int = 10,
    batch_size: int = 32,
    test_percent: float = 0.2,
    learning_rate: float = 0.001,
    initial_checkpoint_path: str | None = None,
) -> None:
    """Trains the steering net on the given dataset.

    Args:
        dataset_dir: Path to the dataset directory.
        model_checkpoint_dir: Directory to save model checkpoints.
        num_epochs: Number of epochs to train for.
        batch_size: Batch size for training.
        test_percent: Percentage of the dataset to use for testing/validation.
        learning_rate: Learning rate for the optimizer.
        initial_checkpoint_path: Optional path to a model checkpoint to initialize the model with.
    """

    # Create dataset
    dataset = SteeringDataset(dataset_dir=dataset_dir)

    # Divide dataset into train and test
    test_size = int(test_percent * len(dataset))
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Instantiate network with random parameters
    if initial_checkpoint_path is not None:
        model = SteeringNet()
        model.load_state_dict(torch.load(initial_checkpoint_path))
        print(f"Loaded initial checkpoint from: {initial_checkpoint_path}")
    else:
        model = SteeringNet()

    # Instantiate Adam optimizer and loss criterion
    criterion = MSELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for i, (images, steering_angles) in enumerate(train_loader):
            optimizer.zero_grad()
            predicted_steering_angles = model(images)
            loss = criterion(predicted_steering_angles, steering_angles)
            loss.backward()
            optimizer.step()

            # Print training and validation loss every 10 batches
            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(train_loader)}], Training Loss: {loss.item():.4f}")

        # Calculate validation loss on test dataset
        with torch.no_grad():
            total_loss = 0
            for images, steering_angles in test_loader:
                predicted_steering_angles = model(images)
                loss = criterion(predicted_steering_angles, steering_angles)
                total_loss += loss.item()
            avg_loss = total_loss / len(test_loader)
            print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_loss:.4f}")

        # Save model checkpoint
        checkpoint_path = f"{model_checkpoint_dir}/steering_net_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model checkpoint saved at: {checkpoint_path}")

    # Save training params in a JSON file in the checkpoint directory for future reference
    training_params = {
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "test_percent": test_percent,
        "learning_rate": learning_rate,
        "initial_checkpoint_path": initial_checkpoint_path,
    }
    with open(f"{model_checkpoint_dir}/training_params.json", "w") as f:
        json.dump(training_params, f, indent=4)


if __name__ == "__main__":
    # Example usage
    train(
        dataset_dir="/home/jftaggart02/datasets/trial_03_augmented_02",
        model_checkpoint_dir="/home/jftaggart02/models/trial_03_augmented_02/train_01",
        num_epochs=5,
        batch_size=32,
        test_percent=0.2,
        learning_rate=0.001,
    )
