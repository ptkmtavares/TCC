import torch
import numpy as np
import torch.amp as amp
import torch.nn as nn
import torch.optim as optim
import time
import os
import logging

DELIMETER = "=" * 75
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()


class Generator(nn.Module):
    """Generator model for GAN"""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        """Initialize the generator model.

        Args:
            input_dim (int): Dimension of the input noise vector.
            output_dim (int): Dimension of the output data.
        """
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the generator model.

        Args:
            x (torch.Tensor): Input noise vector.

        Returns:
            torch.Tensor: Generated samples.
        """
        return self.model(x)


class Discriminator(nn.Module):
    """Discriminator model for GAN"""

    def __init__(self, input_dim: int) -> None:
        """Initialize the discriminator model.

        Args:
            input_dim (int): Dimension of the input data.
        """
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the discriminator model.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Output logits.
        """
        return self.model(x)


def save_checkpoint(
    G: Generator,
    D: Discriminator,
    optimizer_G: torch.optim.Optimizer,
    optimizer_D: torch.optim.Optimizer,
    epoch: int,
    checkpoint_dir: str = "checkpoints",
) -> None:
    """
    Save the model checkpoint.

    Args:
        G (Generator): Generator model.
        D (Discriminator): Discriminator model.
        optimizer_G (torch.optim.Optimizer): Optimizer for the generator.
        optimizer_D (torch.optim.Optimizer): Optimizer for the discriminator.
        epoch (int): Current epoch number.
        checkpoint_dir (str, optional): Directory to save the checkpoint. Defaults to "checkpoints".
    """
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    torch.save(
        {
            "epoch": epoch,
            "G_state_dict": G.state_dict(),
            "D_state_dict": D.state_dict(),
            "optimizer_G_state_dict": optimizer_G.state_dict(),
            "optimizer_D_state_dict": optimizer_D.state_dict(),
        },
        os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth"),
    )


def load_checkpoint(
    G: Generator,
    D: Discriminator,
    optimizer_G: torch.optim.Optimizer,
    optimizer_D: torch.optim.Optimizer,
    checkpoint_path: str,
) -> int:
    """
    Load the model checkpoint.

    Args:
        G (Generator): Generator model.
        D (Discriminator): Discriminator model.
        optimizer_G (torch.optim.Optimizer): Optimizer for the generator.
        optimizer_D (torch.optim.Optimizer): Optimizer for the discriminator.
        checkpoint_path (str): Path to the checkpoint file.

    Returns:
        int: Epoch number from the loaded checkpoint.
    """
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    G.load_state_dict(checkpoint["G_state_dict"])
    D.load_state_dict(checkpoint["D_state_dict"])
    optimizer_G.load_state_dict(checkpoint["optimizer_G_state_dict"])
    optimizer_D.load_state_dict(checkpoint["optimizer_D_state_dict"])
    return checkpoint["epoch"]


def get_latest_checkpoint(
    checkpoint_dir: str = "checkpoints", num_epochs: int = 1000
) -> str:
    """
    Get the latest checkpoint file.

    Args:
        checkpoint_dir (str, optional): Directory containing checkpoints. Defaults to "checkpoints".
        num_epochs (int, optional): Total number of epochs. Defaults to 1000.

    Returns:
        str: Path to the latest checkpoint file.
    """
    checkpoints = [
        f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_epoch_")
    ]
    if not checkpoints:
        return None

    valid_checkpoints = [
        f for f in checkpoints if int(f.split("_")[-1].split(".")[0]) <= num_epochs
    ]

    if not valid_checkpoints:
        return None

    closest_checkpoint = min(
        valid_checkpoints,
        key=lambda f: abs(int(f.split("_")[-1].split(".")[0]) - num_epochs),
    )
    return os.path.join(checkpoint_dir, closest_checkpoint)


def train_gan(
    G: Generator,
    D: Discriminator,
    train_loader: torch.utils.data.DataLoader,
    input_dim: int,
    num_epochs: int = 1000,
    n_critic: int = 1,
    device: str = "cpu",
    checkpoint_dir: str = "checkpoints",
) -> None:
    """
    Train the GAN model.

    Args:
        G (Generator): Generator model.
        D (Discriminator): Discriminator model.
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        input_dim (int): Dimension of the input noise vector.
        num_epochs (int, optional): Number of epochs to train. Defaults to 1000.
        n_critic (int, optional): Number of critic iterations per generator iteration. Defaults to 1.
        device (str, optional): Device to train on. Defaults to "cpu".
        checkpoint_dir (str, optional): Directory to save checkpoints. Defaults to "checkpoints".
    """
    try:
        criterion = nn.BCEWithLogitsLoss()
        optimizer_G = optim.Adam(G.parameters(), lr=0.0002)
        optimizer_D = optim.Adam(D.parameters(), lr=0.00065)
        scaler_G = amp.GradScaler("cuda")
        scaler_D = amp.GradScaler("cuda")

        start_epoch = 0
        latest_checkpoint = get_latest_checkpoint(checkpoint_dir, num_epochs)
        if latest_checkpoint:
            logging.info(f"Loading latest checkpoint: {latest_checkpoint}")
            start_epoch = load_checkpoint(
                G, D, optimizer_G, optimizer_D, latest_checkpoint
            )
            if start_epoch >= num_epochs:
                logging.info(
                    f"\nCheckpoint epoch is greater than or equal to num_epochs. Returning...\n"
                    f"{DELIMETER}"
                )
                return
            logging.info(
                f"\nResuming training from epoch {start_epoch}\n" f"{DELIMETER}"
            )
        save_interval = (num_epochs - start_epoch) // 4

        for epoch in range(start_epoch, num_epochs):
            epoch_start_time = time.time()

            for real_data, _ in train_loader:
                real_data = real_data.to(device, non_blocking=True)
                real_labels = torch.ones(real_data.size(0), 1, device=device) * 0.9
                fake_labels = torch.zeros(real_data.size(0), 1, device=device) + 0.1

                for _ in range(n_critic):
                    optimizer_D.zero_grad(set_to_none=True)
                    with amp.autocast("cuda"):
                        noise = torch.randn(real_data.size(0), input_dim, device=device)
                        fake_data = G(noise)
                        real_loss = criterion(D(real_data), real_labels)
                        fake_loss = criterion(D(fake_data.detach()), fake_labels)
                        d_loss = real_loss + fake_loss

                    scaler_D.scale(d_loss).backward()
                    scaler_D.step(optimizer_D)
                    scaler_D.update()

                optimizer_G.zero_grad(set_to_none=True)
                with amp.autocast("cuda"):
                    noise = torch.randn(real_data.size(0), input_dim, device=device)
                    fake_data = G(noise)
                    g_loss = criterion(D(fake_data), real_labels)

                scaler_G.scale(g_loss).backward()
                scaler_G.step(optimizer_G)
                scaler_G.update()

                if epoch % 5 == 0:
                    torch.cuda.empty_cache()
                    del noise, fake_data

            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            remaining_time = (num_epochs - (epoch + 1)) * epoch_duration

            if (epoch + 1) % save_interval == 0 or (epoch + 1) == num_epochs:
                torch.cuda.empty_cache()
                logging.info(
                    f"\nEpoch [{epoch + 1}/{num_epochs}]\n"
                    f"D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}\n"
                    f"Time for this epoch: {epoch_duration:.2f} seconds\n"
                    f"Estimated remaining time: {remaining_time:.2f} seconds\n"
                    f"{DELIMETER}"
                )
                save_checkpoint(
                    G, D, optimizer_G, optimizer_D, epoch + 1, checkpoint_dir
                )

    finally:
        torch.cuda.empty_cache()


def generate_adversarial_examples(
    G: Generator, num_samples: int, input_dim: int, device: str = "cpu"
) -> np.ndarray:
    """
    Generate adversarial examples using the trained generator.

    Args:
        G (Generator): Trained generator model.
        num_samples (int): Number of samples to generate.
        input_dim (int): Dimension of the input noise vector.
        device (str, optional): Device to generate on. Defaults to "cpu".

    Returns:
        np.ndarray: Generated samples.
    """
    try:
        G.eval()
        with torch.no_grad(), amp.autocast("cuda"):
            noise = torch.randn(num_samples, input_dim, device=device)
            generated_data = G(noise).cpu().numpy()
            return generated_data
    finally:
        torch.cuda.empty_cache()
