import torch
import numpy as np
import torch.amp as amp
import torch.nn as nn
import torch.optim as optim
import time
import os
import logging
from typing import Tuple
from config import DELIMITER, LOG_FORMAT


MOVING_AVERAGE_WINDOW = 100
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

class Generator(nn.Module):
    def __init__(self, input_dim: int = 61, output_dim: int = None) -> None:
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, output_dim),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

def compute_gradient_penalty(discriminator, real_samples, fake_samples, device):
    """Calculates the gradient penalty loss for WGAN GP"""
    alpha = torch.rand((real_samples.size(0), 1), device=device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = discriminator(interpolates)
    
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates, device=device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def __save_checkpoint(
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


def __load_checkpoint(
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


def __get_latest_checkpoint(
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
    input_dim,
    num_epochs: int = 1000,
    n_critic: int = 5,
    device: str = "cpu",
    checkpoint_dir: str = "checkpoints",
    lr_g: float = 1e-4,
    lr_d: float = 1e-4,
    lambda_gp: float = 10,
) -> Tuple[list, list]:
    """Train the WGAN-GP model."""
    
    d_losses = []
    g_losses = []
    
    try:
        optimizer_G = optim.Adam(G.parameters(), lr=lr_g, betas=(0.0, 0.9))
        optimizer_D = optim.Adam(D.parameters(), lr=lr_d, betas=(0.0, 0.9))
        scaler_G = amp.GradScaler(enabled=device=="cuda")
        scaler_D = amp.GradScaler(enabled=device=="cuda")

        start_epoch = 0
        latest_checkpoint = __get_latest_checkpoint(checkpoint_dir, num_epochs)
        if latest_checkpoint:
            start_epoch = __load_checkpoint(G, D, optimizer_G, optimizer_D, latest_checkpoint)
            if start_epoch >= num_epochs:
                return [], []
                
        save_interval = max(1, (num_epochs - start_epoch) // 4)
        epoch_times = []

        for epoch in range(start_epoch, num_epochs):
            epoch_start_time = time.time()
            
            for batch_idx, (real_data, _) in enumerate(train_loader):
                real_data = real_data.to(device, non_blocking=True)
                batch_size = real_data.size(0)
                
                for _ in range(n_critic):
                    optimizer_D.zero_grad(set_to_none=True)
                    
                    with amp.autocast(device_type="cuda", enabled=device=="cuda"):
                        noise = torch.randn(batch_size, input_dim, device=device)
                        fake_data = G(noise)
                        
                        real_validity = D(real_data)
                        fake_validity = D(fake_data.detach())
                        
                        gradient_penalty = compute_gradient_penalty(D, real_data, fake_data, device)
                        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty

                    scaler_D.scale(d_loss).backward()
                    scaler_D.step(optimizer_D)
                    scaler_D.update()

                # Train Generator
                optimizer_G.zero_grad(set_to_none=True)
                
                with amp.autocast(device_type="cuda", enabled=device=="cuda"):
                    fake_data = G(noise)
                    fake_validity = D(fake_data)
                    g_loss = -torch.mean(fake_validity)

                scaler_G.scale(g_loss).backward()
                scaler_G.step(optimizer_G)
                scaler_G.update()

                if batch_idx % 5 == 0:
                    torch.cuda.empty_cache()

            d_losses.append(d_loss.item())
            g_losses.append(g_loss.item())

            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            epoch_times.append(epoch_duration)
            
            if len(epoch_times) > MOVING_AVERAGE_WINDOW:
                epoch_times.pop(0)
            avg_epoch_time = sum(epoch_times) / len(epoch_times)
            remaining_epochs = num_epochs - (epoch + 1)
            remaining_time = remaining_epochs * avg_epoch_time

            logging.info(
                f"\nEpoch [{epoch + 1}/{num_epochs}]\n"
                f"D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}\n"
                f"Time for epoch: {epoch_duration:.2f}s | Est. remaining: {remaining_time:.2f}s\n"
                f"{DELIMITER}"
            )

            if (epoch + 1) % save_interval == 0 or (epoch + 1) == num_epochs:
                __save_checkpoint(G, D, optimizer_G, optimizer_D, epoch + 1, checkpoint_dir)

    finally:
        torch.cuda.empty_cache()
        return d_losses, g_losses


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
