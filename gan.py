import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import time
import os
import logging
from typing import Tuple
from config import DELIMITER, LOG_FORMAT
from c2st import perform_c2st

MOVING_AVERAGE_WINDOW = 100
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim=61, num_classes=3):
        super(Generator, self).__init__()
        self.output_dim = output_dim
        self.num_classes = num_classes

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, output_dim * num_classes),
        )

    def forward(self, z, temperature=1.0, hard=False):
        batch_size = z.size(0)
        logits = self.model(z)
        logits = logits.view(batch_size, self.output_dim, self.num_classes)
        return nn.functional.gumbel_softmax(logits, tau=temperature, hard=hard, dim=-1)


class Discriminator(nn.Module):
    def __init__(self, input_dim=61 * 3):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.model(x).view(-1)


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


def temperature_scheduling(epoch, num_epochs, initial_temp=1.0, min_temp=0.1):
    progress = epoch / num_epochs
    return min_temp + 0.5 * (initial_temp - min_temp) * (1 + np.cos(progress * np.pi))


def compute_gradient_penalty(D, real_samples, fake_samples, device):
    """Calcula o gradient penalty para WGAN-GP"""
    alpha = torch.rand((real_samples.size(0), 1), device=device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(
        True
    )
    d_interpolates = D(interpolates)

    fake = torch.ones(d_interpolates.size(), device=device, requires_grad=False)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


# Deus nos abandonou aqui
def train_gan(
    G: Generator,
    D: Discriminator,
    train_loader: torch.utils.data.DataLoader,
    input_dim: int,
    num_epochs: int = 1000,
    device: str = "cpu",
    checkpoint_dir: str = "checkpoints",
    n_critic: int = 5,
    lambda_gp: float = 10.0,
    lr_g: float = 1e-4,
    lr_d: float = 1e-4,
    initial_temperature: float = 1.0,
    min_temperature: float = 0.1,
) -> Tuple[list, list]:

    best_generator_state = None
    best_ctst_score = float("inf")
    patience = 5
    patience_counter = 0

    d_losses = []
    g_losses = []

    try:
        G = G.to(device)
        D = D.to(device)

        optimizer_G = optim.Adam(G.parameters(), lr=lr_g, betas=(0.5, 0.999))
        optimizer_D = optim.Adam(D.parameters(), lr=lr_d, betas=(0.5, 0.999))

        criterion = nn.BCEWithLogitsLoss()

        start_epoch = 0
        latest_checkpoint = __get_latest_checkpoint(checkpoint_dir, num_epochs)
        if latest_checkpoint:
            start_epoch = __load_checkpoint(
                G, D, optimizer_G, optimizer_D, latest_checkpoint
            )
            if start_epoch >= num_epochs:
                return [], []

        save_interval = max(1, (num_epochs - start_epoch) // 4)
        eval_interval = max(1, (num_epochs - start_epoch) // 50)
        epoch_times = []

        for epoch in range(start_epoch, num_epochs):
            epoch_start_time = time.time()
            epoch_g_losses = []
            epoch_d_losses = []

            G.train()
            D.train()

            # temperature = max(
            #    initial_temperature * np.exp(-3e-3 * epoch),
            #    min_temperature
            # )

            # temperature = max(min_temperature, initial_temperature - (initial_temperature - min_temperature) * epoch/num_epochs)

            temperature = temperature_scheduling(
                epoch,
                num_epochs,
                initial_temp=initial_temperature,
                min_temp=min_temperature,
            )

            for i, real_data in enumerate(train_loader):
                real_data = real_data[0].to(device)
                batch_size = real_data.size(0)

                # Converter dados reais para one-hot
                real_labels = (real_data + 1).long()
                one_hot_real = nn.functional.one_hot(real_labels, num_classes=3).float()
                real_samples = one_hot_real.view(batch_size, -1)

                # Treinar crítico/discriminador
                for _ in range(n_critic):
                    optimizer_D.zero_grad()

                    z = torch.randn(batch_size, input_dim, device=device)
                    fake = G(z, temperature=temperature, hard=True)
                    fake_samples = fake.view(batch_size, -1)

                    real_validity = D(real_samples)
                    fake_validity = D(fake_samples.detach())

                    # Gradient penalty
                    gp = compute_gradient_penalty(
                        D, real_samples, fake_samples.detach(), device
                    )

                    # Loss Wasserstein com gradient penalty
                    d_loss = (
                        -torch.mean(real_validity)
                        + torch.mean(fake_validity)
                        + lambda_gp * gp
                    )

                    d_loss.backward()
                    optimizer_D.step()

                # Treinar gerador
                optimizer_G.zero_grad()

                fake_validity = D(fake_samples)
                g_loss = -torch.mean(fake_validity)

                g_loss.backward()
                optimizer_G.step()

                epoch_d_losses.append(d_loss.item())
                epoch_g_losses.append(g_loss.item())

            avg_d_loss = np.mean(epoch_d_losses)
            avg_g_loss = np.mean(epoch_g_losses)

            d_losses.append(avg_d_loss)
            g_losses.append(avg_g_loss)

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
                f"D Loss: {avg_d_loss:.4f} | G Loss: {avg_g_loss:.4f}\n"
                f"Time for epoch: {epoch_duration:.2f}s | Est. remaining: {remaining_time:.2f}s\n"
                f"Learning rates - G: {optimizer_G.param_groups[0]['lr']:.2e} | "
                f"D: {optimizer_D.param_groups[0]['lr']:.2e}\n"
                f"{DELIMITER}"
            )

            if (epoch + 1) % save_interval == 0 or (epoch + 1) == num_epochs:
                __save_checkpoint(
                    G, D, optimizer_G, optimizer_D, epoch + 1, checkpoint_dir
                )

            # if (epoch + 1) % eval_interval == 0 or (epoch + 1) == num_epochs:
            #    fake_data = generate_adversarial_examples(G, len(real_data), input_dim, device)
            #    ctst_score, _ = perform_c2st(real_data.cpu().numpy(), fake_data, device)
            #    logging.info(f"CTST score: {ctst_score:.4f}")
            #    if ctst_score < best_ctst_score:
            #        best_ctst_score = ctst_score
            #        best_generator_state = G.state_dict()
            #        patience_counter = 0
            #    elif ctst_score > best_ctst_score:
            #        patience_counter += 1

            #    if ctst_score < ctst_threshold or patience_counter >= patience:
            #        logging.info(f"Early stopping at epoch {epoch + 1} with CTST score {ctst_score:.4f}")
            #        break

    except Exception as e:
        logging.error(f"Erro durante o treinamento: {str(e)}", exc_info=True)
        raise
    finally:
        torch.cuda.empty_cache()

    if best_generator_state:
        G.load_state_dict(best_generator_state)

    return d_losses, g_losses


def generate_adversarial_examples(
    G: Generator,
    num_samples: int,
    input_dim: int,
    device: str = "cpu",
    temperature: float = 0.1,
) -> np.ndarray:
    G.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, input_dim, device=device)
        samples = G(z, temperature=temperature, hard=True)
        samples = samples.argmax(dim=-1) - 1
        return samples.cpu().numpy()
