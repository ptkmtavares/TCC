import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import time
import os
import logging
from typing import Tuple
from config import (
    CHECKPOINT_DIR,
    DECAY_FACTOR,
    DELIMITER,
    DEVICE,
    FEATURES,
    GEN_TEMPERATURE,
    INIT_TEMPERATURE,
    LAMBDA_GP,
    LATENT_DIM,
    LOG_FORMAT,
    LR_GAN,
    MIN_TEMPERATURE,
    N_CRITIC,
    NUM_EPOCHS_GAN,
    WARMUP_EPOCHS,
    MOVING_AVERAGE_WINDOW,
)

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, use_ln=False):
        super(ResidualBlock, self).__init__()
        layers = []
        if use_ln:
            layers.append(nn.LayerNorm(in_dim))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(in_dim, out_dim))
        if use_ln:
            layers.append(nn.LayerNorm(out_dim))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(out_dim, out_dim))
        self.main = nn.Sequential(*layers)
        self.skip = nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else nn.Identity()
    
    def forward(self, x):
        return self.main(x) + self.skip(x)

class Generator(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, output_dim=len(FEATURES), num_classes=3):
        super(Generator, self).__init__()
        self.output_dim = output_dim
        self.num_classes = num_classes
        self.fc_in = nn.Linear(latent_dim, latent_dim)
        
        self.res_block1 = ResidualBlock(latent_dim, latent_dim, use_ln=True)
        self.res_block2 = ResidualBlock(latent_dim, latent_dim, use_ln=True)
        
        self.fc_out = nn.Linear(latent_dim, output_dim * num_classes)

    def forward(self, z, temperature=INIT_TEMPERATURE, hard=False):
        x = self.fc_in(z)
        x = self.res_block1(x)
        x = self.res_block2(x)
        logits = self.fc_out(x).view(-1, self.output_dim, self.num_classes)
        return nn.functional.gumbel_softmax(logits, tau=temperature, hard=hard, dim=-1)

class Discriminator(nn.Module):
    def __init__(self, input_dim=len(FEATURES)*3):
        super(Discriminator, self).__init__()
        hidden_dim = LATENT_DIM // 2
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.res_block1 = ResidualBlock(hidden_dim, hidden_dim, use_ln=False)
        self.res_block2 = ResidualBlock(hidden_dim, hidden_dim, use_ln=False)
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.fc_in(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        return self.fc_out(x).view(-1)


def __save_checkpoint(
    G: Generator,
    D: Discriminator,
    optimizer_G: torch.optim.Optimizer,
    optimizer_D: torch.optim.Optimizer,
    epoch: int,
    best_generator_state: dict = None,
    best_ctst_score: float = float("inf"),
    d_losses: list = None,
    g_losses: list = None,
    checkpoint_dir: str = CHECKPOINT_DIR,
) -> None:
    """
    Salva o checkpoint com estados dos modelos e métricas importantes.

    Args:
        G (Generator): Modelo gerador
        D (Discriminator): Modelo discriminador
        optimizer_G (Optimizer): Otimizador do gerador
        optimizer_D (Optimizer): Otimizador do discriminador
        epoch (int): Época atual
        best_generator_state (dict, optional): Melhor estado do gerador
        best_ctst_score (float, optional): Melhor pontuação C2ST
        d_losses (list, optional): Histórico de perdas do discriminador
        g_losses (list, optional): Histórico de perdas do gerador
        checkpoint_dir (str, optional): Diretório para salvar
    """
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    checkpoint = {
        "epoch": epoch,
        "G_state_dict": G.state_dict(),
        "D_state_dict": D.state_dict(),
        "optimizer_G_state_dict": optimizer_G.state_dict(),
        "optimizer_D_state_dict": optimizer_D.state_dict(),
        "best_generator_state": best_generator_state,
        "best_ctst_score": best_ctst_score,
        "d_losses": d_losses,
        "g_losses": g_losses
    }
    
    torch.save(
        checkpoint,
        os.path.join(checkpoint_dir, f"gan_checkpoint_epoch_{epoch}.pth"),
    )


def __load_checkpoint(
    G: Generator,
    D: Discriminator,
    optimizer_G: torch.optim.Optimizer,
    optimizer_D: torch.optim.Optimizer,
    checkpoint_path: str = CHECKPOINT_DIR,
) -> Tuple[int, dict, float, list, list]:
    """
    Carrega o checkpoint com todos os estados e métricas.

    Returns:
        Tuple[int, dict, float, list, list]: 
            - Época
            - Melhor estado do gerador
            - Melhor pontuação C2ST
            - Histórico de perdas do discriminador
            - Histórico de perdas do gerador
    """
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    G.load_state_dict(checkpoint["G_state_dict"])
    D.load_state_dict(checkpoint["D_state_dict"])
    optimizer_G.load_state_dict(checkpoint["optimizer_G_state_dict"])
    optimizer_D.load_state_dict(checkpoint["optimizer_D_state_dict"])
    
    return (
        checkpoint["epoch"],
        checkpoint.get("best_generator_state", None),
        checkpoint.get("best_ctst_score", float("inf")),
        checkpoint.get("d_losses", []),
        checkpoint.get("g_losses", [])
    )


def __get_latest_checkpoint(
    checkpoint_dir: str = CHECKPOINT_DIR, num_epochs: int = NUM_EPOCHS_GAN
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
        f for f in os.listdir(checkpoint_dir) if f.startswith("gan_checkpoint_epoch_")
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


def temperature_scheduling(
    epoch,
    num_epochs=NUM_EPOCHS_GAN,
    initial_temp=INIT_TEMPERATURE,
    min_temp=MIN_TEMPERATURE,
    decay_factor=DECAY_FACTOR,
    warmup_epochs=WARMUP_EPOCHS
):
    if epoch < warmup_epochs:
        return initial_temp

    progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
    cosine_decay = 0.5 * (1 + np.cos(progress * np.pi))
    
    temp = initial_temp * (decay_factor ** (epoch - warmup_epochs))
    temp = max(temp * cosine_decay, min_temp)
    
    return temp


def compute_gradient_penalty(D, real_samples, fake_samples, device=DEVICE):
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
    device: str = DEVICE,
    checkpoint_dir: str = CHECKPOINT_DIR,
    latent_dim: int = LATENT_DIM,
    num_epochs: int = NUM_EPOCHS_GAN,
    n_critic: int = N_CRITIC,
    lambda_gp: float = LAMBDA_GP,
    lr_g: float = LR_GAN[0],
    lr_d: float = LR_GAN[1],
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

        optimizer_G = optim.Adam(G.parameters(), lr=lr_g, betas=(0.5, 0.9))
        optimizer_D = optim.Adam(D.parameters(), lr=lr_d, betas=(0.5, 0.9))

        total_iterations = num_epochs * len(train_loader)
        lambda_lr = lambda iteration: max(0.0, 1.0 - iteration / total_iterations)
        scheduler_G = optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lambda_lr)
        scheduler_D = optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=lambda_lr)
        

        start_epoch = 0
        latest_checkpoint = __get_latest_checkpoint(checkpoint_dir, num_epochs)
        if latest_checkpoint:
            start_epoch, best_generator_state, best_ctst_score, d_losses, g_losses = __load_checkpoint(
                G, D, optimizer_G, optimizer_D, latest_checkpoint
            )
            if start_epoch >= num_epochs:
                return d_losses, g_losses

        save_interval = max(1, (num_epochs - start_epoch) // 4)
        eval_interval = max(1, (num_epochs - start_epoch) // 50)
        epoch_times = []

        for epoch in range(start_epoch, num_epochs):
            epoch_start_time = time.time()
            epoch_g_losses = []
            epoch_d_losses = []

            G.train()
            D.train()

            temperature = temperature_scheduling(epoch)

            for _, real_data in enumerate(train_loader):
                real_data = real_data[0].to(device)
                batch_size = real_data.size(0)

                # Converter dados reais para one-hot
                real_labels = (real_data + 1).long()
                one_hot_real = nn.functional.one_hot(real_labels, num_classes=3).float()
                real_samples = one_hot_real.view(batch_size, -1)

                # Treinar crítico/discriminador
                for _ in range(n_critic):
                    optimizer_D.zero_grad()

                    z = torch.randn(batch_size, latent_dim, device=device)
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
                
                scheduler_G.step()
                scheduler_D.step()

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
                    G, D, optimizer_G, optimizer_D, epoch + 1,
                    best_generator_state, best_ctst_score,
                    d_losses, g_losses, checkpoint_dir
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
    latent_dim: int = LATENT_DIM,
    device: str = DEVICE,
    temperature: float = GEN_TEMPERATURE,
) -> np.ndarray:
    G.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim, device=device)
        samples = G(z, temperature=temperature, hard=True)
        samples = samples.argmax(dim=-1) - 1
        return samples.cpu().numpy()
