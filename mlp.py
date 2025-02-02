import numpy as np
import torch
import torch.amp as amp
import torch.nn as nn
import torch.nn.functional as F
import logging
import os
import time
from torch.utils.data import DataLoader
from typing import Tuple, Dict, Optional, Any
from pathlib import Path
from config import DELIMITER, CHECKPOINT_DIR, DEVICE, MOVING_AVERAGE_WINDOW


class ResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.3):
        super(ResidualBlock, self).__init__()
        layers = []
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(in_dim, out_dim))
        layers.append(nn.Dropout(p=dropout))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(out_dim, out_dim))
        layers.append(nn.Dropout(p=dropout))
        self.main = nn.Sequential(*layers)
        self.skip = (
            nn.Linear(in_dim, out_dim, bias=False)
            if in_dim != out_dim
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x) + self.skip(x)


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        output_dim: int = 2,
        l1_lambda: float = 0.0002,
        l2_lambda: float = 0.0005,
        dropout: float = 0.3,
    ) -> None:
        super(MLP, self).__init__()
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.res_block1 = ResidualBlock(hidden_dim, hidden_dim, dropout=dropout)
        self.res_block2 = ResidualBlock(hidden_dim, hidden_dim, dropout=dropout)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc_in(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        return self.fc_out(x)

    def l1_penalty(self) -> torch.Tensor:
        l1_norm = sum(p.abs().sum() for p in self.parameters())
        return self.l1_lambda * l1_norm

    def l2_penalty(self) -> torch.Tensor:
        l2_norm = sum(p.pow(2.0).sum() for p in self.parameters())
        return self.l2_lambda * l2_norm


def __save_checkpoint(
    model: MLP,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_loss: float,
    train_losses: list,
    val_losses: list,
    patience_counter: int,
    checkpoint_dir: str,
    original_model: bool,
    metrics: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Salva o checkpoint do modelo MLP.

    Args:
        model: Modelo MLP
        optimizer: Otimizador
        epoch: Época atual
        best_val_loss: Melhor loss de validação
        train_losses: Histórico de losses de treino
        val_losses: Histórico de losses de validação
        patience_counter: Contador de paciência
        checkpoint_dir: Diretório para salvar
        metrics: Métricas adicionais (opcional)
    """
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "patience_counter": patience_counter,
        "metrics": metrics,
    }

    save_path = Path(checkpoint_dir) / f"mlp_{'original' if original_model else 'augmented'}_checkpoint_epoch_{epoch}.pth"
    torch.save(checkpoint, save_path)


def __load_checkpoint(
    model: MLP, optimizer: torch.optim.Optimizer, checkpoint_path: str
) -> tuple[int, float, list, list, int, Optional[Dict[str, Any]]]:
    """
    Carrega um checkpoint do modelo MLP.

    Returns:
        tuple: (época, melhor_val_loss, train_losses, val_losses, patience_counter, métricas)
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return (
        checkpoint["epoch"],
        checkpoint["best_val_loss"],
        checkpoint.get("train_losses", []),
        checkpoint.get("val_losses", []),
        checkpoint.get("patience_counter", 0),
        checkpoint.get("metrics", None),
    )

def __get_latest_checkpoint(checkpoint_dir: str, num_epochs: int, original_model: bool) -> Optional[str]:
    """
    Encontra o checkpoint mais recente.

    Args:
        checkpoint_dir: Diretório dos checkpoints
        num_epochs: Número total de épocas

    Returns:
        Optional[str]: Caminho do checkpoint mais recente ou None
    """
    if not os.path.exists(checkpoint_dir):
        return None

    checkpoints = [
        f for f in os.listdir(checkpoint_dir) if f.startswith(f"mlp_{'original' if original_model else 'augmented'}_checkpoint_epoch_")
    ]

    if not checkpoints:
        return None

    valid_checkpoints = [
        f for f in checkpoints if int(f.split("_")[-1].split(".")[0]) <= num_epochs
    ]

    if not valid_checkpoints:
        return None

    latest = max(valid_checkpoints, key=lambda x: int(x.split("_")[-1].split(".")[0]))

    return os.path.join(checkpoint_dir, latest)


def predict_mlp(model: MLP, X_test: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Make predictions using the MLP model.

    Args:
        model (MLP): Trained MLP model.
        X_test (torch.Tensor): Test data.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Predicted labels and probabilities.
    """
    try:
        model.eval()
        with torch.no_grad(), amp.autocast("cuda"):
            outputs = model(X_test)
            probabilities = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            return predicted, probabilities
    finally:
        torch.cuda.empty_cache()


def train_mlp(
    model: MLP,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 1000,
    patience: int = 15,
    printInfo: bool = True,
    checkpoint_dir: str = CHECKPOINT_DIR,
    original_model: bool = True,
) -> Tuple[float, list, list]:

    scaler = amp.GradScaler()
    start_epoch = 0
    epoch_times = []
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    patience_counter = 0

    latest_checkpoint = __get_latest_checkpoint(checkpoint_dir, num_epochs, original_model)
    if latest_checkpoint:
        start_epoch, best_val_loss, train_losses, val_losses, patience_counter, _ = __load_checkpoint(
            model, optimizer, latest_checkpoint
        )
        if start_epoch >= num_epochs or patience_counter >= patience:
            return best_val_loss, train_losses, val_losses

    save_interval = max(1, num_epochs // 4)

    try:
        for epoch in range(start_epoch, num_epochs):
            if patience_counter >= patience:
                if printInfo:
                    logging.info(f"Early stopping at epoch [{epoch}/{num_epochs}]")
                __save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch + 1,
                    best_val_loss=best_val_loss,
                    train_losses=train_losses,
                    val_losses=val_losses,
                    patience_counter=patience_counter,
                    checkpoint_dir=checkpoint_dir,
                    original_model=original_model,
                    metrics={
                        "patience_counter": patience_counter,
                        "last_train_loss": train_losses[-1] if train_losses else None,
                        "last_val_loss": val_losses[-1] if val_losses else None,
                    },
                )
                break
            
            epoch_start_time = time.time()
            model.train()
            epoch_loss = 0.0

            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
                optimizer.zero_grad(set_to_none=True)

                with amp.autocast("cuda"):
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss += model.l1_penalty() + model.l2_penalty()

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.item()

            avg_train_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
                    with amp.autocast("cuda"):
                        val_outputs = model(batch_X)
                        batch_val_loss = criterion(val_outputs, batch_y)
                        batch_val_loss += model.l1_penalty() + model.l2_penalty()
                    val_loss += batch_val_loss.item()

            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % save_interval == 0 or (epoch + 1) == num_epochs:
                __save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch + 1,
                    best_val_loss=best_val_loss,
                    train_losses=train_losses,
                    val_losses=val_losses,
                    patience_counter=patience_counter,
                    checkpoint_dir=checkpoint_dir,
                    original_model=original_model,
                    metrics={
                        "patience_counter": patience_counter,
                        "last_train_loss": avg_train_loss,
                        "last_val_loss": avg_val_loss,
                    },
                )

            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            epoch_times.append(epoch_duration)

            if len(epoch_times) > MOVING_AVERAGE_WINDOW:
                epoch_times.pop(0)

            avg_epoch_time = sum(epoch_times) / len(epoch_times)
            remaining_epochs = num_epochs - (epoch + 1)
            remaining_time = remaining_epochs * avg_epoch_time

            # if printInfo and (epoch + 1) % (num_epochs // 4) == 0:
            if printInfo:
                logging.info(
                    f"\nEpoch [{epoch + 1}/{num_epochs}]\n"
                    f"Train Loss: {avg_train_loss:.4f}\n"
                    f"Val Loss: {avg_val_loss:.4f}\n"
                    f"Best Val Loss: {best_val_loss:.4f}\n"
                    f"Patience Counter: {patience_counter}/{patience}\n"
                    f"Time for epoch: {epoch_duration:.2f}s | Est. remaining: {remaining_time:.2f}s\n"
                    f"Learning rate: {optimizer.param_groups[0]['lr']:.2e}\n"
                    f"{DELIMITER}"
                )

        return best_val_loss, train_losses, val_losses
    finally:
        torch.cuda.empty_cache()


def evaluate_mlp(
    model: MLP, test_loader: DataLoader, num_classes: int = 2
) -> Tuple[float, np.ndarray, dict]:
    """
    Avalia o modelo MLP usando DataLoader.

    Args:
        model (MLP): Modelo MLP treinado
        test_loader (DataLoader): Loader com dados de teste
        num_classes (int): Número de classes

    Returns:
        Tuple[float, np.ndarray, dict]:
            - Acurácia do modelo
            - Matriz de confusão
            - Dicionário com precision, recall e f1-score por classe
    """
    try:
        model.eval()
        all_predictions = []
        all_targets = []
        total_samples = 0
        correct_predictions = 0

        with torch.no_grad(), amp.autocast("cuda"):
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
                outputs = model(batch_X)
                _, predicted = torch.max(outputs.data, 1)

                total_samples += batch_y.size(0)
                correct_predictions += (predicted == batch_y).sum().item()

                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())

        accuracy = (correct_predictions / total_samples) * 100

        cm = np.zeros((num_classes, num_classes), dtype=int)
        for t, p in zip(all_targets, all_predictions):
            cm[t, p] += 1

        metrics = {}
        for i in range(num_classes):
            tp = cm[i, i]
            fp = np.sum(cm[:, i]) - tp
            fn = np.sum(cm[i, :]) - tp
            tn = np.sum(cm) - (tp + fp + fn)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0
            )

            metrics[f"class_{i}"] = {
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
            }

        total_samples = np.sum(cm, axis=1)
        weighted_precision = sum(
            m["precision"] * count for m, count in zip(metrics.values(), total_samples)
        ) / sum(total_samples)
        weighted_recall = sum(
            m["recall"] * count for m, count in zip(metrics.values(), total_samples)
        ) / sum(total_samples)
        weighted_f1 = sum(
            m["f1_score"] * count for m, count in zip(metrics.values(), total_samples)
        ) / sum(total_samples)

        metrics["weighted_avg"] = {
            "precision": weighted_precision,
            "recall": weighted_recall,
            "f1_score": weighted_f1,
        }

        return accuracy, cm, metrics
    finally:
        torch.cuda.empty_cache()
