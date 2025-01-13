import torch
import torch.amp as amp
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import logging
from typing import Tuple

DELIMETER = '='*75
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim1: int, hidden_dim2: int, output_dim: int, l1_lambda: float = 0.0002, l2_lambda: float = 0.0005, dropout: float = 0.3):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=dropout)
        self.fc3 = nn.Linear(hidden_dim2, output_dim)
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        out = self.fc3(out)
        return out

    def l1_penalty(self) -> torch.Tensor:
        l1_norm = sum(p.abs().sum() for p in self.parameters())
        return self.l1_lambda * l1_norm

    def l2_penalty(self) -> torch.Tensor:
        l2_norm = sum(p.pow(2.0).sum() for p in self.parameters())
        return self.l2_lambda * l2_norm

def predict_mlp(model: MLP, X_test: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    try:
        model.eval()
        with torch.no_grad(), amp.autocast('cuda'):
            outputs = model(X_test)
            probabilities = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            return predicted, probabilities
    finally:
        torch.cuda.empty_cache()

def train_mlp(model: MLP, criterion: nn.Module, optimizer: torch.optim.Optimizer, X_train: torch.Tensor, y_train: torch.Tensor, X_val: torch.Tensor, y_val: torch.Tensor, num_epochs: int = 10000, patience: int = 15, printInfo: bool = True) -> float:
    scaler = amp.GradScaler('cuda')
    best_val_loss = float('inf')
    patience_counter = 0

    try:
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            
            with amp.autocast('cuda'):
                outputs = model(X_train)
                loss = criterion(outputs, y_train)
                l1_penalty = model.l1_penalty()
                l2_penalty = model.l2_penalty()
                loss += l1_penalty + l2_penalty
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            model.eval()
            with torch.no_grad(), amp.autocast('cuda'):
                val_outputs = model(X_val)
                val_loss = criterion(val_outputs, y_val)
                val_loss += model.l1_penalty() + model.l2_penalty()
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                if printInfo: logging.info(f"Early stopping at epoch {epoch + 1}")
                break

            if (epoch + 1) % (num_epochs // 4) == 0:
                if printInfo: logging.info(
                    f"\nEpoch [{epoch + 1}/{num_epochs}]\n"
                    f"Loss: {loss.item():.4f}\n"
                    f"Validation Loss: {val_loss.item():.4f}\n"
                    f"{DELIMETER}"
                )
                torch.cuda.empty_cache()

        return best_val_loss.item()
    finally:
        torch.cuda.empty_cache()

def evaluate_mlp(model: MLP, X_test: torch.Tensor, y_test: torch.Tensor, printInfo: bool = True) -> float:
    try:
        model.eval()
        with torch.no_grad(), amp.autocast('cuda'):
            outputs = model(X_test)
            _, predicted = torch.max(outputs.data, 1)
            
            accuracy = (predicted == y_test).sum().item() / y_test.size(0) * 100
            
            cm = confusion_matrix(y_test.cpu(), predicted.cpu())
            if printInfo: logging.info(
                f"\nConfusion matrix for MLP classifier:\n"
                f"{cm}\n"
                f"{DELIMETER}"
            )
            return accuracy
    finally:
        torch.cuda.empty_cache()