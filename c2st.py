import numpy as np
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from typing import Tuple
import logging

class C2STClassifier(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.network(x)

def calculate_pvalue_permutation(
    predicted_probs: np.ndarray,
    true_labels: np.ndarray,
    n_permutations: int = 1000,
    metric: str = 'accuracy'
) -> float:
    """
    Calcula o p-valor usando teste de permutação.

    Args:
        predicted_probs (np.ndarray): Probabilidades preditas pelo modelo
        true_labels (np.ndarray): Labels verdadeiros
        n_permutations (int): Número de permutações para o teste
        metric (str): Métrica de avaliação ('accuracy' ou 'auc')

    Returns:
        float: P-valor calculado
    """
    if metric == 'accuracy':
        original_stat = np.mean((predicted_probs > 0.5) == true_labels)
    elif metric == 'auc':
        original_stat = roc_auc_score(true_labels, predicted_probs)
    else:
        raise ValueError("Métrica não suportada. Use 'accuracy' ou 'auc'.")

    perm_stats = np.zeros(n_permutations)
    
    for i in range(n_permutations):
        perm_labels = np.random.permutation(true_labels)
        if metric == 'accuracy':
            perm_stats[i] = np.mean((predicted_probs > 0.5) == perm_labels)
        elif metric == 'auc':
            perm_stats[i] = roc_auc_score(perm_labels, predicted_probs)
    
    p_value = np.mean(perm_stats >= original_stat)
    
    return p_value

def perform_c2st(
    real_data: np.ndarray,
    generated_data: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
    num_epochs: int = 100,
    metric: str = 'accuracy'
) -> Tuple[float, float]:
    """
    Realiza o Classifier Two-Sample Test (C2ST) nos dados reais e gerados.

    Args:
        real_data (np.ndarray): Dados reais
        generated_data (np.ndarray): Dados gerados
        device (torch.device): Dispositivo para processamento
        batch_size (int, optional): Tamanho do batch. Defaults to 64.
        num_epochs (int, optional): Número de épocas. Defaults to 100.
        metric (str, optional): Métrica de avaliação ('accuracy' ou 'auc'). Defaults to 'accuracy'.

    Returns:
        Tuple[float, float]: Acurácia do teste e p-valor
    """
    num_samples = min(len(real_data), len(generated_data))
    real_data = real_data[:num_samples]
    generated_data = generated_data[:num_samples]
    
    X = np.vstack([real_data, generated_data])
    y = np.hstack([np.ones(len(real_data)), np.zeros(len(generated_data))])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train = torch.FloatTensor(X_train).to(device)
    y_train = torch.FloatTensor(y_train).to(device)
    X_test = torch.FloatTensor(X_test).to(device)
    y_test = torch.FloatTensor(y_test).to(device)

    train_dataset = TensorDataset(X_train, y_train.reshape(-1, 1))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = C2STClassifier(X.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for _ in range(num_epochs):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)
        predicted_probs = y_pred.cpu().numpy().flatten()
        true_labels = y_test.cpu().numpy().flatten()
        
        if metric == 'accuracy':
            accuracy = np.mean((predicted_probs > 0.5) == true_labels) * 100
        elif metric == 'auc':
            accuracy = roc_auc_score(true_labels, predicted_probs) * 100
        else:
            raise ValueError("Métrica não suportada. Use 'accuracy' ou 'auc'.")

        p_value = calculate_pvalue_permutation(predicted_probs, true_labels, metric=metric)

    return accuracy, p_value