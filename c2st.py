import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from typing import Tuple


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
    predicted_probs: np.ndarray, true_labels: np.ndarray, n_permutations: int = 1000
) -> float:
    """
    Calcula o p-valor usando teste de permutação.

    Args:
        predicted_probs (np.ndarray): Probabilidades preditas pelo modelo
        true_labels (np.ndarray): Labels verdadeiros
        n_permutations (int): Número de permutações para o teste

    Returns:
        float: P-valor calculado
    """

    original_stat = np.mean((predicted_probs > 0.5) == true_labels)
    perm_stats = np.zeros(n_permutations)

    for i in range(n_permutations):
        perm_labels = np.random.permutation(true_labels)
        perm_stats[i] = np.mean((predicted_probs > 0.5) == perm_labels)

    p_value = np.mean(perm_stats >= original_stat)

    return p_value


def train_test_split(
    X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Divide os dados em conjuntos de treino e teste.

    Args:
        X (np.ndarray): Dados de entrada
        y (np.ndarray): Labels
        test_size (float): Proporção do conjunto de teste
        random_state (int): Semente para reprodutibilidade

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Conjuntos de treino e teste
    """
    np.random.seed(random_state)
    indices = np.random.permutation(len(X))
    test_size = int(len(X) * test_size)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def gaussian_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """
    Calcula o kernel Gaussiano entre dois tensores.

    Args:
        x (torch.Tensor): Primeiro tensor
        y (torch.Tensor): Segundo tensor
        sigma (float): Parâmetro do kernel

    Returns:
        torch.Tensor: Kernel Gaussiano
    """
    dist = torch.cdist(x, y, p=2)
    return torch.exp(-dist ** 2 / (2 * sigma ** 2))


def calculate_mmd(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0) -> float:
    """
    Calcula a Discrepância Máxima de Média (MMD) entre dois conjuntos de dados.

    Args:
        x (torch.Tensor): Primeiro conjunto de dados
        y (torch.Tensor): Segundo conjunto de dados
        sigma (float): Parâmetro do kernel

    Returns:
        float: MMD calculado
    """
    xx = gaussian_kernel(x, x, sigma)
    yy = gaussian_kernel(y, y, sigma)
    xy = gaussian_kernel(x, y, sigma)
    mmd = xx.mean() + yy.mean() - 2 * xy.mean()
    return mmd.item()


def perform_c2st(
    real_data: np.ndarray,
    generated_data: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
    num_epochs: int = 100,
) -> Tuple[float, float, float]:
    """
    Realiza o Classifier Two-Sample Test (C2ST) nos dados reais e gerados.

    Args:
        real_data (np.ndarray): Dados reais
        generated_data (np.ndarray): Dados gerados
        device (torch.device): Dispositivo para processamento
        batch_size (int, optional): Tamanho do batch. Defaults to 64.
        num_epochs (int, optional): Número de épocas. Defaults to 100.

    Returns:
        Tuple[float, float, float]: Acurácia do teste, p-valor e MMD
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

        accuracy = np.mean((predicted_probs > 0.5) == true_labels) * 100
        p_value = calculate_pvalue_permutation(predicted_probs, true_labels)
        mmd = calculate_mmd(torch.FloatTensor(real_data).to(device), torch.FloatTensor(generated_data).to(device))

    return accuracy, p_value, mmd