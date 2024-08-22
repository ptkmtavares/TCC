import numpy as np
import os
import time
from sklearn.neural_network import MLPClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, TensorDataset
import dataExtractor

# Verificar se a GPU está disponível
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ativar o benchmark do cuDNN para melhorar a performance
torch.backends.cudnn.benchmark = True

# Função para salvar o estado do modelo
def save_checkpoint(epoch, model_G, model_D, optimizer_G, optimizer_D, loss_G, loss_D, filename='checkpoint.pth.tar'):
    state = {
        'epoch': epoch,
        'model_G_state_dict': model_G.state_dict(),
        'model_D_state_dict': model_D.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict(),
        'loss_G': loss_G,
        'loss_D': loss_D
    }
    torch.save(state, filename)

# Função para carregar o estado do modelo
def load_checkpoint(filename, model_G, model_D, optimizer_G, optimizer_D):
    if os.path.isfile(filename):
        print(f"Loading checkpoint '{filename}'")
        checkpoint = torch.load(filename, weights_only=True)
        epoch = checkpoint['epoch']
        model_G.load_state_dict(checkpoint['model_G_state_dict'])
        model_D.load_state_dict(checkpoint['model_D_state_dict'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        loss_G = checkpoint['loss_G']
        loss_D = checkpoint['loss_D']
        print(f"Loaded checkpoint '{filename}' (epoch {epoch})")
        return epoch, loss_G, loss_D
    else:
        print(f"No checkpoint found at '{filename}'")
        return 0, None, None

# Definir a arquitetura do gerador
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

# Definir a arquitetura do discriminador
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.model(x)
    
# Definir a MLP usando PyTorch
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

if __name__ == '__main__':
    # Configurar o método de início para multiprocessing
    torch.multiprocessing.set_start_method('spawn')

    # Carregar e pré-processar os dados
    selected_data = ['ham', 'spam', 'phishing']
    data, index = dataExtractor.getTrainingTestSet('Dataset/index', selected_data, 1.0)

    # Converter listas para arrays NumPy
    data = np.array(data)
    index = np.array(index)

    # Dividir os dados em conjuntos de treinamento e teste
    train_set, test_set, train_labels, test_labels = train_test_split(data, index, train_size=0.75, random_state=9, shuffle=True)

    # Normalizar os dados usando Min-Max Normalization
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_set_normalized = scaler.fit_transform(train_set)
    test_set_normalized = scaler.transform(test_set)

    # Mover os dados normalizados para a GPU
    train_set_normalized = torch.tensor(train_set_normalized, dtype=torch.float32).to(device)
    train_labels = torch.tensor(train_labels, dtype=torch.float32).to(device)

    # Criar DataLoader para conjuntos de treinamento
    train_dataset = TensorDataset(train_set_normalized, train_labels)

    # Inicializar o gerador e o discriminador
    input_dim = train_set.shape[1]
    G = Generator(input_dim, input_dim).to(device)
    D = Discriminator(input_dim).to(device)

    # Ajustar o tamanho do batch com base na memória disponível
    batch_size = 256
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Definir funções de perda e otimizadores
    criterion = nn.BCEWithLogitsLoss()
    optimizer_G = optim.Adam(G.parameters(), lr=0.0002)
    optimizer_D = optim.Adam(D.parameters(), lr=0.0001)

    # Carregar checkpoint se existir
    start_epoch, prev_loss_G, prev_loss_D = load_checkpoint('checkpoint.pth.tar', G, D, optimizer_G, optimizer_D)

    # Treinar a GANa
    num_epochs = 1000
    n_critic = 1  # Atualizar o discriminador uma vez por cada atualização do gerador
    epoch_times = []

    # Usar Mixed Precision Training
    grad_scaler = torch.amp.GradScaler("cuda")

    # Treinamento da GAN
    for epoch in range(start_epoch, num_epochs):
        start_time = time.time()
        
        for real_data, _ in train_loader:
            real_data = real_data.to(device)
            real_labels = torch.ones(real_data.size(0), 1).to(device) * 0.9
            fake_labels = torch.zeros(real_data.size(0), 1).to(device) + 0.1

            # Atualizar o discriminador
            for _ in range(n_critic):
                noise = torch.randn(real_data.size(0), input_dim).to(device)
                fake_data = G(noise)
                optimizer_D.zero_grad()
                with torch.amp.autocast("cuda"):  # Usar Mixed Precision Training
                    real_loss = criterion(D(real_data), real_labels)
                    fake_loss = criterion(D(fake_data.detach()), fake_labels)
                    d_loss = real_loss + fake_loss
                grad_scaler.scale(d_loss).backward()
                grad_scaler.step(optimizer_D)
                grad_scaler.update()

            # Atualizar o gerador
            noise = torch.randn(real_data.size(0), input_dim).to(device)
            fake_data = G(noise)
            optimizer_G.zero_grad()
            with torch.amp.autocast("cuda"):  # Usar Mixed Precision Training
                g_loss = criterion(D(fake_data), real_labels)
            grad_scaler.scale(g_loss).backward()
            grad_scaler.step(optimizer_G)
            grad_scaler.update()

        end_time = time.time()
        epoch_time = end_time - start_time
        epoch_times.append(epoch_time)
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        remaining_epochs = num_epochs - (epoch + 1)
        estimated_time_remaining = remaining_epochs * avg_epoch_time

        print(f'Epoch [{epoch}/{num_epochs}] | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f} | Time: {epoch_time:.2f}s | Estimated Time Remaining: {estimated_time_remaining:.2f}s')

        # Salvar checkpoint
        if (epoch + 1) % 250 == 0:
            save_checkpoint(epoch, G, D, optimizer_G, optimizer_D, g_loss.item(), d_loss.item())

    # Gerar exemplos adversariais (da metade do tamanho do conjunto de treinamento)
    num_samples = len(train_set) // 2
    print(f'Generating {num_samples} adversarial examples...')
    noise = torch.randn(num_samples, input_dim).to(device)
    generated_data = G(noise).cpu().detach().numpy()

    # Adicionar exemplos adversariais ao conjunto de treinamento
    #augmented_train_set = np.vstack([train_set, generated_data])

    # Ajustar os rótulos dos exemplos adversariais
    # Assumindo que os exemplos gerados são falsos e queremos rotulá-los como 'spam' (2)
    #augmented_train_labels = np.hstack([train_labels.cpu().numpy(), np.full(num_samples, 1)])
    
    augmented_train_set = train_set
    augmented_train_labels = train_labels

    # Redefinir o scaler como MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))

    # Normalizar o conjunto de dados aumentado
    augmented_train_set_normalized = scaler.fit_transform(augmented_train_set)
    test_set_normalized = scaler.transform(test_set)

    # Parâmetros do modelo
    input_dim = train_set_normalized.shape[1]
    hidden_dim = 40
    output_dim = 3  # Número de classes (ham, spam, phishing)

    # Inicializar o modelo, critério de perda e otimizador
    model = MLP(input_dim, hidden_dim, output_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0005)  # weight_decay para regularização L2

    # Converter os dados para tensores do PyTorch e mover para a GPU
    X_train = train_set_normalized.float().clone().detach().to(device)
    y_train = train_labels.long().clone().detach().to(device)
    X_test = torch.from_numpy(test_set_normalized).float().clone().detach().to(device)
    y_test = torch.from_numpy(test_labels).long().clone().detach().to(device)

    # Treinar o modelo
    num_epochs = 10000
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 500 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Avaliar o modelo
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == y_test).sum().item() / y_test.size(0) * 100
        print(f'Accuracy for MLP classifier with adversarial examples(%)={accuracy:.2f}')

        cm = confusion_matrix(y_test.cpu(), predicted.cpu())
        print('Confusion matrix for MLP classifier with adversarial examples:\n', cm)