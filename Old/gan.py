import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, TensorDataset
import dataExtractor

def main():
    # Verificar se a GPU está disponível
    print("CUDA Available: ", torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )

        def forward(self, x):
            return self.model(x)

    # Carregar e pré-processar os dados
    selected_data = ['ham', 'spam', 'phishing']
    data, index = dataExtractor.getTrainingTestSet('Dataset/index', selected_data, 1.0)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Dividir os dados em conjuntos de treinamento e teste
    train_set, test_set, train_labels, test_labels = train_test_split(data_scaled, index, train_size=0.75, random_state=9, shuffle=True)

    # Criar DataLoader para conjuntos de treinamento
    train_dataset = TensorDataset(torch.tensor(train_set, dtype=torch.float32), torch.tensor(train_labels, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)

    # Inicializar o gerador e o discriminador
    input_dim = train_set.shape[1]
    G = Generator(input_dim, input_dim).to(device)
    D = Discriminator(input_dim).to(device)

    # Definir funções de perda e otimizadores
    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(G.parameters(), lr=0.0002)
    optimizer_D = optim.Adam(D.parameters(), lr=0.0002)

    # Treinar a GAN
    num_epochs = 1000

    for epoch in range(num_epochs):
        for real_data, _ in train_loader:
            real_data = real_data.to(device)
            real_labels = torch.ones(real_data.size(0), 1).to(device)

            # Gerar dados falsos e rótulos
            noise = torch.randn(real_data.size(0), input_dim).to(device)
            fake_data = G(noise)
            fake_labels = torch.zeros(fake_data.size(0), 1).to(device)

            # Treinar o discriminador
            optimizer_D.zero_grad()
            real_loss = criterion(D(real_data), real_labels)
            fake_loss = criterion(D(fake_data.detach()), fake_labels)
            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizer_D.step()

            # Treinar o gerador
            optimizer_G.zero_grad()
            g_loss = criterion(D(fake_data), real_labels)
            g_loss.backward()
            optimizer_G.step()

        if epoch % 100 == 0:
            print(f'Epoch [{epoch}/{num_epochs}] | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}')

    # Avaliar a performance
    with torch.no_grad():
        test_data = torch.tensor(test_set, dtype=torch.float32).to(device)
        test_labels_tensor = torch.tensor(test_labels, dtype=torch.float32).view(-1, 1).to(device)
        predictions = D(test_data)
        predicted_labels = (predictions > 0.5).float()

    accuracy = accuracy_score(test_labels_tensor.cpu(), predicted_labels.cpu())
    precision = precision_score(test_labels_tensor.cpu(), predicted_labels.cpu(), average='weighted')
    recall = recall_score(test_labels_tensor.cpu(), predicted_labels.cpu(), average='weighted')
    f1 = f1_score(test_labels_tensor.cpu(), predicted_labels.cpu(), average='weighted')

    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')

if __name__ == '__main__':
    main()