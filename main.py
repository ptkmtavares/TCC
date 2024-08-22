import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from dataExtractor import getTrainingTestSet
from torch.utils.data import DataLoader, TensorDataset
from mlp import MLP, train_mlp, evaluate_mlp
from gan import Generator, Discriminator, train_gan, generate_adversarial_examples

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carregar e pré-processar os dados
print(
    f"{'='*50}\n"
    f"🚀 Loading and preprocessing data...\n"
    f"{'='*50}"
)
selected_data = ['ham', 'spam', 'phishing']
data, index = getTrainingTestSet('Dataset/index', selected_data, 1.0)

data = np.array(data)
index = np.array(index)

train_set, test_set, train_labels, test_labels = train_test_split(data, index, train_size=0.75, random_state=9, shuffle=True)

scaler = MinMaxScaler(feature_range=(0, 1))
train_set_normalized = scaler.fit_transform(train_set)
test_set_normalized = scaler.transform(test_set)

train_set_normalized = torch.tensor(train_set_normalized, dtype=torch.float32).to(device)
train_labels = torch.tensor(train_labels, dtype=torch.float32).to(device)

train_dataset = TensorDataset(train_set_normalized, train_labels)
batch_size = 256
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

input_dim = train_set.shape[1]

# Treinar GAN para phishing
print(
    f"🎣 Training GAN for phishing...\n"
    f"{'='*50}"
)
phishing_data = train_set[train_labels.cpu().numpy() == 2]
phishing_labels = train_labels[train_labels.cpu().numpy() == 2]
phishing_dataset = TensorDataset(torch.tensor(phishing_data, dtype=torch.float32).to(device), phishing_labels)
phishing_loader = DataLoader(phishing_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

G_phishing = Generator(input_dim, input_dim).to(device)
D_phishing = Discriminator(input_dim).to(device)
train_gan(G_phishing, D_phishing, phishing_loader, input_dim, device=device, checkpoint_dir='checkpoints/phishing/', num_epochs=5000)

# Treinar GAN para spam
print(
    f"📧 Training GAN for spam...\n"
    f"{'='*50}"
)
spam_data = train_set[train_labels.cpu().numpy() == 1]
spam_labels = train_labels[train_labels.cpu().numpy() == 1]
spam_dataset = TensorDataset(torch.tensor(spam_data, dtype=torch.float32).to(device), spam_labels)
spam_loader = DataLoader(spam_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

G_spam = Generator(input_dim, input_dim).to(device)
D_spam = Discriminator(input_dim).to(device)
train_gan(G_spam, D_spam, spam_loader, input_dim, device=device, checkpoint_dir='checkpoints/spam/')

# Gerar exemplos adversariais
print(
    f"🔍 Generating adversarial examples...\n"
    f"{'='*50}"
)
num_samples_phishing = len(phishing_data) // 2
num_samples_spam = len(spam_data) // 2

generated_phishing = generate_adversarial_examples(G_phishing, num_samples_phishing, input_dim, device=device)
generated_spam = generate_adversarial_examples(G_spam, num_samples_spam, input_dim, device=device)

# Aumentar o conjunto de treinamento
print(
    f"📈 Augmenting the training set...\n"
    f"{'='*50}"
)
augmented_train_set = np.vstack([train_set, generated_phishing, generated_spam])
augmented_train_labels = np.hstack([train_labels.cpu().numpy(), np.full(num_samples_phishing, 2), np.full(num_samples_spam, 1)])

scaler = MinMaxScaler(feature_range=(0, 1))
augmented_train_set_normalized = scaler.fit_transform(augmented_train_set)
test_set_normalized = scaler.transform(test_set)

X_train = torch.tensor(augmented_train_set_normalized, dtype=torch.float32).to(device)
y_train = torch.tensor(augmented_train_labels, dtype=torch.long).to(device)
X_test = torch.tensor(test_set_normalized, dtype=torch.float32).to(device)
y_test = torch.tensor(test_labels, dtype=torch.long).to(device)

# Treinar e avaliar o MLP
print(
    f"🧠 Training and evaluating the MLP...\n"
    f"{'='*50}"
)
input_dim = X_train.shape[1]
hidden_dim = 40
output_dim = 3

model = MLP(input_dim, hidden_dim, output_dim).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0005)

train_mlp(model, criterion, optimizer, X_train, y_train)
evaluate_mlp(model, X_test, y_test)