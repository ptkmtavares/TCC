import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from dataExtractor import getExampleTestSet, getTrainingTestSet
from torch.utils.data import DataLoader, TensorDataset
from mlp import MLP, predict_mlp, train_mlp, evaluate_mlp
from gan import Generator, Discriminator, train_gan, generate_adversarial_examples
from rayParam import getHyperparameters

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carregar e pré-processar os dados
print(
    f"{'='*75}\n"
    f"🚀 Loading and preprocessing data...\n"
    f"{'='*75}"
)
selected_data = ['ham', 'phishing']
data, index = getTrainingTestSet('Dataset/index', selected_data, 1.0)

# Calcular as contagens de amostras
sample_counts = np.bincount(index)

print(
    f"🎣 Phishing samples: {sample_counts[2]}\n"
    f"📨 Ham samples: {sample_counts[0]}\n"
    f"{'='*75}"
)

train_set, test_set, train_labels, test_labels = train_test_split(data, index, train_size=0.75, random_state=9, shuffle=True)

scaler = MinMaxScaler(feature_range=(0, 1))
train_set_normalized = scaler.fit_transform(train_set)
test_set_normalized = scaler.transform(test_set)

train_set_normalized = torch.tensor(train_set_normalized, dtype=torch.float32).to(device)
train_labels = torch.tensor(train_labels, dtype=torch.float32).to(device)

train_dataset = TensorDataset(train_set_normalized, train_labels)
batch_size = 1024
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

train_set = np.array(train_set)
test_set = np.array(test_set)

input_dim = train_set.shape[1]

# Treinar GAN para phishing
print(
    f"🎣 Training GAN for phishing...\n"
    f"{'='*75}"
)
phishing_data = train_set[train_labels.cpu().numpy() == 2]
phishing_labels = train_labels[train_labels.cpu().numpy() == 2]
phishing_dataset = TensorDataset(torch.tensor(phishing_data, dtype=torch.float32).to(device), phishing_labels)
phishing_loader = DataLoader(phishing_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

G_phishing = Generator(input_dim, input_dim).to(device)
D_phishing = Discriminator(input_dim).to(device)
train_gan(G_phishing, D_phishing, phishing_loader, input_dim, device=device, checkpoint_dir='checkpoints/phishing/', num_epochs=7500)

# Gerar exemplos adversariais
print(
    f"🔍 Generating adversarial examples...\n"
    f"{'='*75}"
)
ham_data = train_set[train_labels.cpu().numpy() == 0]
num_samples_phishing = len(ham_data) - len(phishing_data)

if num_samples_phishing <= 0:
    print(
        f"🚫 Number of phishing samples is greater than or equal to the number of ham samples. Defaulting to 1000...\n"
        f"{'='*75}"
    )
    num_samples_phishing = 1000

generated_phishing = generate_adversarial_examples(G_phishing, num_samples_phishing, input_dim, device=device)

print(
    f"🎣 Generated phishing examples: {num_samples_phishing}\n"
    f"{'='*75}"
)

# Aumentar o conjunto de treinamento
print(
    f"📈 Augmenting the training set...\n"
    f"{'='*75}"
)
augmented_train_set = np.vstack([train_set, generated_phishing])
augmented_train_labels = np.hstack([train_labels.cpu().numpy(), np.full(num_samples_phishing, 2)])

scaler = MinMaxScaler(feature_range=(0, 1))
augmented_train_set_normalized = scaler.fit_transform(augmented_train_set)
test_set_normalized = scaler.transform(test_set)

X_train_augmented = torch.tensor(augmented_train_set_normalized, dtype=torch.float32).to(device)
y_train_augmented = torch.tensor(augmented_train_labels, dtype=torch.long).to(device)
X_test = torch.tensor(test_set_normalized, dtype=torch.float32).to(device)
y_test = torch.tensor(test_labels, dtype=torch.long).to(device)

# Treinar e avaliar o MLP com dados aumentados
print(
    f"🧠 Training and evaluating the MLP with augmented data...\n"
    f"{'='*75}"
)

best_config = getHyperparameters(augmented_train_set, test_set, augmented_train_labels, test_labels)
print(
    f"🔧 Best hyperparameters found:\n"
    f"Hidden dimension: {best_config['hidden_dim']}\n"
    f"Learning rate: {best_config['lr']}\n"
    f"Weight decay: {best_config['weight_decay']}\n"
    f"Number of epochs: {best_config['num_epochs']}\n"
    f"Patience: {best_config['patience']}\n"
    f"{'='*75}"
)
input_dim = X_train_augmented.shape[1]
output_dim = 3

model_augmented = MLP(input_dim, best_config["hidden_dim"], output_dim).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_augmented.parameters(), lr=best_config["lr"], weight_decay=best_config["weight_decay"])

train_mlp(model_augmented, criterion, optimizer, X_train_augmented, y_train_augmented, X_test, y_test, num_epochs=best_config["num_epochs"], patience=best_config["patience"])
accuracy_augmented = evaluate_mlp(model_augmented, X_test, y_test)

# Treinar e avaliar o MLP sem dados aumentados
print(
    f"🧠 Training and evaluating the MLP without augmented data...\n"
    f"{'='*75}"
)
X_train_original = torch.tensor(train_set_normalized.cpu().numpy(), dtype=torch.float32).to(device)
y_train_original = torch.tensor(train_labels.cpu().numpy(), dtype=torch.long).to(device)

best_config = getHyperparameters(train_set_normalized.cpu().numpy(), test_set, train_labels.cpu().numpy(), test_labels)
print(
    f"🔧 Best hyperparameters found:\n"
    f"Hidden dimension: {best_config['hidden_dim']}\n"
    f"Learning rate: {best_config['lr']}\n"
    f"Weight decay: {best_config['weight_decay']}\n"
    f"Number of epochs: {best_config['num_epochs']}\n"
    f"Patience: {best_config['patience']}\n"
    f"{'='*75}"
)
input_dim = X_train_original.shape[1]
model_original = MLP(input_dim, best_config["hidden_dim"], output_dim).to(device)
optimizer = optim.Adam(model_original.parameters(), lr=best_config["lr"], weight_decay=best_config["weight_decay"])

train_mlp(model_original, criterion, optimizer, X_train_original, y_train_original, X_test, y_test, num_epochs=best_config["num_epochs"], patience=best_config["patience"])
accuracy_original = evaluate_mlp(model_original, X_test, y_test)

# Comparar os resultados
print(
    f"📊 Comparison of results:\n"
    f"✅ Accuracy with GAN-augmented data: {accuracy_augmented:.2f}%\n"
    f"✅ Accuracy without GAN-augmented data: {accuracy_original:.2f}%\n"
    f"Accuracy gain: {accuracy_augmented - accuracy_original:.2f}%\n"
    f"{'='*75}"
)

# Testar com exemplos de teste
print(
    f"🔬 Testing with example test set...\n"
    f"{'='*75}"
)

example_data, example_index = getExampleTestSet('Dataset/exampleIndex')
example_test_set_normalized = scaler.transform(example_data)
X_example_test = torch.tensor(example_test_set_normalized, dtype=torch.float32).to(device)

predicted_example_label, label_probabilities = predict_mlp(model_augmented, X_example_test)

label_dict = {0: 'ham', 1: 'spam', 2: 'phishing'}
predicted_labels_readable = [label_dict[label.item()] for label in predicted_example_label]
expected_labels_readable = [label_dict[label] for label in example_index]

print(
    f"🔖 Predicted labels for example test set:"
)
for i, (label, prob) in enumerate(zip(predicted_example_label, label_probabilities)):
    print(
        f"The email of index {i} is {[f'{label_dict[j]}: {prob:.4f}' for j, prob in enumerate(prob.cpu().numpy())]} (expected: {expected_labels_readable[i]})"
    )
print(
    f"{'='*75}"
)