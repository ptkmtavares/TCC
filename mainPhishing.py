import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from typing import Tuple
from dataExtractor import get_example_test_set, get_training_test_set
from torch.utils.data import DataLoader, TensorDataset, random_split
from mlp import MLP, predict_mlp, train_mlp, evaluate_mlp
from gan import Generator, Discriminator, train_gan, generate_adversarial_examples
from rayParam import get_hyperparameters

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1024
TRAIN_SPLIT = 0.75
NUM_EPOCHS_GAN = 5625
CHECKPOINT_DIR = 'checkpoints/phishing/'
INDEX_DIR = 'Dataset/index'
EXAMPLE_DIR = 'Dataset/exampleIndex'
DELIMITER = '='*75
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s\n'

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()

def load_and_preprocess_data() -> Tuple[TensorDataset, TensorDataset, torch.Tensor, torch.Tensor, int]:
    try:
        logging.info(
            f"\n{DELIMITER}\n"
            f"Loading and preprocessing data...\n"
            f"{DELIMITER}"
        )
        selected_data = ['ham', 'phishing']
        data, index = get_training_test_set(INDEX_DIR, selected_data, 1.0)

        sample_counts = np.bincount(index)
        logging.info(
            f"\nPhishing samples: {sample_counts[1]}\n"
            f"Ham samples: {sample_counts[0]}\n"
            f"{DELIMITER}"
        )

        data_tensor = torch.tensor(data, dtype=torch.float32, pin_memory=True)
        index_tensor = torch.tensor(index, dtype=torch.float32, pin_memory=True)

        dataset = TensorDataset(data_tensor, index_tensor)
        train_size = int(TRAIN_SPLIT * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        return train_dataset, test_dataset, data_tensor, index_tensor, data_tensor.shape[1]
    finally:
        torch.cuda.empty_cache()

def setup_gan(train_set: torch.Tensor, train_labels: torch.Tensor, input_dim: int) -> Generator:
    logging.info(
        f"\nTraining GAN for phishing...\n"
        f"{DELIMITER}"
    )
    phishing_data = train_set[train_labels.cpu().numpy() == 1]
    phishing_labels = train_labels[train_labels.cpu().numpy() == 1]
    phishing_dataset = TensorDataset(phishing_data.to(DEVICE), phishing_labels)
    phishing_loader = DataLoader(phishing_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    g_phishing = Generator(input_dim, input_dim).to(DEVICE)
    d_phishing = Discriminator(input_dim).to(DEVICE)
    train_gan(g_phishing, d_phishing, phishing_loader, input_dim, device=DEVICE, checkpoint_dir=CHECKPOINT_DIR, num_epochs=NUM_EPOCHS_GAN)

    return g_phishing

def generate_and_augment_data(g_phishing: Generator, train_set: torch.Tensor, train_labels: torch.Tensor, input_dim: int) -> Tuple[np.ndarray, np.ndarray]:
    logging.info(
        f"\nGenerating adversarial examples...\n"
        f"{DELIMITER}"
    )
    ham_data = train_set[train_labels.cpu().numpy() == 0]
    phishing_data = train_set[train_labels.cpu().numpy() == 1]
    num_samples_phishing = len(ham_data) - len(phishing_data)

    if num_samples_phishing <= 0:
        logging.warning(
            f"\nNumber of phishing samples is greater than or equal to the number of ham samples. Defaulting to 1000...\n"
            f"{DELIMITER}"
        )
        num_samples_phishing = 1000

    generated_phishing = generate_adversarial_examples(g_phishing, num_samples_phishing, input_dim, device=DEVICE)
    logging.info(
        f"\nGenerated phishing examples: {num_samples_phishing}\n"
        f"{DELIMITER}"
    )

    logging.info(
        f"\nAugmenting the training set...\n"
        f"{DELIMITER}"
    )
    augmented_train_set = np.vstack([train_set.cpu().numpy(), generated_phishing])
    augmented_train_labels = np.hstack([train_labels.cpu().numpy(), np.full(num_samples_phishing, 1)])

    return augmented_train_set, augmented_train_labels

def train_and_evaluate_mlp(train_set: np.ndarray, train_labels: np.ndarray, augmented_train_set: np.ndarray, augmented_train_labels: np.ndarray, test_set: np.ndarray, test_labels: np.ndarray) -> Tuple[MLP, float, float]:
    try:
        logging.info(
            f"\nTraining and evaluating the MLP with augmented data...\n"
            f"{DELIMITER}"
        )
        example_data, example_index, _ = get_example_test_set(EXAMPLE_DIR)
        best_config = get_hyperparameters(augmented_train_set, test_set, augmented_train_labels, test_labels, example_data=example_data, example_labels=example_index, config='phishing')
        logging.info(
            f"\nBest hyperparameters found:\n"
            f"L1: {best_config['l1_lambda']}\n"
            f"L2: {best_config['l2_lambda']}\n"
            f"Hidden dimension 1: {best_config['hidden_dim1']}\n"
            f"Hidden dimension 2: {best_config['hidden_dim2']}\n"
            f"Learning rate: {best_config['lr']}\n"
            f"Weight decay: {best_config['weight_decay']}\n"
            f"Number of epochs: {best_config['num_epochs']}\n"
            f"Dropout: {best_config['dropout']}\n"
            f"Patience: {best_config['patience']}\n"
            f"{DELIMITER}"
        )

        X_train_augmented = torch.tensor(augmented_train_set, dtype=torch.float32).to(DEVICE)
        y_train_augmented = torch.tensor(augmented_train_labels, dtype=torch.long).to(DEVICE)
        X_test = torch.tensor(test_set, dtype=torch.float32).to(DEVICE)
        y_test = torch.tensor(test_labels, dtype=torch.long).to(DEVICE)

        input_dim = X_train_augmented.shape[1]
        output_dim = 2

        model_augmented = MLP(input_dim, best_config["hidden_dim1"], best_config["hidden_dim2"], output_dim, l1_lambda=best_config['l1_lambda'], l2_lambda=best_config['l2_lambda'], dropout=best_config['dropout']).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model_augmented.parameters(), lr=best_config["lr"], weight_decay=best_config["weight_decay"])
        
        train_mlp(model_augmented, criterion, optimizer, X_train_augmented, y_train_augmented, X_test, y_test, num_epochs=best_config["num_epochs"], patience=best_config["patience"])
        accuracy_augmented = evaluate_mlp(model_augmented, X_test, y_test)

        logging.info(
            f"\nTraining and evaluating the MLP without augmented data...\n"
            f"{DELIMITER}"
        )
        X_train_original = torch.tensor(train_set, dtype=torch.float32).to(DEVICE)
        y_train_original = torch.tensor(train_labels, dtype=torch.long).to(DEVICE)

        model_original = MLP(input_dim, best_config["hidden_dim1"], best_config["hidden_dim2"], output_dim, l1_lambda=best_config['l1_lambda'], l2_lambda=best_config['l2_lambda'], dropout=best_config['dropout']).to(DEVICE)
        optimizer = optim.Adam(model_original.parameters(), lr=best_config["lr"], weight_decay=best_config["weight_decay"])

        train_mlp(model_original, criterion, optimizer, X_train_original, y_train_original, X_test, y_test, num_epochs=best_config["num_epochs"], patience=best_config["patience"])
        accuracy_original = evaluate_mlp(model_original, X_test, y_test)

        return model_augmented, accuracy_augmented, accuracy_original
    finally:
        torch.cuda.empty_cache()

def main():
    try:
        train_dataset, test_dataset, data_tensor, index_tensor, input_dim = load_and_preprocess_data()

        train_set = data_tensor[train_dataset.indices]
        test_set = data_tensor[test_dataset.indices]
        train_labels = index_tensor[train_dataset.indices]
        test_labels = index_tensor[test_dataset.indices]

        g_phishing = setup_gan(train_set, train_labels, input_dim)
        augmented_train_set, augmented_train_labels = generate_and_augment_data(g_phishing, train_set, train_labels, input_dim)
        mlp_augmented, accuracy_augmented, accuracy_original = train_and_evaluate_mlp(train_set.cpu().numpy(), train_labels.cpu().numpy(), augmented_train_set, augmented_train_labels, test_set.cpu().numpy(), test_labels.cpu().numpy())

        logging.info(
            f"\nComparison of results:\n"
            f"Accuracy with GAN-augmented data: {accuracy_augmented:.2f}%\n"
            f"Accuracy without GAN-augmented data: {accuracy_original:.2f}%\n"
            f"Accuracy gain: {accuracy_augmented - accuracy_original:.2f}%\n"
            f"{DELIMITER}"
        )

        logging.info("Testing with example test set...")
        example_data, example_index, email_paths = get_example_test_set(EXAMPLE_DIR)
        X_example_test = torch.tensor(example_data, dtype=torch.float32).to(DEVICE)

        predicted_example_label, label_probabilities = predict_mlp(mlp_augmented, X_example_test)
        expected_labels_readable = ['phishing' if label in [1] else 'ham' for label in example_index]
        
        for i, (_, prob) in enumerate(zip(predicted_example_label, label_probabilities)):
            prob_phishing_spam = prob[1].item()
            prob_ham = prob[0].item()
            logging.info(f"The email at path {email_paths[i]} is [ham: {prob_ham:.4f}, phishing: {prob_phishing_spam:.4f}] (expected: {expected_labels_readable[i]})")
        
        logging.info(DELIMITER)
        
        torch.cuda.empty_cache()
    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
    finally:
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()