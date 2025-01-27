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
from plot import plot_mlp_training, plot_gan_losses, plot_feature_distribution
from config import (
    DEVICE,
    DELIMITER,
    LOG_FORMAT,
    INDEX_PATH,
    EXAMPLE_PATH,
    CHECKPOINT_DIR,
    MLP_ORIGINAL_PLOT_PATH,
    MLP_AUGMENTED_PLOT_PATH,
    GAN_PLOT_PATH,
    FD_ORIGINAL_DATA_PLOT_PATH,
    FD_AUGMENTED_DATA_PLOT_PATH,
    ONE_CLASS,
)

BATCH_SIZE = 4096 if ONE_CLASS == "spam" else 1024
TRAIN_SPLIT = 0.75
NUM_EPOCHS_GAN = 5967
LR_GAN = (
    [0.000305, 0.0004] if ONE_CLASS == "spam" else [0.0002, 0.0004]
)  # Taxa de aprendizado para o gerador e discriminador

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
torch.cuda.empty_cache()


def __identify_minority_class(labels: np.ndarray) -> Tuple[int, int, int]:
    """Identifica a classe minoritária e retorna informações relevantes.

    Args:
        labels (np.ndarray): Array com os rótulos dos dados

    Returns:
        Tuple[int, int, int]: Classe minoritária, quantidade de amostras necessárias,
                             quantidade total de amostras da classe majoritária
    """
    unique_labels, counts = np.unique(labels, return_counts=True)
    class_counts = dict(zip(unique_labels, counts))

    minority_class = min(class_counts.items(), key=lambda x: x[1])[0]
    majority_class = max(class_counts.items(), key=lambda x: x[1])[0]

    samples_needed = class_counts[majority_class] - class_counts[minority_class]

    logging.info(
        f"\nDistribuição das classes:\n"
        f"Classe 0 (ham): {class_counts[0]} amostras\n"
        f"Classe 1 ({ONE_CLASS}): {class_counts[1]} amostras\n"
        f"Classe minoritária: {np.int32(minority_class)}\n"
        f"Amostras necessárias para balanceamento: {samples_needed}\n"
        f"{DELIMITER}"
    )

    return minority_class, samples_needed, class_counts[majority_class]


def set_seed(seed: int) -> None:
    """Sets the seed for reproducibility.

    Args:
        seed (int): The seed value.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def __load_and_preprocess_data() -> (
    Tuple[TensorDataset, TensorDataset, torch.Tensor, torch.Tensor, int]
):
    """Loads and preprocesses the data for training and testing.

    Returns:
        Tuple[TensorDataset, TensorDataset, torch.Tensor, torch.Tensor, int]:
        The training dataset, test dataset, data tensor, index tensor, and input dimension.
    """
    try:
        logging.info(
            f"\n{DELIMITER}\n" f"Loading and preprocessing data...\n" f"{DELIMITER}"
        )
        selected_data = ["ham", ONE_CLASS]
        data, index = get_training_test_set(INDEX_PATH, selected_data, 1.0)

        sample_counts = np.bincount(index)
        logging.info(
            f"\n{ONE_CLASS.title()} samples: {sample_counts[1]}\n"
            f"Ham samples: {sample_counts[0]}\n"
            f"{DELIMITER}"
        )

        data_tensor = torch.tensor(data, dtype=torch.float32).pin_memory()
        index_tensor = torch.tensor(index, dtype=torch.float32).pin_memory()

        dataset = TensorDataset(data_tensor, index_tensor)
        train_size = int(TRAIN_SPLIT * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        return (
            train_dataset,
            test_dataset,
            data_tensor,
            index_tensor,
            data_tensor.shape[1],
        )
    finally:
        torch.cuda.empty_cache()


def __setup_dynamic_gan(
    train_set: torch.Tensor,
    train_labels: torch.Tensor,
    input_dim: int,
    minority_class: int,
) -> Generator:
    """Configura e treina o GAN para a classe minoritária.

    Args:
        train_set (torch.Tensor): Conjunto de treinamento
        train_labels (torch.Tensor): Rótulos de treinamento
        input_dim (int): Dimensão de entrada
        minority_class (int): Classe minoritária identificada

    Returns:
        Generator: Modelo gerador treinado
    """
    class_name = "ham" if minority_class == 0 else ONE_CLASS
    logging.info(f"\nTreinando GAN para {class_name}...\n{DELIMITER}")

    minority_data = train_set[train_labels.cpu().numpy() == minority_class]
    minority_labels = train_labels[train_labels.cpu().numpy() == minority_class]
    minority_dataset = TensorDataset(minority_data.to(DEVICE), minority_labels)
    minority_loader = DataLoader(
        minority_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )

    generator = Generator(input_dim, input_dim).to(DEVICE)
    discriminator = Discriminator(input_dim).to(DEVICE)

    d_loss, g_loss = train_gan(
        generator,
        discriminator,
        minority_loader,
        input_dim,
        device=DEVICE,
        checkpoint_dir=CHECKPOINT_DIR + class_name + "/",
        num_epochs=NUM_EPOCHS_GAN,
        lr_g=LR_GAN[0],
        lr_d=LR_GAN[1],
    )

    if d_loss and g_loss:
        plot_gan_losses(d_loss, g_loss, GAN_PLOT_PATH)

    return generator


def __generate_and_augment_data(
    generator: Generator,
    train_set: torch.Tensor,
    train_labels: torch.Tensor,
    input_dim: int,
    minority_class: int,
    samples_needed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Gera exemplos sintéticos e aumenta o conjunto de treinamento.

    Args:
        generator (Generator): Modelo gerador treinado
        train_set (torch.Tensor): Conjunto de treinamento
        train_labels (torch.Tensor): Rótulos de treinamento
        input_dim (int): Dimensão de entrada
        minority_class (int): Classe minoritária
        samples_needed (int): Quantidade de amostras necessárias

    Returns:
        Tuple[np.ndarray, np.ndarray]: Conjunto de treinamento aumentado e rótulos
    """
    class_name = "ham" if minority_class == 0 else ONE_CLASS
    logging.info(
        f"\nGerando {samples_needed} exemplos sintéticos de {class_name}...\n{DELIMITER}"
    )

    generated_samples = generate_adversarial_examples(
        generator, samples_needed, input_dim, device=DEVICE
    )

    logging.info(f"\nAumentando o conjunto de treinamento...\n{DELIMITER}")
    augmented_train_set = np.vstack([train_set.cpu().numpy(), generated_samples])
    augmented_train_labels = np.hstack(
        [train_labels.cpu().numpy(), np.full(samples_needed, minority_class)]
    )

    return augmented_train_set, augmented_train_labels


def __train_and_evaluate_mlp(
    train_set: np.ndarray,
    train_labels: np.ndarray,
    augmented_train_set: np.ndarray,
    augmented_train_labels: np.ndarray,
    test_set: np.ndarray,
    test_labels: np.ndarray,
) -> Tuple[MLP, float, float]:
    """Trains and evaluates the MLP model with and without augmented data.

    Args:
        train_set (np.ndarray): The original training set.
        train_labels (np.ndarray): The original training labels.
        augmented_train_set (np.ndarray): The augmented training set.
        augmented_train_labels (np.ndarray): The augmented training labels.
        test_set (np.ndarray): The test set.
        test_labels (np.ndarray): The test labels.

    Returns:
        Tuple[MLP, float, float]: The trained MLP model, accuracy with augmented data, and accuracy without augmented data.
    """
    try:
        logging.info(
            f"\nTraining and evaluating the MLP with augmented data...\n" f"{DELIMITER}"
        )
        example_data, example_index, _ = get_example_test_set(EXAMPLE_PATH)
        best_config = get_hyperparameters(
            augmented_train_set,
            test_set,
            augmented_train_labels,
            test_labels,
            example_data=example_data,
            example_labels=example_index,
            config=ONE_CLASS,
        )
        num_epochs = 15000
        # best_config = {
        #    'l1_lambda': 0.00011,
        #    'l2_lambda': 0.00005,
        #    'hidden_dim1': 32,
        #    'hidden_dim2': 16,
        #    'lr': 0.0002,
        #    'weight_decay': 0.0001,
        #    'num_epochs': num_epochs,
        #    'dropout': 0.0,
        #    'patience': 80
        # }
        print(DELIMITER)
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

        X_train_augmented = torch.tensor(augmented_train_set, dtype=torch.float32).to(
            DEVICE
        )
        y_train_augmented = torch.tensor(augmented_train_labels, dtype=torch.long).to(
            DEVICE
        )
        X_test = torch.tensor(test_set, dtype=torch.float32).to(DEVICE)
        y_test = torch.tensor(test_labels, dtype=torch.long).to(DEVICE)

        input_dim = X_train_augmented.shape[1]
        output_dim = 2

        model_augmented = MLP(
            input_dim,
            best_config["hidden_dim1"],
            best_config["hidden_dim2"],
            output_dim,
            l1_lambda=best_config["l1_lambda"],
            l2_lambda=best_config["l2_lambda"],
            dropout=best_config["dropout"],
        ).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            model_augmented.parameters(),
            lr=best_config["lr"],
            weight_decay=best_config["weight_decay"],
        )

        _, augmented_train_loss, augmented_val_loss = train_mlp(
            model_augmented,
            criterion,
            optimizer,
            X_train_augmented,
            y_train_augmented,
            X_test,
            y_test,
            num_epochs=best_config["num_epochs"],
            patience=best_config["patience"],
        )

        accuracy_augmented, cm_augmented = evaluate_mlp(model_augmented, X_test, y_test)
        logging.info(
            f"\nConfusion matrix for MLP classifier:\n"
            f"{cm_augmented}\n"
            f"{DELIMITER}"
        )

        plot_mlp_training(
            augmented_train_loss,
            augmented_val_loss,
            cm_augmented,
            MLP_AUGMENTED_PLOT_PATH,
        )

        logging.info(
            f"\nTraining and evaluating the MLP without augmented data...\n"
            f"{DELIMITER}"
        )
        X_train_original = torch.tensor(train_set, dtype=torch.float32).to(DEVICE)
        y_train_original = torch.tensor(train_labels, dtype=torch.long).to(DEVICE)

        model_original = MLP(
            input_dim,
            best_config["hidden_dim1"],
            best_config["hidden_dim2"],
            output_dim,
            l1_lambda=best_config["l1_lambda"],
            l2_lambda=best_config["l2_lambda"],
            dropout=best_config["dropout"],
        ).to(DEVICE)
        optimizer = optim.Adam(
            model_original.parameters(),
            lr=best_config["lr"],
            weight_decay=best_config["weight_decay"],
        )

        _, original_train_loss, original_val_loss = train_mlp(
            model_original,
            criterion,
            optimizer,
            X_train_original,
            y_train_original,
            X_test,
            y_test,
            num_epochs=best_config["num_epochs"],
            patience=best_config["patience"],
        )

        accuracy_original, cm_original = evaluate_mlp(model_original, X_test, y_test)
        logging.info(
            f"\nConfusion matrix for MLP classifier:\n"
            f"{cm_original}\n"
            f"{DELIMITER}"
        )

        plot_mlp_training(
            original_train_loss, original_val_loss, cm_original, MLP_ORIGINAL_PLOT_PATH
        )

        return model_augmented, accuracy_augmented, accuracy_original
    finally:
        torch.cuda.empty_cache()


def main() -> None:
    """Main function to execute the training and evaluation process."""
    try:
        set_seed(42)
        train_dataset, test_dataset, data_tensor, index_tensor, input_dim = (
            __load_and_preprocess_data()
        )

        train_set = data_tensor[train_dataset.indices]
        test_set = data_tensor[test_dataset.indices]
        train_labels = index_tensor[train_dataset.indices]
        test_labels = index_tensor[test_dataset.indices]

        minority_class, samples_needed, _ = __identify_minority_class(
            train_labels.cpu().numpy()
        )

        ham_data_original = train_set[train_labels.cpu().numpy() == 0].tolist()
        phishing_spam_data_original = train_set[
            train_labels.cpu().numpy() == 1
        ].tolist()

        plot_feature_distribution(
            ham_data_original,
            phishing_spam_data_original,
            FD_ORIGINAL_DATA_PLOT_PATH,
        )

        generator = __setup_dynamic_gan(
            train_set, train_labels, input_dim, minority_class
        )

        augmented_train_set, augmented_train_labels = __generate_and_augment_data(
            generator,
            train_set,
            train_labels,
            input_dim,
            minority_class,
            samples_needed,
        )

        ham_data_augmented = augmented_train_set[augmented_train_labels == 0].tolist()
        phishing_spam_data_augmented = augmented_train_set[
            augmented_train_labels == 1
        ].tolist()

        plot_feature_distribution(
            ham_data_augmented,
            phishing_spam_data_augmented,
            FD_AUGMENTED_DATA_PLOT_PATH,
        )

        mlp_augmented, accuracy_augmented, accuracy_original = __train_and_evaluate_mlp(
            train_set.cpu().numpy(),
            train_labels.cpu().numpy(),
            augmented_train_set,
            augmented_train_labels,
            test_set.cpu().numpy(),
            test_labels.cpu().numpy(),
        )

        logging.info(
            f"\nComparison of results:\n"
            f"Accuracy with GAN-augmented data: {accuracy_augmented:.2f}%\n"
            f"Accuracy without GAN-augmented data: {accuracy_original:.2f}%\n"
            f"Accuracy gain: {accuracy_augmented - accuracy_original:.2f}%\n"
            f"{DELIMITER}"
        )

        logging.info("Testing with example test set...")
        example_data, example_index, email_paths = get_example_test_set(EXAMPLE_PATH)
        X_example_test = torch.tensor(example_data, dtype=torch.float32).to(DEVICE)

        predicted_example_label, label_probabilities = predict_mlp(
            mlp_augmented, X_example_test
        )
        expected_labels_readable = [
            ONE_CLASS if label in [1] else "ham" for label in example_index
        ]

        for i, (_, prob) in enumerate(
            zip(predicted_example_label, label_probabilities)
        ):
            prob_phishing_spam = prob[1].item()
            prob_ham = prob[0].item()
            logging.info(
                f"The email at path {email_paths[i]} is [ham: {prob_ham:.4f}, {ONE_CLASS}: {prob_phishing_spam:.4f}] (expected: {expected_labels_readable[i]})"
            )

        print(DELIMITER)

        torch.cuda.empty_cache()
    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
    finally:
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
