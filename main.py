import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from typing import Tuple
from dataExtractor import get_example_test_set, get_training_test_set
from torch.utils.data import (
    DataLoader,
    TensorDataset,
    random_split,
    WeightedRandomSampler,
)
from mlp import MLP, predict_mlp, train_mlp, evaluate_mlp
from gan import Generator, Discriminator, train_gan, generate_adversarial_examples
from rayParam import get_hyperparameters
from plot import (
    plot_linear_regression,
    plot_mlp_training,
    plot_gan_losses,
    plot_feature_distribution,
)
from c2st import perform_c2st
from config import (
    DEVICE,
    DELIMITER,
    GEN_TEMPERATURE,
    LOG_FORMAT,
    MLP_ORIGINAL_PLOT_PATH,
    MLP_AUGMENTED_PLOT_PATH,
    FD_ORIGINAL_DATA_PLOT_PATH,
    FD_AUGMENTED_DATA_PLOT_PATH,
    LR_ORIGINAL_DATA_PLOT_PATH,
    LR_AUGMENTED_DATA_PLOT_PATH,
    NUM_WORKERS,
    ONE_CLASS,
    GAN_BATCH_SIZE,
    MLP_AUGMENTED_BATCH_SIZE,
    MLP_ORIGINAL_BATCH_SIZE,
    TRAIN_SPLIT,
)

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
torch.cuda.empty_cache()
os.environ["NUMEXPR_MAX_THREADS"] = "8"


def check_class_distribution(dataloader, name=""):
    """Verifica a distribuição das classes em um dataloader"""
    class_counts = {0.0: 0, 1.0: 0}
    for _, labels in dataloader:
        for label in labels:
            class_counts[label.item()] += 1

    total = sum(class_counts.values())
    logging.info(
        f"\nDistribuição de classes no {name}:"
        f"\nTotal de amostras: {total}"
        f"\nClasse 0 (ham): {class_counts[0]} ({(class_counts[0]/total)*100:.2f}%)"
        f"\nClasse 1 ({ONE_CLASS}): {class_counts[1]} ({(class_counts[1]/total)*100:.2f}%)"
        f"\n{DELIMITER}"
    )


def create_dataloader(dataset, batch_size):
    labels = [y.item() for _, y in dataset]

    class_counts = torch.bincount(torch.tensor(labels, dtype=torch.long))
    total_samples = len(labels)

    class_weights = total_samples / (class_counts.float() * len(class_counts))

    sample_weights = torch.tensor(
        [class_weights[int(label)].item() for label in labels]
    )

    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        persistent_workers=True,
        pin_memory=True,
    )


def create_deterministic_dataloader(dataset, batch_size, shuffle_seed=23):
    """
    Cria um DataLoader com amostragem ponderada para lidar com classes desequilibradas.

    Args:
        dataset: Dataset do PyTorch
        batch_size: Tamanho do batch
        shuffle_seed: Semente para reprodutibilidade
    """
    labels = [y.item() for _, y in dataset]

    class_counts = torch.bincount(torch.tensor(labels, dtype=torch.long))
    total_samples = len(labels)

    class_weights = total_samples / (class_counts.float() * len(class_counts))

    sample_weights = torch.tensor(
        [class_weights[int(label)].item() for label in labels]
    )

    generator = torch.Generator()
    generator.manual_seed(shuffle_seed)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
        generator=generator,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        persistent_workers=True,
        pin_memory=True,
    )


def __identify_minority_class(labels: np.ndarray) -> Tuple[int, int, int]:
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
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def __load_and_preprocess_data() -> (
    Tuple[TensorDataset, TensorDataset, torch.Tensor, torch.Tensor]
):
    try:
        logging.info(
            f"\n{DELIMITER}\n" f"Loading and preprocessing data...\n" f"{DELIMITER}"
        )
        data, index = get_training_test_set()

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
        )
    finally:
        torch.cuda.empty_cache()


def __setup_dynamic_gan(
    train_set: torch.Tensor,
    train_labels: torch.Tensor,
    minority_class: int,
) -> Generator:
    class_name = "ham" if minority_class == 0 else ONE_CLASS
    logging.info(f"\nTreinando GAN para {class_name}...\n{DELIMITER}")

    minority_data = train_set[train_labels.cpu().numpy() == minority_class].cpu()
    minority_labels = train_labels[train_labels.cpu().numpy() == minority_class].cpu()
    minority_dataset = TensorDataset(minority_data, minority_labels)
    minority_loader = create_dataloader(minority_dataset, GAN_BATCH_SIZE)

    generator = Generator().to(DEVICE)
    discriminator = Discriminator().to(DEVICE)

    d_loss, g_loss = train_gan(generator, discriminator, minority_loader)

    if d_loss and g_loss:
        plot_gan_losses(d_loss, g_loss)

    return generator


def __generate_and_augment_data(
    generator: Generator,
    train_set: torch.Tensor,
    train_labels: torch.Tensor,
    minority_class: int,
    samples_needed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    class_name = "ham" if minority_class == 0 else ONE_CLASS

    logging.info(
        f"\nGerando {samples_needed} exemplos sintéticos com temperatura {GEN_TEMPERATURE} de {class_name}...\n{DELIMITER}"
    )

    generated_samples = generate_adversarial_examples(generator, samples_needed)

    # Realizar C2ST
    logging.info(f"\nRealizando Classifier Two-Sample Test (C2ST)...\n{DELIMITER}")
    real_samples = train_set[train_labels == minority_class].cpu().numpy()
    c2st_accuracy, p_value = perform_c2st(real_samples, generated_samples, DEVICE)
    logging.info(
        f"\nResultados do Classifier Two-Sample Test (C2ST):\n"
        f"Acurácia do C2ST: {c2st_accuracy:.2f}%\n"
        f"P-valor: {p_value:.4f}\n"
        f"{'Os dados gerados são estatisticamente diferentes dos reais' if p_value < 0.05 else 'Os dados gerados são similares aos reais'}\n"
        f"{DELIMITER}"
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
    try:
        logging.info(f"\nVerificando os melhores hiperparâmetros...\n{DELIMITER}")
        example_data, example_index, _ = get_example_test_set()
        # best_config = get_hyperparameters(
        #    augmented_train_set,
        #    test_set,
        #    augmented_train_labels,
        #    test_labels,
        #    example_data=example_data,
        #    example_labels=example_index,
        # )

        best_config = {
            "l1_lambda": 1e-4,
            "l2_lambda": 1e-5,
            "hidden_dim": 32,
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "num_epochs": 1000,
            "patience": 20,
            "dropout": 0.4,
        }

        print(DELIMITER)
        logging.info(
            f"\nBest hyperparameters found:\n"
            f"L1: {best_config['l1_lambda']}\n"
            f"L2: {best_config['l2_lambda']}\n"
            f"Hidden dimension: {best_config['hidden_dim']}\n"
            f"Learning rate: {best_config['lr']}\n"
            f"Weight decay: {best_config['weight_decay']}\n"
            f"Number of epochs: {best_config['num_epochs']}\n"
            f"Dropout: {best_config['dropout']}\n"
            f"Patience: {best_config['patience']}\n"
            f"{DELIMITER}"
        )

        input_dim = train_set.shape[1]
        output_dim = 2

        logging.info(
            f"\nTraining and evaluating the MLP without augmented data...\n"
            f"{DELIMITER}"
        )

        X_train_original = torch.tensor(train_set, dtype=torch.float32).cpu()
        y_train_original = torch.tensor(train_labels, dtype=torch.long).cpu()
        X_test = torch.tensor(test_set, dtype=torch.float32).cpu()
        y_test = torch.tensor(test_labels, dtype=torch.long).cpu()

        train_dataset_original = TensorDataset(X_train_original, y_train_original)
        val_dataset_original = TensorDataset(X_test, y_test)

        train_loader_original = create_deterministic_dataloader(
            train_dataset_original,
            MLP_ORIGINAL_BATCH_SIZE,
        )

        val_loader_original = create_dataloader(
            val_dataset_original, MLP_ORIGINAL_BATCH_SIZE
        )

        check_class_distribution(train_loader_original, "DataLoader Original (Treino)")
        check_class_distribution(val_loader_original, "DataLoader Original (Validação)")

        model_original = MLP(
            input_dim=input_dim,
            hidden_dim=best_config["hidden_dim"],
            output_dim=output_dim,
            l1_lambda=best_config["l1_lambda"],
            l2_lambda=best_config["l2_lambda"],
            dropout=best_config["dropout"],
        ).to(DEVICE)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            model_original.parameters(),
            lr=best_config["lr"],
            weight_decay=best_config["weight_decay"],
        )

        _, original_train_loss, original_val_loss = train_mlp(
            model=model_original,
            criterion=criterion,
            optimizer=optimizer,
            train_loader=train_loader_original,
            val_loader=val_loader_original,
            num_epochs=best_config["num_epochs"],
            patience=best_config["patience"],
            original_model=True,
        )

        accuracy_original, cm_original, metrics_original = evaluate_mlp(
            model_original, val_loader_original
        )

        logging.info(
            f"\nAccuracy with original data: {accuracy_original:.2f}%\n"
            f"\nConfusion matrix for MLP classifier:\n"
            f"{cm_original}\n"
            f"\nMetrics per class:"
            f"\nClass 0 (ham):"
            f"\n  Precision: {metrics_original['class_0']['precision']:.4f}"
            f"\n  Recall: {metrics_original['class_0']['recall']:.4f}"
            f"\n  F1-score: {metrics_original['class_0']['f1_score']:.4f}"
            f"\nClass 1 ({ONE_CLASS}):"
            f"\n  Precision: {metrics_original['class_1']['precision']:.4f}"
            f"\n  Recall: {metrics_original['class_1']['recall']:.4f}"
            f"\n  F1-score: {metrics_original['class_1']['f1_score']:.4f}"
            f"\n\nPonderated metrics:"
            f"\n  Precision: {metrics_original['weighted_avg']['precision']:.4f}"
            f"\n  Recall: {metrics_original['weighted_avg']['recall']:.4f}"
            f"\n  F1-score: {metrics_original['weighted_avg']['f1_score']:.4f}\n"
            f"{DELIMITER}"
        )

        logging.info(f"\nMétricas por classe:")

        plot_mlp_training(
            original_train_loss, original_val_loss, cm_original, MLP_ORIGINAL_PLOT_PATH
        )

        logging.info(
            f"\nTraining and evaluating the MLP with augmented data...\n" f"{DELIMITER}"
        )

        X_train_augmented = torch.tensor(augmented_train_set, dtype=torch.float32).cpu()
        y_train_augmented = torch.tensor(augmented_train_labels, dtype=torch.long).cpu()

        train_dataset_augmented = TensorDataset(X_train_augmented, y_train_augmented)

        train_loader_augmented = create_deterministic_dataloader(
            train_dataset_augmented, MLP_AUGMENTED_BATCH_SIZE
        )

        check_class_distribution(train_loader_augmented, "DataLoader Aumentado")

        model_augmented = MLP(
            input_dim=input_dim,
            hidden_dim=best_config["hidden_dim"],
            output_dim=output_dim,
            l1_lambda=best_config["l1_lambda"],
            l2_lambda=best_config["l2_lambda"],
            dropout=best_config["dropout"],
        ).to(DEVICE)

        optimizer = optim.Adam(
            model_augmented.parameters(),
            lr=best_config["lr"],
            weight_decay=best_config["weight_decay"],
        )

        criterion = nn.CrossEntropyLoss()

        _, augmented_train_loss, augmented_val_loss = train_mlp(
            model=model_augmented,
            criterion=criterion,
            optimizer=optimizer,
            train_loader=train_loader_augmented,
            val_loader=val_loader_original,
            num_epochs=best_config["num_epochs"],
            patience=best_config["patience"],
            original_model=False,
        )

        accuracy_augmented, cm_augmented, metrics_augmented = evaluate_mlp(
            model_augmented, val_loader_original
        )

        logging.info(
            f"\nAccuracy with augmented data: {accuracy_augmented:.2f}%\n"
            f"\nConfusion matrix for MLP classifier:\n"
            f"{cm_augmented}\n"
            f"\nMetrics per class:"
            f"\nClass 0 (ham):"
            f"\n  Precision: {metrics_augmented['class_0']['precision']:.4f}"
            f"\n  Recall: {metrics_augmented['class_0']['recall']:.4f}"
            f"\n  F1-score: {metrics_augmented['class_0']['f1_score']:.4f}"
            f"\nClass 1 ({ONE_CLASS}):"
            f"\n  Precision: {metrics_augmented['class_1']['precision']:.4f}"
            f"\n  Recall: {metrics_augmented['class_1']['recall']:.4f}"
            f"\n  F1-score: {metrics_augmented['class_1']['f1_score']:.4f}"
            f"\n\nPonderated metrics:"
            f"\n  Precision: {metrics_augmented['weighted_avg']['precision']:.4f}"
            f"\n  Recall: {metrics_augmented['weighted_avg']['recall']:.4f}"
            f"\n  F1-score: {metrics_augmented['weighted_avg']['f1_score']:.4f}\n"
            f"{DELIMITER}"
        )

        plot_mlp_training(
            augmented_train_loss,
            augmented_val_loss,
            cm_augmented,
            MLP_AUGMENTED_PLOT_PATH,
        )

        # plot_linear_regression(
        #    augmented_train_set, augmented_train_labels, LR_AUGMENTED_DATA_PLOT_PATH
        # )

        return model_augmented, model_original, accuracy_augmented, accuracy_original
    finally:
        torch.cuda.empty_cache()


def main() -> None:
    try:
        set_seed(23)
        train_dataset, test_dataset, data_tensor, index_tensor = (
            __load_and_preprocess_data()
        )

        train_set = data_tensor[train_dataset.indices]
        test_set = data_tensor[test_dataset.indices]
        train_labels = index_tensor[train_dataset.indices]
        test_labels = index_tensor[test_dataset.indices]

        unique_labels, counts = np.unique(test_labels, return_counts=True)
        class_counts = dict(zip(unique_labels, counts))

        logging.info(
            f"\nDistribuição das classes:\n"
            f"Classe 0 (ham): {class_counts[0]} amostras\n"
            f"Classe 1 ({ONE_CLASS}): {class_counts[1]} amostras\n"
            f"{DELIMITER}"
        )

        minority_class, samples_needed, _ = __identify_minority_class(
            train_labels.cpu().numpy()
        )

        ham_data_original = train_set[train_labels.cpu().numpy() == 0].cpu().numpy()
        phishing_spam_data_original = (
            train_set[train_labels.cpu().numpy() == 1].cpu().numpy()
        )

        plot_feature_distribution(
            ham_data_original,
            phishing_spam_data_original,
            FD_ORIGINAL_DATA_PLOT_PATH,
        )

        generator = __setup_dynamic_gan(
            torch.cat([train_set, test_set]),
            torch.cat([train_labels, test_labels]),
            minority_class,
        )

        augmented_train_set, augmented_train_labels = __generate_and_augment_data(
            generator,
            train_set,
            train_labels,
            minority_class,
            samples_needed,
        )

        ham_data_augmented = augmented_train_set[augmented_train_labels == 0]
        phishing_spam_data_augmented = augmented_train_set[augmented_train_labels == 1]

        plot_feature_distribution(
            ham_data_augmented,
            phishing_spam_data_augmented,
            FD_AUGMENTED_DATA_PLOT_PATH,
        )

        mlp_augmented, mlp_original, accuracy_augmented, accuracy_original = (
            __train_and_evaluate_mlp(
                train_set.cpu().numpy(),
                train_labels.cpu().numpy(),
                augmented_train_set,
                augmented_train_labels,
                test_set.cpu().numpy(),
                test_labels.cpu().numpy(),
            )
        )

        logging.info(
            f"\nComparison of results:\n"
            f"Accuracy with GAN-augmented data: {accuracy_augmented:.2f}%\n"
            f"Accuracy without GAN-augmented data: {accuracy_original:.2f}%\n"
            f"Accuracy gain: {accuracy_augmented - accuracy_original:.2f}%\n"
            f"{DELIMITER}"
        )

        logging.info("Testing with example test set...")
        example_data, example_index, email_paths = get_example_test_set()

        X_example_test = torch.tensor(example_data, dtype=torch.float32).to(DEVICE)

        logging.info("\nResultados com modelo aumentado:")
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

        logging.info("\nResultados com modelo original:")
        predicted_example_label_original, label_probabilities_original = predict_mlp(
            mlp_original, X_example_test
        )

        for i, (_, prob) in enumerate(
            zip(predicted_example_label_original, label_probabilities_original)
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
