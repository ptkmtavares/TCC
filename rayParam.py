import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from ray import tune
from ray.tune import search
from ray.tune.schedulers import ASHAScheduler
from ray.tune.experiment.trial import Trial
from ray.air import session
from mlp import MLP, train_mlp, predict_mlp
from typing import Dict, Any, List
from config import (
    MLP_AUGMENTED_BATCH_SIZE,
    MLP_ORIGINAL_BATCH_SIZE,
    DEVICE,
    ONE_CLASS,
    RAYTUNE_PLOT_PATH,
    NUM_SAMPLES,
)
from plot import plot_ray_results


def __train_mlp_tune(
    config: Dict[str, Any],
    input_dim: int,
    train_loader: DataLoader,
    test_loader: DataLoader,
    example_data: torch.Tensor,
    example_labels: torch.Tensor,
) -> None:
    """Train the MLP model with hyperparameter tuning.

    Args:
        config (Dict[str, Any]): Configuration dictionary containing hyperparameters.
        train_loader (DataLoader): Training data loader.
        test_loader (DataLoader): Test data loader.
        example_data (torch.Tensor): Example data for evaluation.
        example_labels (torch.Tensor): Example labels for evaluation.
    """
    try:

        model = MLP(
            input_dim,
            config["hidden_dim"],
            output_dim=2,
            l1_lambda=config["l1_lambda"],
            l2_lambda=config["l2_lambda"],
            dropout=config["dropout"],
        ).to(DEVICE)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
        )

        _, _, val_losses = train_mlp(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=test_loader,
            num_epochs=config["num_epochs"],
            patience=config["patience"],
            printInfo=False,
        )

        val_loss = val_losses[-1]

        X_example = torch.tensor(example_data, dtype=torch.float32, device=DEVICE)
        y_example = torch.tensor(example_labels, dtype=torch.long, device=DEVICE)
        _, label_probabilities = predict_mlp(model, X_example)
        correct_probabilities = label_probabilities[range(len(y_example)), y_example]
        accuracy = correct_probabilities.mean().item()

        session.report({"accuracy": accuracy, "val_loss": val_loss})

    finally:
        del model, X_example, y_example
        torch.cuda.empty_cache()


def __trial_dirname_creator(trial: Trial) -> str:
    """Create a directory name for the trial.

    Args:
        trial (Trial): The trial object.

    Returns:
        str: The directory name for the trial.
    """
    return f"trial_{trial.trial_id}"


def get_hyperparameters(
    input_dim: int,
    train_loader: DataLoader,
    test_loader: DataLoader,
    example_data: List[List[int]],
    example_labels: List[int],
    config: str = ONE_CLASS,
) -> Dict[str, Any]:
    """Get hyperparameters for tuning.

    Args:
        train_set (np.ndarray): Training dataset.
        test_set (np.ndarray): Test dataset.
        train_labels (np.ndarray): Training labels.
        test_labels (np.ndarray): Test labels.
        example_data (List[List[int]]): Example data for evaluation.
        example_labels (List[int]]): Example labels for evaluation.
        config (str, optional): Configuration type. Defaults to None.

    Returns:
        Dict[str, Any]: Best configuration found during tuning.
    """
    if config == "phishing":
        config = {
            "l1_lambda": tune.loguniform(1e-5, 2e-4),
            "l2_lambda": tune.loguniform(1e-6, 1e-4),
            "hidden_dim": tune.choice([128, 256]),
            "lr": tune.loguniform(1e-5, 1e-4),
            "weight_decay": tune.loguniform(1e-5, 1e-4),
            "num_epochs": tune.lograndint(200, 1000),
            "patience": 50,
            "dropout": tune.uniform(0.2, 0.3),
        }
    elif config == "spam":
        config = {
            "l1_lambda": tune.loguniform(1e-5, 1e-4),
            "l2_lambda": tune.loguniform(1e-6, 1e-5),
            "hidden_dim": tune.choice([256, 512]),
            "lr": tune.loguniform(1e-4, 1e-3),
            "weight_decay": tune.loguniform(1e-5, 1e-4),
            "num_epochs": tune.lograndint(200, 1000),
            "patience": 50,
            "dropout": tune.uniform(0.2, 0.3),
        }

    scheduler = ASHAScheduler(
        metric="accuracy",
        mode="max",
        max_t=200,
        grace_period=20,
        reduction_factor=3,
        brackets=3,
    )

    search_alg = search.BasicVariantGenerator(max_concurrent=4)

    try:
        analysis = tune.run(
            tune.with_parameters(
                __train_mlp_tune,
                input_dim=input_dim,
                train_loader=train_loader,
                test_loader=test_loader,
                example_data=example_data,
                example_labels=example_labels,
            ),
            resources_per_trial={"cpu": 1, "gpu": 1},
            config=config,
            num_samples=NUM_SAMPLES,
            scheduler=scheduler,
            search_alg=search_alg,
            trial_dirname_creator=__trial_dirname_creator,
            verbose=1,
        )
        plot_ray_results(analysis, RAYTUNE_PLOT_PATH)
    except Exception as e:
        logging.error(f"Error during hyperparameter tuning: {e}")
        raise

    return analysis.get_best_config(metric="val_loss", mode="min")
