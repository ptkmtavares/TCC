import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.experiment.trial import Trial
from ray.air import session
from mlp import MLP, train_mlp, predict_mlp
from typing import Dict, Any, List
from config import DEVICE, RAYTUNE_PLOT_PATH
from plot import plot_ray_results


def __train_mlp_tune(
    config: Dict[str, Any],
    train_data: torch.Tensor,
    train_labels: torch.Tensor,
    test_data: torch.Tensor,
    test_labels: torch.Tensor,
    example_data: torch.Tensor,
    example_labels: torch.Tensor,
) -> None:
    """Train the MLP model with hyperparameter tuning.

    Args:
        config (Dict[str, Any]): Configuration dictionary containing hyperparameters.
        train_data (torch.Tensor): Training data.
        train_labels (torch.Tensor): Training labels.
        test_data (torch.Tensor): Test data.
        test_labels (torch.Tensor): Test labels.
        example_data (torch.Tensor): Example data for evaluation.
        example_labels (torch.Tensor): Example labels for evaluation.
    """
    try:
        X_train = torch.tensor(train_data, dtype=torch.float32, device=DEVICE)
        y_train = torch.tensor(train_labels, dtype=torch.long, device=DEVICE)
        X_test = torch.tensor(test_data, dtype=torch.float32, device=DEVICE)
        y_test = torch.tensor(test_labels, dtype=torch.long, device=DEVICE)
        X_example = torch.tensor(example_data, dtype=torch.float32, device=DEVICE)
        y_example = torch.tensor(example_labels, dtype=torch.long, device=DEVICE)

        input_dim = train_data.shape[1]
        hidden_dim1 = config["hidden_dim1"]
        hidden_dim2 = config["hidden_dim2"]
        output_dim = 2

        model = MLP(
            input_dim,
            hidden_dim1,
            hidden_dim2,
            output_dim,
            l1_lambda=config["l1_lambda"],
            l2_lambda=config["l2_lambda"],
            dropout=config["dropout"],
        ).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
        )

        val_loss, _, _ = train_mlp(
            model,
            criterion,
            optimizer,
            X_train,
            y_train,
            X_test,
            y_test,
            num_epochs=config["num_epochs"],
            patience=config["patience"],
            printInfo=False,
        )

        _, label_probabilities = predict_mlp(model, X_example)

        correct_probabilities = label_probabilities[range(len(y_example)), y_example]
        accuracy = correct_probabilities.mean().item()

        session.report({"accuracy": accuracy, "val_loss": val_loss})

    finally:
        del model, X_train, y_train, X_test, y_test, X_example, y_example
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
    train_set: np.ndarray,
    test_set: np.ndarray,
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    example_data: List[List[int]],
    example_labels: List[int],
    config: str = None,
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
            "l1_lambda": tune.loguniform(1e-4, 1.2e-4),
            "l2_lambda": tune.loguniform(4.5e-5, 5.5e-5),
            "hidden_dim1": tune.choice([32, 128]),
            "hidden_dim2": tune.choice([64]),
            "lr": tune.loguniform(2e-5, 2.8e-5),
            "weight_decay": tune.loguniform(4e-4, 6e-4),
            "num_epochs": tune.lograndint(3500, 4500),
            "patience": tune.randint(100, 200),
            "dropout": tune.uniform(0.27, 0.28),
        }

    scheduler = ASHAScheduler(
        metric="accuracy", mode="max", max_t=200, grace_period=20, reduction_factor=3
    )

    try:
        analysis = tune.run(
            tune.with_parameters(
                __train_mlp_tune,
                train_data=train_set,
                train_labels=train_labels,
                test_data=test_set,
                test_labels=test_labels,
                example_data=example_data,
                example_labels=example_labels,
            ),
            resources_per_trial={"cpu": 1, "gpu": 1},
            config=config,
            num_samples=30,
            scheduler=scheduler,
            trial_dirname_creator=__trial_dirname_creator,
            verbose=1,
        )
        plot_ray_results(analysis, RAYTUNE_PLOT_PATH)
    except Exception as e:
        logging.error(f"Error during hyperparameter tuning: {e}")
        raise

    return analysis.get_best_config(metric="val_loss", mode="min")
