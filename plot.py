import matplotlib.pyplot as plt
import numpy as np
from ray.tune.analysis import ExperimentAnalysis
from typing import List
from config import FEATURES, MLP_ORIGINAL_PLOT_PATH
from pandas import DataFrame


def plot_feature_distribution(
    ham_features: np.ndarray, phishing_features: np.ndarray, output_path: str
) -> None:
    """Plots the distribution of ham and phishing features and saves as an SVG file.

    Args:
        ham_features (np.ndarray): The list of ham feature arrays.
        spam_features (np.ndarray): The list of phishing feature arrays.
        output_path (str): The path to save the SVG file.
    """
    ham_feature_counts = np.sum(np.abs(ham_features), axis=0)
    phishing_feature_counts = np.sum(np.abs(phishing_features), axis=0)
    total_ham = len(ham_features)
    total_phishing = len(phishing_features)
    feature_names = FEATURES

    _, axs = plt.subplots(1, 2, figsize=(20, 8), sharey=True)

    axs[0].barh(feature_names, ham_feature_counts, color="skyblue")
    axs[0].set_xlabel("Count")
    axs[0].set_title(
        f"{'Augmented ' if total_ham == total_phishing else 'Original '}Ham Feature Distribution (Total:{total_ham})"
    )

    axs[1].barh(feature_names, phishing_feature_counts, color="salmon")
    axs[1].set_xlabel("Count")
    axs[1].set_title(
        f"{'Augmented ' if total_ham == total_phishing else 'Original '}Phishing Feature Distribution (Total:{total_phishing})"
    )

    plt.tight_layout()
    plt.savefig(output_path, format="svg", transparent=True)
    plt.close()


def plot_ray_results(analysis: ExperimentAnalysis, output_path: str) -> None:
    """Plot the results of the hyperparameter tuning.

    Args:
        analysis (ExperimentAnalysis): The analysis object returned by tune.run.
        output_path (str): The path to save the SVG file.
    """
    best_trial = analysis.get_best_trial("val_loss", "min", "last")
    df: DataFrame = analysis.results_df
    dropped_trials: List[str] = []
    for trial in analysis.trials:
        if trial.last_result["val_loss"] > 0.56:
            dropped_trials.append(trial.trial_id)

        config = trial.config
        for param, value in config.items():
            df.loc[trial.trial_id, param] = value

    plot_list = [
        "accuracy",
        "val_loss",
        "l1_lambda",
        "l2_lambda",
        "hidden_dim1",
        "hidden_dim2",
        "lr",
        "weight_decay",
        "num_epochs",
        "patience",
        "dropout",
    ]

    num_plots = len(plot_list)
    plt.figure(figsize=(15, 5 * num_plots))
    for i, param in enumerate(plot_list, start=1):
        plt.subplot(num_plots, 1, i)
        plt.plot(df[param], label=param)
        plt.xticks(rotation=90)
        plt.xlabel("Trial")
        plt.ylabel(param)
        plt.title(f"{param} over Trials")
        plt.legend()
        plt.grid(axis="x", linestyle="--")
        plt.axvline(x=best_trial.trial_id, color="green", linestyle="--")
        for trial_id in dropped_trials:
            plt.axvline(x=trial_id, color="red", linestyle="--")

    plt.tight_layout()
    plt.savefig(output_path, format="svg", transparent=True)
    plt.close()


def plot_mlp_training(
    train_losses: list,
    val_losses: list,
    cm: np.ndarray,
    output_path: str = "training_plot.svg",
) -> None:
    """
    Plots the training and validation loss over epochs and saves the plot as an SVG file.

    Args:
        train_losses (list): List of training losses.
        val_losses (list): List of validation losses.
        cm (np.ndarray): Confusion matrix.
        output_path (str, optional): Path to save the SVG file. Defaults to "training_plot.svg".
    """
    fig, axs = plt.subplots(1, 2, figsize=(20, 6))
    class_names = ["Ham", "Phishing"]

    axs[0].plot(train_losses, label="Training Loss")
    axs[0].plot(val_losses, label="Validation Loss")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss")
    axs[0].set_ylim(0.0, 0.8)
    axs[0].set_title(
        f"{'Original ' if MLP_ORIGINAL_PLOT_PATH == output_path else 'Augmented '}Training and Validation Loss"
    )
    axs[0].legend()

    cax = axs[1].matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax, ax=axs[1])
    axs[1].set_xlabel("Predicted")
    axs[1].set_ylabel("True")
    axs[1].set_title(
        f"{'Original ' if MLP_ORIGINAL_PLOT_PATH == output_path else 'Augmented '}Confusion Matrix"
    )
    axs[1].set_xticks(np.arange(len(class_names)))
    axs[1].set_yticks(np.arange(len(class_names)))
    axs[1].set_xticklabels(class_names)
    axs[1].set_yticklabels(class_names)

    plt.setp(axs[1].get_xticklabels(), ha="center")

    for (i, j), val in np.ndenumerate(cm):
        color = "white" if cm[i, j] > cm.max() / 2 else "black"
        axs[1].text(
            j,
            i,
            f"{val}",
            ha="center",
            va="center",
            color=color,
            fontsize=12,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(output_path, format="svg", transparent=True)
    plt.close()


def plot_gan_losses(
    d_losses: list, g_losses: list, save_path: str = "training_losses.svg"
) -> None:
    """
    Plot and save the training losses of the discriminator and generator.

    Args:
        d_losses (list): List of discriminator losses.
        g_losses (list): List of generator losses.
        save_path (str, optional): Path to save the SVG plot. Defaults to "training_losses.svg".
    """
    plt.figure(figsize=(10, 5))
    plt.plot(d_losses, label="Discriminator Loss")
    plt.plot(g_losses, label="Generator Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training Losses")
    plt.savefig(save_path, format="svg", transparent=True)
    plt.close()
