import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from ray.tune.analysis import ExperimentAnalysis
from typing import List
from config import (
    FEATURES,
    GAN_PLOT_PATH,
    MLP_ORIGINAL_PLOT_PATH,
    ONE_CLASS,
    PLOT_DIR,
    RAYTUNE_PLOT_PATH,
)
from pandas import DataFrame

matplotlib.use("Agg")


def plot_feature_distribution(
    ham_features: np.ndarray,
    phishing_spam_features: np.ndarray,
    output_path: str = PLOT_DIR + "fd.svg",
) -> None:
    """Plots the distribution of ham and phishing features and saves as an SVG file.

    Args:
        ham_features (np.ndarray): The list of ham feature arrays.
        phishing_spam_features (np.ndarray): The list of phishing/spam feature arrays.
        output_path (str): The path to save the SVG file.
    """
    feature_names = FEATURES
    dataset_type = "Augmented " if len(ham_features) == len(phishing_spam_features) else "Original "

    ham_counts = np.array([
        [np.sum(ham_features[:, i] == -1), np.sum(ham_features[:, i] == 0), np.sum(ham_features[:, i] == 1)]
        for i in range(ham_features.shape[1])
    ])
    phishing_spam_counts = np.array([
        [np.sum(phishing_spam_features[:, i] == -1), np.sum(phishing_spam_features[:, i] == 0), np.sum(phishing_spam_features[:, i] == 1)]
        for i in range(phishing_spam_features.shape[1])
    ])

    bar_width = 0.25
    r1 = np.arange(len(feature_names))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]

    _, axs = plt.subplots(1, 2, figsize=(20, 12), sharey=True)

    axs[0].barh(r1, ham_counts[:, 0], color='red', height=bar_width, label='-1')
    axs[0].barh(r2, ham_counts[:, 1], color='blue', height=bar_width, label='0')
    axs[0].barh(r3, ham_counts[:, 2], color='green', height=bar_width, label='1')
    axs[0].set_yticks([r + bar_width for r in range(len(feature_names))])
    axs[0].set_yticklabels(feature_names)
    axs[0].set_xlabel("Count")
    axs[0].set_title(f"{dataset_type}Ham Feature Distribution (Total:{len(ham_features)})")

    axs[1].barh(r1, phishing_spam_counts[:, 0], color='red', height=bar_width, label='-1')
    axs[1].barh(r2, phishing_spam_counts[:, 1], color='blue', height=bar_width, label='0')
    axs[1].barh(r3, phishing_spam_counts[:, 2], color='green', height=bar_width, label='1')
    axs[1].set_yticks([r + bar_width for r in range(len(feature_names))])
    axs[1].set_yticklabels(feature_names)
    axs[1].set_xlabel("Count")
    axs[1].set_title(f"{dataset_type + ONE_CLASS.title()} Feature Distribution (Total:{len(phishing_spam_features)})")

    plt.tight_layout(pad=3.0, w_pad=0.5)
    plt.legend(loc='upper right')
    plt.savefig(output_path, format="svg", transparent=True, bbox_inches="tight")
    plt.close()


def plot_ray_results(
    analysis: ExperimentAnalysis, output_path: str = RAYTUNE_PLOT_PATH
) -> None:
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
        "hidden_dim",
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
    output_path: str = PLOT_DIR + "mlp.svg",
) -> None:
    """
    Plots the training and validation loss over epochs and saves the plot as an SVG file.

    Args:
        train_losses (list): List of training losses.
        val_losses (list): List of validation losses.
        cm (np.ndarray): Confusion matrix.
        output_path (str, optional): Path to save the SVG file.
    """
    fig, axs = plt.subplots(1, 2, figsize=(20, 6))
    class_names = ["Ham", ONE_CLASS.title()]

    axs[0].plot(train_losses, label="Training Loss")
    axs[0].plot(val_losses, label="Validation Loss")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss")
    axs[0].set_ylim(0.0, 0.4)
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
    d_losses: list, g_losses: list, output_path: str = GAN_PLOT_PATH
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
    plt.axhline(y=0.0, color="r", linestyle="--", label="Ideal Convergence")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training Losses")
    plt.savefig(output_path, format="svg", transparent=True)
    plt.close()

def plot_linear_regression(
    x: np.ndarray, 
    y: np.ndarray, 
    y_pred: np.ndarray,
    output_path: str = PLOT_DIR + "linear_regression.svg"
) -> None:
    """
    Plota os resultados da regressão linear e salva como arquivo SVG.

    Args:
        x (np.ndarray): Valores do eixo x (features)
        y (np.ndarray): Valores reais (ground truth)
        y_pred (np.ndarray): Valores preditos pelo modelo
        output_path (str): Caminho para salvar o arquivo SVG
    """
    plt.figure(figsize=(10, 6))
    
    # Plot dos pontos reais
    plt.scatter(x, y, color='blue', alpha=0.5, label='Dados Reais')
    
    # Plot da linha de regressão
    plt.plot(x, y_pred, color='red', label='Linha de Regressão')
    
    plt.xlabel('Features')
    plt.ylabel('Valores')
    plt.title('Regressão Linear')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path, format="svg", transparent=True)
    plt.close()