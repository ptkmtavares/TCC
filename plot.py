import matplotlib.pyplot as plt
from ray.tune.analysis import ExperimentAnalysis
from typing import List
from config import FEATURES


def plot_feature_distribution(features: List[List[int]], output_path: str) -> None:
    """Plots the distribution of features and saves as an SVG file.

    Args:
        features (List[List[int]]): The list of feature arrays.
        output_path (str): The path to save the SVG file.
    """
    feature_counts = [sum(feature) for feature in zip(*features)]
    feature_names = FEATURES

    plt.figure(figsize=(10, 8))
    plt.barh(feature_names, feature_counts, color="skyblue")
    plt.xlabel("Count")
    plt.title("Feature Distribution")
    plt.tight_layout()
    plt.savefig(output_path, format="svg", transparent=True)
    plt.close()


def plot_ray_results(analysis: ExperimentAnalysis, output_path: str) -> None:
    """Plot the results of the hyperparameter tuning.

    Args:
        analysis (ExperimentAnalysis): The analysis object returned by tune.run.
        output_path (str): The path to save the SVG file.
    """
    df = analysis.results_df
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(df["accuracy"], label="Accuracy")
    plt.xlabel("Trial")
    plt.ylabel("Accuracy")
    plt.title("Accuracy over Trials")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(df["val_loss"], label="Validation Loss")
    plt.xlabel("Trial")
    plt.ylabel("Validation Loss")
    plt.title("Validation Loss over Trials")
    plt.legend()

    plt.savefig(output_path, format="svg", transparent=True)
    plt.close()


def plot_mlp_training(
    train_losses: list, val_losses: list, output_path: str = "training_plot.svg"
) -> None:
    """
    Plots the training and validation loss over epochs and saves the plot as an SVG file.

    Args:
        train_losses (list): List of training losses.
        val_losses (list): List of validation losses.
        output_path (str, optional): Path to save the SVG file. Defaults to "training_plot.svg".
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
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
