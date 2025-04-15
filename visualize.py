"""
Visualization utilities for the MCPC classifier.

This module provides functions to visualize the decision boundaries and centers
of the MCPC classifier.
"""

import time
from typing import Optional, Tuple

import numpy as np
import matplotlib

# Use a non-interactive backend to avoid warnings
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap

from mcpcc import MCPCClassifier


def plot_decision_boundary(
    classifier: MCPCClassifier,
    X: np.ndarray,
    y: np.ndarray,
    title: str = "Decision Boundary",
    feature_names: Optional[Tuple[str, str]] = None,
    resolution: float = 0.02,
    figsize: Tuple[int, int] = (10, 8),
    show_centers: bool = True,
    save_path: Optional[str] = None,
):
    """
    Plot the decision boundary of a trained MCPC classifier.

    Parameters
    ----------
    classifier : MCPCClassifier
        A trained MCPC classifier.
    X : np.ndarray
        The feature matrix (must be 2D for visualization).
    y : np.ndarray
        The target vector.
    title : str, optional
        The title of the plot, by default "Decision Boundary".
    feature_names : Optional[Tuple[str, str]], optional
        Names of the two features, by default None.
    resolution : float, optional
        The resolution of the grid, by default 0.02.
    figsize : Tuple[int, int], optional
        The size of the figure, by default (10, 8).
    show_centers : bool, optional
        Whether to show the centers, by default True.
    save_path : Optional[str], optional
        Path to save the figure, by default None.
    """
    if X.shape[1] != 2:
        raise ValueError("This function only works for 2D feature spaces")

    # Create color maps
    cmap_light = ListedColormap(["#FFAAAA", "#AAFFAA"])
    cmap_bold = ListedColormap(["#FF0000", "#00FF00"])

    # Create a mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, resolution), np.arange(y_min, y_max, resolution)
    )

    # Predict on the mesh grid
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Create the plot
    plt.figure(figsize=figsize)
    plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolors="k", s=20)

    # Plot centers if requested
    if show_centers and classifier.centers is not None:
        plt.scatter(
            classifier.centers[:, 0],
            classifier.centers[:, 1],
            c="black",
            marker="*",
            s=200,
            label="Centers",
        )
        plt.legend()

    # Set title and labels
    plt.title(title, fontsize=14)
    if feature_names:
        plt.xlabel(feature_names[0], fontsize=12)
        plt.ylabel(feature_names[1], fontsize=12)
    else:
        plt.xlabel("Feature 1", fontsize=12)
        plt.ylabel("Feature 2", fontsize=12)

    plt.tight_layout()

    # Save the figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        print("Warning: No save_path provided, figure will not be displayed or saved.")

    plt.close()


def plot_metrics_comparison(
    classifiers: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None,
):
    """
    Compare multiple classifiers using various metrics.

    Parameters
    ----------
    classifiers : dict
        A dictionary of classifiers, where keys are names and values are trained classifiers.
    X_test : np.ndarray
        The test feature matrix.
    y_test : np.ndarray
        The test target vector.
    figsize : Tuple[int, int], optional
        The size of the figure, by default (12, 6).
    save_path : Optional[str], optional
        Path to save the figure, by default None.
    """
    metrics = ["accuracy", "precision", "recall", "f1"]
    results = {}

    # Evaluate each classifier
    for name_, clf_ in classifiers.items():
        evaluation = clf_.evaluate(X_test, y_test)
        results[name_] = [evaluation[metric] for metric in metrics]

    # Create the plot
    plt.figure(figsize=figsize)
    x = np.arange(len(metrics))
    width = 0.8 / len(classifiers)
    offset = -width * (len(classifiers) - 1) / 2

    # Plot bars for each classifier
    for i, (name, values) in enumerate(results.items()):
        plt.bar(x + offset + i * width, values, width, label=name)

    # Set labels and title
    plt.xlabel("Metrics", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.title("Classifier Performance Comparison", fontsize=14)
    plt.xticks(x, metrics)
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Save the figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        print("Warning: No save_path provided, figure will not be displayed or saved.")

    plt.close()


def plot_training_time_comparison(
    classifiers_config: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
):
    """
    Compare training times for different classifier configurations.

    Parameters
    ----------
    classifiers_config : dict
        A dictionary of classifier configurations, where keys are names and values are MCPCConfig objects.
    X_train : np.ndarray
        The training feature matrix.
    y_train : np.ndarray
        The training target vector.
    figsize : Tuple[int, int], optional
        The size of the figure, by default (10, 6).
    save_path : Optional[str], optional
        Path to save the figure, by default None.
    """
    times = []
    names = []

    # Train each classifier and measure time
    for name_, config_ in classifiers_config.items():
        start_time = time.time()
        clf_ = MCPCClassifier(config_)
        clf_.fit(X_train, y_train)
        training_time = time.time() - start_time
        times.append(training_time)
        names.append(name_)

    # Create the plot
    plt.figure(figsize=figsize)
    plt.barh(names, times, color="skyblue")
    plt.xlabel("Training Time (seconds)", fontsize=12)
    plt.title("Training Time Comparison", fontsize=14)
    plt.grid(axis="x", linestyle="--", alpha=0.7)

    # Add time values at the end of each bar
    for i, time_val in enumerate(times):
        plt.text(time_val + 0.1, i, f"{time_val:.2f}s", va="center")

    # Save the figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        print("Warning: No save_path provided, figure will not be displayed or saved.")

    plt.tight_layout()
    plt.close()


if __name__ == "__main__":

    from mcpcc import MCPCConfig

    # Load data
    data = np.load("./data/chessboard.npz")
    X_train = data["train_features"]
    y_train = data["train_labels"]
    X_test = data["test_features"]
    y_test = data["test_labels"]

    # Check if data is 2D for visualization
    if X_train.shape[1] == 2:
        # Create different classifier configurations
        configs = {
            "PCC (L2)": MCPCConfig(
                classifier_type="pcc", norm_type="L2", verbose=False
            ),
            "EPCC (L2)": MCPCConfig(
                classifier_type="epcc", norm_type="L2", verbose=False
            ),
            "MCPCC (L2, 10 centers)": MCPCConfig(
                classifier_type="mcpcc",
                norm_type="L2",
                num_centers=10,
                verbose=False,
            ),
        }

        # Train classifiers
        classifiers = {}
        for name, config in configs.items():
            clf = MCPCClassifier(config)
            clf.fit(X_train, y_train)
            classifiers[name] = clf

        # Plot decision boundaries
        for name, clf in classifiers.items():
            plot_decision_boundary(
                clf,
                X_test,
                y_test,
                title=f"Decision Boundary - {name}",
                save_path=f"./results/{name.replace(' ', '_').replace(',', '')}_boundary.png",
            )

        # Compare metrics
        plot_metrics_comparison(
            classifiers,
            X_test,
            y_test,
            save_path="./results/metrics_comparison.png",
        )

        # Compare training times
        plot_training_time_comparison(
            configs,
            X_train,
            y_train,
            save_path="./results/training_time_comparison.png",
        )
    else:
        print(
            f"Data has {X_train.shape[1]} dimensions, visualization requires 2D data."
        )
