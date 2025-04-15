"""
Comprehensive example script for the optimized MCPC classifier.

This script demonstrates all the features of the optimized MCPC classifier,
including:
- Different classifier types (PCC, EPCC, MCPCC)
- Different norm types (L1, L2)
- Performance metrics
- Cross-validation
- Model saving and loading
- Visualization (if data is 2D)
"""

import os
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification, make_moons, make_circles

from mcpcc import MCPCConfig, MCPCClassifier
from visualize import (
    plot_decision_boundary,
    plot_metrics_comparison,
    plot_training_time_comparison,
)

# Create directories for outputs
os.makedirs("./models", exist_ok=True)
os.makedirs("./results", exist_ok=True)


def run_classifier_comparison(X, y, dataset_name):
    """Run a comparison of different classifier configurations."""
    print(f"\n{'='*50}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*50}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Create different classifier configurations
    configs = {
        "PCC (L1)": MCPCConfig(
            classifier_type="pcc",
            norm_type="L1",
            penalty_coefficient=10.0,
            verbose=True,
            use_cache=True,
        ),
        "PCC (L2)": MCPCConfig(
            classifier_type="pcc",
            norm_type="L2",
            penalty_coefficient=10.0,
            verbose=True,
            use_cache=True,
        ),
        "EPCC (L1)": MCPCConfig(
            classifier_type="epcc",
            norm_type="L1",
            penalty_coefficient=10.0,
            verbose=True,
            use_cache=True,
        ),
        "EPCC (L2)": MCPCConfig(
            classifier_type="epcc",
            norm_type="L2",
            penalty_coefficient=10.0,
            verbose=True,
            use_cache=True,
        ),
        "MCPCC (L1, 5 centers)": MCPCConfig(
            classifier_type="mcpcc",
            norm_type="L1",
            num_centers=5,
            penalty_coefficient=10.0,
            verbose=True,
            use_cache=True,
        ),
        "MCPCC (L2, 5 centers)": MCPCConfig(
            classifier_type="mcpcc",
            norm_type="L2",
            num_centers=5,
            penalty_coefficient=10.0,
            verbose=True,
            use_cache=True,
        ),
        "MCPCC (L2, 10 centers)": MCPCConfig(
            classifier_type="mcpcc",
            norm_type="L2",
            num_centers=10,
            penalty_coefficient=10.0,
            verbose=True,
            use_cache=True,
        ),
    }

    # Train and evaluate classifiers
    classifiers = {}
    results = {}

    for name, config in configs.items():
        print(f"\nTraining {name}...")
        start_time = time.time()

        # Create and train classifier
        clf = MCPCClassifier(config)

        # Perform cross-validation
        cv_results = clf.cross_validate(X_train, y_train, cv=5)
        print(
            f"Cross-validation mean accuracy: {cv_results['mean_accuracy']:.4f} ± {cv_results['std_accuracy']:.4f}"
        )

        # Train on full training set
        clf.fit(X_train, y_train)

        # Evaluate on test set
        evaluation = clf.evaluate(X_test, y_test)
        results[name] = evaluation
        classifiers[name] = clf

        # Save model
        model_path = (
            f"./models/{dataset_name}_{name.replace(' ', '_').replace(',', '')}.pkl"
        )
        clf.save(model_path)

        # Print results
        print(f"Training completed in {time.time() - start_time:.2f} seconds")
        print("Test set evaluation:")
        for metric, value in evaluation.items():
            print(f"  {metric.capitalize()}: {value:.4f}")

    # Print comparison table
    print("\nComparison Table:")
    print(
        f"{'Classifier':<25} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}"
    )
    print("-" * 65)
    for name, metrics in results.items():
        print(
            f"{name:<25} {metrics['accuracy']:<10.4f} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} {metrics['f1']:<10.4f}"
        )

    # Visualize results if data is 2D
    if X.shape[1] == 2:
        print("\nGenerating visualizations...")

        # Plot decision boundaries
        for name, clf in classifiers.items():
            plot_decision_boundary(
                clf,
                X_test,
                y_test,
                title=f"Decision Boundary - {name}",
                save_path=f"./results/{dataset_name}_{name.replace(' ', '_').replace(',', '')}_boundary.png",
            )

        # Compare metrics
        plot_metrics_comparison(
            classifiers,
            X_test,
            y_test,
            save_path=f"./results/{dataset_name}_metrics_comparison.png",
        )

        # Compare training times
        plot_training_time_comparison(
            configs,
            X_train,
            y_train,
            save_path=f"./results/{dataset_name}_training_time_comparison.png",
        )

    return classifiers, results


def main():
    """Main function to run the example."""
    # Start timing
    start_time = time.time()

    # Generate synthetic datasets
    print("Generating synthetic datasets...")

    # 1. Linearly separable data
    X_linear, y_linear = make_classification(
        n_samples=1000,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        n_clusters_per_class=1,
        class_sep=1.0,
        random_state=42,
    )

    # 2. Moons dataset (non-linear)
    X_moons, y_moons = make_moons(n_samples=1000, noise=0.1, random_state=42)

    # 3. Circles dataset (non-linear)
    X_circles, y_circles = make_circles(
        n_samples=1000, noise=0.1, factor=0.5, random_state=42
    )

    # 4. Higher dimensional data
    X_high_dim, y_high_dim = make_classification(
        n_samples=1000,
        n_features=10,
        n_redundant=2,
        n_informative=8,
        n_clusters_per_class=1,
        class_sep=1.0,
        random_state=42,
    )

    # Run comparisons on each dataset
    datasets = [
        (X_linear, y_linear, "linear"),
        (X_moons, y_moons, "moons"),
        (X_circles, y_circles, "circles"),
        (X_high_dim, y_high_dim, "high_dimensional"),
    ]

    all_results = {}
    for X, y, name in datasets:
        _, results = run_classifier_comparison(X, y, name)
        all_results[name] = results

    # Print overall summary
    print("\n" + "=" * 50)
    print("Overall Summary")
    print("=" * 50)

    for dataset_name, results in all_results.items():
        print(f"\nDataset: {dataset_name}")
        best_classifier = max(results.items(), key=lambda x: x[1]["accuracy"])
        print(
            f"Best classifier: {best_classifier[0]} (Accuracy: {best_classifier[1]['accuracy']:.4f})"
        )

    print(f"\nTotal execution time: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
