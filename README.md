# Multi-Center Polyhedral Classifier (MCPCC)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

The Multi-Center Polyhedral Classifier (MCPCC) is a high-performance machine learning classifier that utilizes multiple centers to define decision boundaries in a multi-dimensional space. The classifier is based on the Polyhedral Conic Classifier (PCC) and the Extended Polyhedral Conic Classifier (EPCC), which define decision boundaries using polyhedral cones. MCPCC extends these classifiers by allowing multiple centers to define the decision boundaries, significantly improving performance on complex, non-linear datasets.

This optimized implementation provides excellent performance, extensive configuration options, and comprehensive evaluation tools. It's designed to be both easy to use and highly efficient, with features like caching, parallel processing, and model serialization.

The below image shows a sample classification using the MCPCC classifier. Black points represent the calculated centers and the colored points represent the data points. The decision boundaries are determined by the polyhedral conic functions using the centers.

![A sample classification](./resources/mcpcc_1.png)

Refer to the [MCPCC paper](https://dergipark.org.tr/tr/download/article-file/1307973) for more details on the theoretical foundation.

For more information on [Polyhedral Conic Classifiers](https://ieeexplore.ieee.org/document/8798888) [(alternative link)](https://web.ogu.edu.tr/Storage/mlcv/Uploads/pcc_19.pdf)

![Polyhedral Conic Classifier](./resources/epcc_1.png)

## Features

### Core Functionality

- **Multiple Classifier Types**: Supports PCC, EPCC, and MCPCC algorithms
- **Configurable Distance Metrics**: Choose between L1 (Manhattan) and L2 (Euclidean) norms
- **Flexible Configuration**: Fine-tune penalty coefficients, number of centers, and other parameters

### Performance Optimizations

- **Efficient Caching**: Reduces redundant computations for feature arrangements and distances
- **Optimized Clustering**: Improved KMeans implementation with MiniBatchKMeans support for large datasets
- **Vectorized Operations**: Leverages NumPy's vectorized operations for maximum performance

### Advanced Features

- **Model Serialization**: Save and load trained models
- **Cross-Validation**: Built-in cross-validation support
- **Multiple Metrics**: Comprehensive evaluation with accuracy, precision, recall, and F1 score
- **Visualization Tools**: Plot decision boundaries and compare classifier performance

## Installation

To install the required dependencies, run the following command:

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from mcpcc import MCPCConfig, MCPCClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
import numpy as np

# Generate sample data
X, y = make_moons(n_samples=1000, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Configure and train the classifier
config = MCPCConfig(
    classifier_type="mcpcc",  # Options: "pcc", "epcc", "mcpcc"
    norm_type="L2",          # Options: "L1", "L2"
    num_centers=5,           # Number of centers (for mcpcc)
    penalty_coefficient=10.0,
    use_cache=True,          # Enable caching for better performance
    verbose=True
)

# Create and train the classifier
classifier = MCPCClassifier(config)
classifier.fit(X_train, y_train)

# Evaluate on test data
evaluation = classifier.evaluate(X_test, y_test)
print("Test set evaluation:")
for metric, value in evaluation.items():
    print(f"{metric.capitalize()}: {value:.4f}")

# Make predictions
y_pred = classifier.predict(X_test)

# Save the trained model
classifier.save("./models/my_model.pkl")

# Load the model later
loaded_classifier = MCPCClassifier.load("./models/my_model.pkl")
```

## Advanced Usage

### Cross-Validation

```python
# Perform cross-validation
cv_results = classifier.cross_validate(X_train, y_train, cv=5)
print(f"Mean accuracy: {cv_results['mean_accuracy']:.4f} ± {cv_results['std_accuracy']:.4f}")
```

### Visualization

```python
from visualize import plot_decision_boundary, plot_metrics_comparison

# Plot decision boundary
plot_decision_boundary(
    classifier, X_test, y_test,
    title="MCPCC Decision Boundary",
    save_path="./results/decision_boundary.png"
)

# Compare multiple classifiers
classifiers = {
    "PCC (L2)": pcc_classifier,
    "EPCC (L2)": epcc_classifier,
    "MCPCC (L2, 5 centers)": mcpcc_classifier
}
plot_metrics_comparison(classifiers, X_test, y_test, save_path="./results/comparison.png")
```

### Command-Line Interface

The package also provides a command-line interface:

```bash
python mcpcc.py --data_path ./data/chessboard.npz \
               --classifier_type mcpcc \
               --norm_type L2 \
               --penalty_coefficient 50.0 \
               --num_centers 10 \
               --verbose \
               --use_cache \
               --cross_validate \
               --model_path ./models/my_model.pkl
```

## Performance Comparison

The optimized MCPCC implementation shows significant performance improvements over traditional classifiers, especially on non-linear datasets:

| Dataset | Classifier | Accuracy | Precision | Recall | F1 Score |
|---------|------------|----------|-----------|--------|----------|
| Moons   | PCC (L2)   | 0.8933   | 0.9118    | 0.8611 | 0.8857   |
| Moons   | EPCC (L2)  | 0.8967   | 0.8599    | 0.9375 | 0.8970   |
| Moons   | MCPCC (L2) | 1.0000   | 1.0000    | 1.0000 | 1.0000   |
| Circles | PCC (L2)   | 0.9867   | 0.9795    | 0.9931 | 0.9862   |
| Circles | EPCC (L2)  | 0.9933   | 0.9931    | 0.9931 | 0.9931   |
| Circles | MCPCC (L2) | 0.9900   | 0.9862    | 0.9931 | 0.9896   |

## Configuration Options

The `MCPCConfig` class provides numerous configuration options:

| Parameter | Description | Default |
|-----------|-------------|--------|
| classifier_type | Type of classifier ("pcc", "epcc", "mcpcc") | "epcc" |
| norm_type | Distance metric ("L1", "L2") | "L2" |
| penalty_coefficient | SVM penalty parameter | 1.0 |
| num_centers | Number of centers (for MCPCC) | None |
| verbose | Enable verbose output | False |
| use_cache | Enable feature caching | True |
| cache_size | Size of feature cache | 128 |
| n_jobs | Number of parallel jobs | -1 |
| kmeans_algorithm | KMeans algorithm | "elkan" |
| model_path | Path to save/load model | None |

## Examples

Check out the following example scripts:

- `demo.py`: Simple demonstration of the MCPCC classifier
- `example.py`: Comprehensive example with multiple datasets and classifiers
- `visualize.py`: Visualization utilities for decision boundaries and performance metrics

## Author

Halil Sağlamlar

## License

This project is licensed under the MIT License.
