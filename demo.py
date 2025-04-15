"""
This is a demo script to show how to use the optimized MCPC classifier.

This demo showcases the improved performance and additional features of the classifier.
"""

import time
import os
import numpy as np
from mcpcc import MCPCConfig
from mcpcc import MCPCClassifier

# Start timing
start_time = time.time()

# Load data
data = np.load("./data/chessboard.npz")

# Create model directory if it doesn't exist
os.makedirs("./models", exist_ok=True)

# Configure the classifier with optimized settings
config = MCPCConfig(
    classifier_type="mcpcc",
    norm_type="L2",
    penalty_coefficient=50.0,
    num_centers=100,
    verbose=True,
    use_cache=True,
    cache_size=256,
    model_path="./models/mcpcc_model.pkl",
)

# Create and train the classifier
classifier = MCPCClassifier(config)

# Perform cross-validation
print("Performing cross-validation...")
cv_results = classifier.cross_validate(
    data["train_features"], data["train_labels"], cv=5
)
print("Cross-validation results:")
print(
    f"Mean accuracy: {cv_results['mean_accuracy']:.4f} ± {cv_results['std_accuracy']:.4f}"
)

# Train the classifier
print("\nTraining the classifier...")
classifier.fit(data["train_features"], data["train_labels"])

# Evaluate on test data
print("\nEvaluating on test data...")
evaluation = classifier.evaluate(data["test_features"], data["test_labels"])

# Print results
print("\nTest set evaluation:")
for metric, value in evaluation.items():
    print(f"{metric.capitalize()}: {value:.4f}")

# Print total execution time
print(f"\nTotal execution time: {time.time() - start_time:.2f} seconds")

# Demonstrate loading a saved model
print("\nDemonstrating model loading...")
loaded_classifier = MCPCClassifier.load("./models/mcpcc_model.pkl")
loaded_evaluation = loaded_classifier.evaluate(
    data["test_features"], data["test_labels"]
)
print("Evaluation with loaded model:")
print(f"Accuracy: {loaded_evaluation['accuracy']:.4f}")
