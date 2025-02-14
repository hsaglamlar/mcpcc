"""
This is a demo script to show how to use the MCPC classifier."""

from scipy.io import loadmat
import numpy as np

from mcpcc import MCPCConfig
from mcpcc import MCPCClassifier

# Load data
data = np.load("./data/chessboard.npz")

config = MCPCConfig(
    classifier_type="mcpcc",
    norm_type="L2",
    penalty_coefficient=50.0,
    num_centers=100,
    verbose=True,
)

classifier = MCPCClassifier(config)
classifier.fit(data["train_features"], data["train_labels"])

predicted_labels = classifier.predict(data["test_features"])
accuracy = np.mean(predicted_labels == data["test_labels"].ravel())
print(f"Accuracy: {accuracy:.4f}")
