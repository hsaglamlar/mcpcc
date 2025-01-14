"""
Title: Multi-Center Polyhedral Classifier with Various Distance Metrics
Description: Implementation of Multi-Center Polyhedral Classifier with L1 andd L2 distance metrics.
Version: 0.1
Authors: Halil Sağlamlar

Sample terminal command to run this script:
    python mcpcc.py --data_path ./data/chessboard.npz 
    --classifier_type mcpcc --norm_type L2 --penalty_coefficient 100.0 --num_centers 75 --verbose

"""

# Import necessary libraries
from dataclasses import dataclass
from typing import Literal, Optional
import argparse

import numpy as np
from numpy.typing import NDArray
from sklearn.svm import SVC
from sklearn.cluster import KMeans


# Type definitions
ClassifierType = Literal["pcc", "epcc", "mcpcc"]
NormType = Literal["L1", "L2"]

# Constants
RANDOM_SEED = 42
KMEANS_N_INIT = 3


@dataclass(frozen=True)
class MCPCConfig:
    """Configuration options for MCPC classifier."""

    classifier_type: ClassifierType = "epcc"
    norm_type: NormType = "L1"
    penalty_coefficient: float = 1.0
    num_centers: int = 1
    centers: Optional[NDArray] = None
    verbose: bool = False

    def __post_init__(self):
        if self.classifier_type == "mcpcc" and self.num_centers < 2:
            raise ValueError("MCPCC requires at least 2 centers.")
        if self.norm_type not in NormType.__args__:
            raise ValueError("Invalid norm type. Avaliable params: ", NormType.__args__)
        if self.classifier_type not in ClassifierType.__args__:
            raise ValueError(
                "Invalid classifier type. Avaliable params: ", ClassifierType.__args__
            )


class MCPCClassifier:
    """Implements Multi-Center Polyhedral Classifier with L1 and L2 distance metrics."""

    def __init__(self, config: MCPCConfig):
        self.params = config
        self.centers: Optional[NDArray] = None
        self.model: Optional[SVC] = None
        self._norm_order = 1 if config.norm_type == "L1" else 2
        self._arrangement_function = self._get_arrangement_function()

    def _get_arrangement_function(self):
        """Return the appropriate sample arrangement function based on classifier type."""
        algorithm = f"{self.params.classifier_type}{self.params.norm_type}"

        if algorithm.startswith("pcc"):
            return self._arrange_pcc
        elif algorithm == "epccL1":
            return self._arrange_epcc_l1
        elif algorithm == "epccL2":
            return self._arrange_epcc_l2
        elif algorithm.startswith("mcpcc"):
            return self._arrange_mcpcc
        raise ValueError(f"Unknown algorithm: {algorithm}")

    @staticmethod
    def _compute_positive_mean(features: NDArray, labels: NDArray) -> NDArray:
        """Compute mean of positive samples."""
        positive_mask = (labels == 1).flatten()
        return np.mean(features[positive_mask], axis=0, keepdims=True)

    def _compute_centers(self, features: NDArray, labels: NDArray) -> NDArray:
        """Compute centers based on training data and classifier type."""
        centers = self._compute_positive_mean(features, labels)

        if self.params.classifier_type == "mcpcc":
            kmeans = KMeans(
                n_clusters=(self.params.num_centers - 1),
                n_init=KMEANS_N_INIT,
                random_state=RANDOM_SEED,
            )
            additional_centers = kmeans.fit(features).cluster_centers_
            centers = np.vstack([centers, additional_centers])

        return centers

    def _arrange_pcc(self, feature_matrix: NDArray) -> NDArray:
        """Arrange samples for PCC classifier."""
        diff = feature_matrix - self.centers
        norms = np.linalg.norm(diff, ord=self._norm_order, axis=1, keepdims=True)
        return np.hstack([diff, norms])

    def _arrange_epcc_l1(self, feature_matrix: NDArray) -> NDArray:
        """Arrange samples for EPCC with L1 norm."""
        diff = feature_matrix - self.centers
        return np.hstack([diff, np.abs(diff)])

    def _arrange_epcc_l2(self, feature_matrix: NDArray) -> NDArray:
        """Arrange samples for EPCC with L2 norm."""
        diff = feature_matrix - self.centers
        return np.hstack([diff, diff * diff])

    def _arrange_mcpcc(self, feature_matrix: NDArray) -> NDArray:
        """Arrange samples for MCPCC classifier."""
        distances = np.array(
            [
                np.linalg.norm(feature_matrix - center, ord=self._norm_order, axis=1)
                for center in self.centers
            ]
        ).T
        return np.hstack([feature_matrix - self.centers[-1], distances])

    def _arrange_samples(self, feature_matrix: NDArray) -> NDArray:
        """Arrange samples using the pre-selected arrangement function."""
        if self.centers is None:
            raise ValueError("Centers not computed. Call fit() first.")
        return self._arrangement_function(feature_matrix)

    def fit(self, features: NDArray, labels: NDArray) -> "MCPCClassifier":
        """Train the classifier."""
        self.centers = self._compute_centers(features, labels)
        arranged_features = self._arrange_samples(features)

        self.model = SVC(
            C=self.params.penalty_coefficient,
            kernel="linear",
            probability=True,
            random_state=RANDOM_SEED,
        )
        self.model.fit(arranged_features, labels.ravel())

        if self.params.verbose:
            print(f"parameters: {self.params}")

        return self

    def predict(self, features: NDArray) -> NDArray:
        """Predict class labels for samples in features."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        return self.model.predict(self._arrange_samples(features))

    def predict_proba(self, features: NDArray) -> NDArray:
        """Predict class probabilities for samples in features."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        return self.model.predict_proba(self._arrange_samples(features))


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Multi-Center Polyhedral Classifier")
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/chessboard.npz",
        help="Path to data file",
    )
    parser.add_argument(
        "--classifier_type",
        type=str,
        choices=["pcc", "epcc", "mcpcc"],
        default="epcc",
        help="Type of classifier",
    )
    parser.add_argument(
        "--norm_type", type=str, choices=["L1", "L2"], default="L2", help="Norm type"
    )
    parser.add_argument(
        "--penalty_coefficient", type=float, default=1.0, help="Penalty coefficient"
    )
    parser.add_argument("--num_centers", type=int, default=1, help="Number of centers")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")

    return parser.parse_args()


def main():
    """Main function to run the MCPC classifier."""
    args = parse_arguments()

    config = MCPCConfig(
        classifier_type=args.classifier_type,
        norm_type=args.norm_type,
        penalty_coefficient=args.penalty_coefficient,
        num_centers=args.num_centers,
        verbose=args.verbose,
    )

    classifier = MCPCClassifier(config)

    # Load data
    data = np.load(args.data_path)
    train_features = data["train_features"]
    train_labels = data["train_labels"]
    test_features = data["test_features"]
    test_labels = data["test_labels"]

    classifier.fit(train_features, train_labels)
    predicted_labels = classifier.predict(test_features)
    accuracy = np.mean(predicted_labels == test_labels.ravel())
    print(f"Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
