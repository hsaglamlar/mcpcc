"""
Title: Multi-Center Polyhedral Classifier with Various Distance Metrics
Description: Implementation of Multi-Center Polyhedral Classifier with L1 and L2 distance metrics.
Version: 0.2
Authors: Halil Sağlamlar

Sample terminal command to run this script:
    python mcpcc.py --data_path ./data/chessboard.npz
    --classifier_type mcpcc --norm_type L2 --penalty_coefficient 100.0 --num_centers 75 --verbose

"""

# Import necessary libraries
from dataclasses import dataclass
from typing import Literal, Optional, Dict, Callable
import argparse
import os
import pickle
import time
import warnings

import numpy as np
from numpy.typing import NDArray
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Type definitions
ClassifierType = Literal["pcc", "epcc", "mcpcc"]
NormType = Literal["L1", "L2"]
MetricFunction = Callable[[NDArray, NDArray], float]

# Constants
RANDOM_SEED = 42
KMEANS_N_INIT = 10
N_JOBS = -1  # Use all available cores


@dataclass(frozen=True)
class MCPCConfig:
    """Configuration options for MCPC classifier."""

    classifier_type: ClassifierType = "epcc"
    norm_type: NormType = "L2"
    penalty_coefficient: float = 1.0
    num_centers: int = None
    centers: Optional[NDArray] = None
    verbose: bool = False
    use_cache: bool = True
    cache_size: int = 128
    n_jobs: int = N_JOBS
    kmeans_algorithm: str = "elkan"
    model_path: Optional[str] = None

    def __post_init__(self):
        if self.classifier_type == "mcpcc" and self.num_centers < 2:
            raise ValueError("MCPCC requires at least 2 centers.")
        if self.norm_type not in NormType.__args__:
            raise ValueError("Invalid norm type. Available params: ", NormType.__args__)
        if self.classifier_type not in ClassifierType.__args__:
            raise ValueError(
                "Invalid classifier type. Available params: ", ClassifierType.__args__
            )


class MCPCClassifier:
    """Implements Multi-Center Polyhedral Classifier with L1 and L2 distance metrics."""

    def __init__(self, config: MCPCConfig):
        self.params = config
        self.centers: Optional[NDArray] = None
        self.model: Optional[SVC] = None
        self._norm_order = 1 if config.norm_type == "L1" else 2
        self._arrangement_function = self._get_arrangement_function()
        # Initialize metrics and caching
        # Define metric functions
        self._metrics = {
            "accuracy": accuracy_score,
            "precision": self._precision_score,
            "recall": self._recall_score,
            "f1": self._f1_score,
        }

        # Initialize cache dictionaries instead of using lru_cache
        # (NumPy arrays aren't hashable for lru_cache)
        self._distances_cache = {}
        self._samples_cache = {}
        self._use_cache = config.use_cache
        self._cache_size = config.cache_size

    # Helper methods for metrics (to avoid using lambdas which can't be pickled)
    @staticmethod
    def _precision_score(y_true, y_pred):
        return precision_score(y_true, y_pred, zero_division=0)

    @staticmethod
    def _recall_score(y_true, y_pred):
        return recall_score(y_true, y_pred, zero_division=0)

    @staticmethod
    def _f1_score(y_true, y_pred):
        return f1_score(y_true, y_pred, zero_division=0)

    # Scikit-learn estimator interface methods
    def get_params(self):
        """Get parameters for this estimator.

        This method is required by scikit-learn's estimator interface.
        """
        return {"config": self.params}

    def set_params(self, **parameters):
        """Set the parameters of this estimator.

        This method is required by scikit-learn's estimator interface.
        """
        for parameter, value in parameters.items():
            if parameter == "config":
                self.params = value
                self._norm_order = 1 if value.norm_type == "L1" else 2
                self._arrangement_function = self._get_arrangement_function()
        return self

    # Scikit-learn estimator interface for cross-validation
    def _sklearn_fit(self, X, y):
        """Fit the model according to the given training data.

        This method follows scikit-learn's fit interface.
        """
        return self.fit(X, y)

    def score(self, X, y):
        """Return the mean accuracy on the given test data and labels.

        This method follows scikit-learn's score interface.
        """
        return accuracy_score(y, self.predict(X))

    def _get_arrangement_function(self):
        """Return the appropriate sample arrangement function based on classifier type."""
        algorithm = f"{self.params.classifier_type}{self.params.norm_type}"

        arrangement_functions = {
            "pccL1": self._arrange_pcc,
            "pccL2": self._arrange_pcc,
            "epccL1": self._arrange_epcc_l1,
            "epccL2": self._arrange_epcc_l2,
            "mcpccL1": self._arrange_mcpcc,
            "mcpccL2": self._arrange_mcpcc,
        }

        return arrangement_functions[algorithm]

    @staticmethod
    def _compute_positive_mean(features: NDArray, labels: NDArray) -> NDArray:
        """Compute mean of positive samples."""
        positive_mask = (labels == 1).flatten()
        return np.mean(features[positive_mask], axis=0, keepdims=True)

    def _compute_centers(self, features: NDArray, labels: NDArray) -> NDArray:
        """Compute centers based on training data and classifier type."""
        start_time = time.time()
        centers = self._compute_positive_mean(features, labels)

        if self.params.classifier_type == "mcpcc":
            # Use more efficient KMeans parameters
            kmeans = KMeans(
                n_clusters=(self.params.num_centers - 1),
                n_init=KMEANS_N_INIT,
                random_state=RANDOM_SEED,
                algorithm=self.params.kmeans_algorithm,
            )

            # For large datasets, consider using mini-batch KMeans
            if features.shape[0] > 10000:
                from sklearn.cluster import MiniBatchKMeans

                warnings.warn(
                    "Large dataset detected, using MiniBatchKMeans for better performance"
                )
                kmeans = MiniBatchKMeans(
                    n_clusters=(self.params.num_centers - 1),
                    random_state=RANDOM_SEED,
                    batch_size=1000,
                )

            additional_centers = kmeans.fit(features).cluster_centers_
            centers = np.vstack([centers, additional_centers])

        if self.params.verbose:
            print(
                f"Center computation completed in {time.time() - start_time:.2f} seconds"
            )

        return centers

    def _arrange_pcc(self, feature_matrix: NDArray) -> NDArray:
        """Arrange samples for PCC classifier."""
        diff = feature_matrix - self.centers
        norms = np.linalg.norm(diff, ord=self._norm_order, axis=1, keepdims=True)
        return np.hstack((diff, norms))

    def _arrange_epcc_l1(self, feature_matrix: NDArray) -> NDArray:
        """Arrange samples for EPCC with L1 norm."""
        diff = feature_matrix - self.centers
        return np.hstack((diff, np.abs(diff)))

    def _arrange_epcc_l2(self, feature_matrix: NDArray) -> NDArray:
        """Arrange samples for EPCC with L2 norm."""
        diff = feature_matrix - self.centers
        return np.hstack((diff, diff * diff))

    def _compute_distances(self, feature_matrix: NDArray) -> NDArray:
        """Compute distances from samples to all centers.

        This is separated to enable caching of distance calculations.
        """
        # Create a cache key based on the shape and a hash of the data
        if self._use_cache:
            # Simple cache key based on the first few values and shape
            # (not perfect but good enough for most cases)
            if feature_matrix.size > 0:
                cache_key = (
                    feature_matrix.shape,
                    hash(
                        feature_matrix.tobytes()[:1000]
                        if feature_matrix.size > 1000
                        else feature_matrix.tobytes()
                    ),
                )

                if cache_key in self._distances_cache:
                    return self._distances_cache[cache_key]

        # Compute distances
        distances = np.linalg.norm(
            feature_matrix[:, None, :] - self.centers, ord=self._norm_order, axis=2
        )

        # Cache the result
        if self._use_cache:
            # Manage cache size
            if len(self._distances_cache) >= self._cache_size:
                # Simple strategy: clear the cache when it gets too big
                self._distances_cache.clear()

            if feature_matrix.size > 0:
                self._distances_cache[cache_key] = distances

        return distances

    def _arrange_mcpcc(self, feature_matrix: NDArray) -> NDArray:
        """Arrange samples for MCPCC classifier."""
        # Get distances (potentially from cache if enabled)
        distances = self._compute_distances(feature_matrix)

        # Compute the difference from the first center
        diff_from_first = feature_matrix - self.centers[0]

        # Stack horizontally for the final feature representation
        return np.hstack((diff_from_first, distances))

    def _arrange_samples(self, feature_matrix: NDArray) -> NDArray:
        """Arrange samples using the pre-selected arrangement function."""
        if self.centers is None:
            raise ValueError("Centers not computed. Call fit() first.")

        # Check cache
        if self._use_cache:
            # Simple cache key based on the first few values and shape
            if feature_matrix.size > 0:
                cache_key = (
                    feature_matrix.shape,
                    hash(
                        feature_matrix.tobytes()[:1000]
                        if feature_matrix.size > 1000
                        else feature_matrix.tobytes()
                    ),
                )

                if cache_key in self._samples_cache:
                    return self._samples_cache[cache_key]

        # Compute arranged samples
        arranged = self._arrangement_function(feature_matrix)

        # Cache the result
        if self._use_cache:
            # Manage cache size
            if len(self._samples_cache) >= self._cache_size:
                # Simple strategy: clear the cache when it gets too big
                self._samples_cache.clear()

            if feature_matrix.size > 0:
                self._samples_cache[cache_key] = arranged

        return arranged

    def fit(self, features: NDArray, labels: NDArray) -> "MCPCClassifier":
        """Train the classifier. Find centers, modify the features and fit the model."""
        start_time = time.time()

        # Clear caches
        self._distances_cache.clear()
        self._samples_cache.clear()

        # Compute centers
        self.centers = self._compute_centers(features, labels)

        # Arrange features for training
        arranged_features = self._arrange_samples(features)

        # Initialize and train SVM model
        self.model = SVC(
            C=self.params.penalty_coefficient,
            kernel="linear",
            probability=True,
            random_state=RANDOM_SEED,
        )
        self.model.fit(arranged_features, labels.ravel())

        # Save model if path is provided
        if self.params.model_path:
            os.makedirs(os.path.dirname(self.params.model_path), exist_ok=True)
            with open(self.params.model_path, "wb") as f:
                pickle.dump(self, f)

        if self.params.verbose:
            print(f"Parameters: {self.params}")
            print(f"Training completed in {time.time() - start_time:.2f} seconds")

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

    def evaluate(self, features: NDArray, labels: NDArray) -> Dict[str, float]:
        """Evaluate the classifier on test data with multiple metrics."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        # Get predictions
        predictions = self.predict(features)

        # Calculate metrics
        results = {}
        for metric_name, metric_func in self._metrics.items():
            results[metric_name] = metric_func(labels.ravel(), predictions)

        return results

    def cross_validate(
        self, features: NDArray, labels: NDArray, cv: int = 5
    ) -> Dict[str, float]:
        """Perform cross-validation on the data."""
        from sklearn.model_selection import KFold

        # Manual cross-validation implementation to avoid scikit-learn clone issues
        kf = KFold(n_splits=cv, shuffle=True, random_state=RANDOM_SEED)
        scores = []

        for train_idx, test_idx in kf.split(features):
            # Get train/test split for this fold
            X_train, X_test = features[train_idx], features[test_idx]
            y_train, y_test = labels[train_idx], labels[test_idx]

            # Create a new classifier with the same configuration
            fold_clf = MCPCClassifier(self.params)

            # Train and evaluate
            fold_clf.fit(X_train, y_train)
            y_pred = fold_clf.predict(X_test)
            fold_score = accuracy_score(y_test.ravel(), y_pred)
            scores.append(fold_score)

        # Convert to numpy array for calculations
        scores = np.array(scores)

        return {
            "mean_accuracy": np.mean(scores),
            "std_accuracy": np.std(scores),
            "scores": scores,
        }

    def save(self, filepath: str) -> None:
        """Save the trained model to a file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str) -> "MCPCClassifier":
        """Load a trained model from a file."""
        with open(filepath, "rb") as f:
            return pickle.load(f)


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
    parser.add_argument(
        "--use_cache", action="store_true", help="Enable feature caching"
    )
    parser.add_argument(
        "--cache_size", type=int, default=128, help="Size of feature cache"
    )
    parser.add_argument(
        "--n_jobs", type=int, default=N_JOBS, help="Number of parallel jobs"
    )
    parser.add_argument("--model_path", type=str, help="Path to save/load model")
    parser.add_argument(
        "--cross_validate", action="store_true", help="Perform cross-validation"
    )
    parser.add_argument(
        "--cv_folds", type=int, default=5, help="Number of cross-validation folds"
    )

    return parser.parse_args()


def main():
    """Main function to run the MCPC classifier."""
    args = parse_arguments()

    # Start timing
    start_time = time.time()

    config = MCPCConfig(
        classifier_type=args.classifier_type,
        norm_type=args.norm_type,
        penalty_coefficient=args.penalty_coefficient,
        num_centers=args.num_centers,
        verbose=args.verbose,
        use_cache=args.use_cache,
        cache_size=args.cache_size,
        n_jobs=args.n_jobs,
        model_path=args.model_path,
    )

    classifier = MCPCClassifier(config)

    # Load data
    data = np.load(args.data_path)
    train_features = data["train_features"]
    train_labels = data["train_labels"]
    test_features = data["test_features"]
    test_labels = data["test_labels"]

    # Perform cross-validation if requested
    if args.cross_validate:
        cv_results = classifier.cross_validate(
            train_features, train_labels, cv=args.cv_folds
        )
        print("Cross-validation results:")
        print(
            f"Mean accuracy: {cv_results['mean_accuracy']:.4f} ± {cv_results['std_accuracy']:.4f}"
        )
        print(f"Individual fold scores: {cv_results['scores']}")

    # Train the classifier
    classifier.fit(train_features, train_labels)

    # Evaluate on test data
    evaluation = classifier.evaluate(test_features, test_labels)

    # Print results
    print("\nTest set evaluation:")
    for metric, value in evaluation.items():
        print(f"{metric.capitalize()}: {value:.4f}")

    # Print total execution time
    print(f"\nTotal execution time: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
