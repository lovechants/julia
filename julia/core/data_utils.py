import numpy as np
from typing import Tuple, Optional, Dict, Any, Union
from pathlib import Path
import urllib.request
import tarfile
import zipfile


class DataProcessor:
    """
    Data preprocessing utilities
    """

    @staticmethod
    def standardize(
        data: np.ndarray, axis: int = 0, return_stats: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
        """
        Standardize data to zero mean and unit variance

        Args:
            data: Input data
            axis: Axis along which to compute statistics
            return_stats: Whether to return mean and std
        """
        mean = np.mean(data, axis=axis, keepdims=True)
        std = np.std(data, axis=axis, keepdims=True)

        # Avoid division by zero
        std = np.where(std == 0, 1, std)

        standardized = (data - mean) / std

        if return_stats:
            stats = {"mean": mean, "std": std}
            return standardized, stats
        return standardized

    @staticmethod
    def normalize(
        data: np.ndarray, axis: int = 0, return_stats: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
        """
        Normalize data to [0, 1] range

        Args:
            data: Input data
            axis: Axis along which to compute statistics
            return_stats: Whether to return min and max
        """
        min_val = np.min(data, axis=axis, keepdims=True)
        max_val = np.max(data, axis=axis, keepdims=True)

        # Avoid division by zero
        range_val = max_val - min_val
        range_val = np.where(range_val == 0, 1, range_val)

        normalized = (data - min_val) / range_val

        if return_stats:
            stats = {"min": min_val, "max": max_val}
            return normalized, stats
        return normalized

    @staticmethod
    def one_hot_encode(
        labels: np.ndarray, num_classes: Optional[int] = None
    ) -> np.ndarray:
        """
        Convert labels to one-hot encoding

        Args:
            labels: Integer labels
            num_classes: Number of classes (inferred if None)
        """
        if num_classes is None:
            num_classes = int(np.max(labels)) + 1

        one_hot = np.zeros((len(labels), num_classes))
        one_hot[np.arange(len(labels)), labels.astype(int)] = 1
        return one_hot

    @staticmethod
    def categorical_to_numeric(
        data: np.ndarray, category_map: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Convert categorical data to numeric

        Args:
            data: Categorical data
            category_map: Existing mapping (created if None)
        """
        if category_map is None:
            unique_categories = np.unique(data)
            category_map = {cat: i for i, cat in enumerate(unique_categories)}

        numeric_data = np.array([category_map.get(item, -1) for item in data])
        return numeric_data, category_map

    @staticmethod
    def handle_missing_values(
        data: np.ndarray, strategy: str = "mean", fill_value: Optional[float] = None
    ) -> np.ndarray:
        """
        Handle missing values in data

        Args:
            data: Input data with potential NaN values
            strategy: 'mean', 'median', 'mode', 'constant', 'drop'
            fill_value: Value to use for 'constant' strategy
        """
        if strategy == "mean":
            fill_value = np.nanmean(data, axis=0)
        elif strategy == "median":
            fill_value = np.nanmedian(data, axis=0)
        elif strategy == "mode":
            # For mode, take most frequent value
            fill_value = []
            for col in range(data.shape[1] if data.ndim > 1 else 1):
                col_data = data[:, col] if data.ndim > 1 else data
                values, counts = np.unique(
                    col_data[~np.isnan(col_data)], return_counts=True
                )
                mode_val = values[np.argmax(counts)] if len(values) > 0 else 0
                fill_value.append(mode_val)
            fill_value = np.array(fill_value)
        elif strategy == "constant":
            if fill_value is None:
                fill_value = 0
        elif strategy == "drop":
            return data[~np.isnan(data).any(axis=1)]

        # Fill NaN values
        filled_data = data.copy()
        nan_mask = np.isnan(filled_data)

        if data.ndim > 1:
            for col in range(data.shape[1]):
                col_mask = nan_mask[:, col]
                if np.any(col_mask):
                    if isinstance(fill_value, np.ndarray):
                        filled_data[col_mask, col] = fill_value[col]
                    else:
                        filled_data[col_mask, col] = fill_value
        else:
            filled_data[nan_mask] = fill_value

        return filled_data

    @staticmethod
    def remove_outliers(
        data: np.ndarray, method: str = "iqr", threshold: float = 1.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove outliers from data

        Args:
            data: Input data
            method: 'iqr', 'zscore', 'percentile'
            threshold: Threshold for outlier detection

        Returns:
            Cleaned data and boolean mask of outliers
        """
        if method == "iqr":
            Q1 = np.percentile(data, 25, axis=0)
            Q3 = np.percentile(data, 75, axis=0)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

            outlier_mask = (data < lower_bound) | (data > upper_bound)
            if data.ndim > 1:
                outlier_mask = np.any(outlier_mask, axis=1)

        elif method == "zscore":
            z_scores = np.abs((data - np.mean(data, axis=0)) / np.std(data, axis=0))
            outlier_mask = z_scores > threshold
            if data.ndim > 1:
                outlier_mask = np.any(outlier_mask, axis=1)

        elif method == "percentile":
            lower_percentile = (100 - threshold * 100) / 2
            upper_percentile = 100 - lower_percentile
            lower_bound = np.percentile(data, lower_percentile, axis=0)
            upper_bound = np.percentile(data, upper_percentile, axis=0)

            outlier_mask = (data < lower_bound) | (data > upper_bound)
            if data.ndim > 1:
                outlier_mask = np.any(outlier_mask, axis=1)

        else:
            raise ValueError(f"Unknown method: {method}")

        cleaned_data = data[~outlier_mask]
        return cleaned_data, outlier_mask


class DataAugmentation:
    """
    Data augmentation utilities
    """

    @staticmethod
    def add_noise(
        data: np.ndarray, noise_factor: float = 0.1, noise_type: str = "gaussian"
    ) -> np.ndarray:
        """
        Add noise to data for augmentation
        """
        if noise_type == "gaussian":
            noise = np.random.normal(0, noise_factor, data.shape)
        elif noise_type == "uniform":
            noise = np.random.uniform(-noise_factor, noise_factor, data.shape)
        elif noise_type == "salt_pepper":
            noise = np.random.choice(
                [-noise_factor, 0, noise_factor], size=data.shape, p=[0.1, 0.8, 0.1]
            )
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")

        return data + noise

    @staticmethod
    def mixup(
        x1: np.ndarray,
        x2: np.ndarray,
        y1: np.ndarray,
        y2: np.ndarray,
        alpha: float = 0.2,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Mixup data augmentation
        """
        lam = np.random.beta(alpha, alpha)
        mixed_x = lam * x1 + (1 - lam) * x2
        mixed_y = lam * y1 + (1 - lam) * y2
        return mixed_x, mixed_y

    @staticmethod
    def cutmix(
        x1: np.ndarray,
        x2: np.ndarray,
        y1: np.ndarray,
        y2: np.ndarray,
        alpha: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        CutMix data augmentation for 2D data
        """
        lam = np.random.beta(alpha, alpha)

        if len(x1.shape) < 2:
            # For 1D data, just do mixup
            return DataAugmentation.mixup(x1, x2, y1, y2, alpha)

        # For 2D data
        H, W = x1.shape[-2:]
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        # Random center
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        # Bounding box
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        mixed_x = x1.copy()
        mixed_x[..., bby1:bby2, bbx1:bbx2] = x2[..., bby1:bby2, bbx1:bbx2]

        # Adjust lambda to exact area ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        mixed_y = lam * y1 + (1 - lam) * y2

        return mixed_x, mixed_y


class DatasetDownloader:
    """
    Download and cache common datasets
    """

    def __init__(self, cache_dir: str = "./data"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def download_file(self, url: str, filename: str) -> Path:
        """
        Download file from URL with caching
        """
        file_path = self.cache_dir / filename

        if file_path.exists():
            print(f"File {filename} already exists, skipping download")
            return file_path

        print(f"Downloading {filename} from {url}")
        urllib.request.urlretrieve(url, file_path)
        print(f"Downloaded {filename}")

        return file_path

    def extract_archive(
        self, archive_path: Path, extract_to: Optional[Path] = None
    ) -> Path:
        """
        Extract archive file
        """
        if extract_to is None:
            extract_to = archive_path.parent / archive_path.stem

        extract_to.mkdir(exist_ok=True)

        if archive_path.suffix == ".gz" and archive_path.stem.endswith(".tar"):
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(extract_to)
        elif archive_path.suffix == ".zip":
            with zipfile.ZipFile(archive_path, "r") as zip_file:
                zip_file.extractall(extract_to)
        elif archive_path.suffix == ".tar":
            with tarfile.open(archive_path, "r") as tar:
                tar.extractall(extract_to)
        else:
            raise ValueError(f"Unsupported archive format: {archive_path.suffix}")

        return extract_to

    def load_iris(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load Iris dataset
        """
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
        file_path = self.download_file(url, "iris.data")

        data = np.loadtxt(file_path, delimiter=",", dtype=str)
        X = data[:, :-1].astype(np.float32)

        # Convert species to numeric
        species_map = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
        y = np.array([species_map[species] for species in data[:, -1]], dtype=np.int64)

        return X, y

    def load_wine(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load Wine dataset
        """
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
        file_path = self.download_file(url, "wine.data")

        data = np.loadtxt(file_path, delimiter=",")
        y = data[:, 0].astype(np.int64) - 1  # Convert to 0-indexed
        X = data[:, 1:].astype(np.float32)

        return X, y

    def load_boston_housing(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load Boston Housing dataset
        """
        # Note: This dataset has been removed from sklearn due to ethical concerns
        # This is a simple implementation for educational purposes
        print(
            "Warning: Boston Housing dataset has ethical concerns and should be used carefully"
        )

        url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
        file_path = self.download_file(url, "boston_housing.csv")

        data = np.loadtxt(file_path, delimiter=",", skiprows=1)
        X = data[:, :-1].astype(np.float32)
        y = data[:, -1].astype(np.float32)

        return X, y


class FeatureSelector:
    """
    Feature selection utilities
    """

    @staticmethod
    def variance_threshold(
        X: np.ndarray, threshold: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove features with low variance
        """
        variances = np.var(X, axis=0)
        mask = variances > threshold
        return X[:, mask], mask

    @staticmethod
    def correlation_filter(
        X: np.ndarray, threshold: float = 0.95
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove highly correlated features
        """
        corr_matrix = np.corrcoef(X.T)

        # Find pairs of features with high correlation
        upper_tri = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        high_corr_pairs = np.where((np.abs(corr_matrix) > threshold) & upper_tri)

        # Remove one feature from each highly correlated pair
        features_to_remove = set()
        for i, j in zip(high_corr_pairs[0], high_corr_pairs[1]):
            features_to_remove.add(j)  # Remove the second feature

        mask = np.ones(X.shape[1], dtype=bool)
        mask[list(features_to_remove)] = False

        return X[:, mask], mask

    @staticmethod
    def univariate_selection(
        X: np.ndarray, y: np.ndarray, k: int = 10, method: str = "f_score"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select k best features using univariate statistical tests
        """
        if method == "f_score":
            scores = FeatureSelector._f_score(X, y)
        elif method == "mutual_info":
            scores = FeatureSelector._mutual_info(X, y)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Select top k features
        top_k_indices = np.argsort(scores)[-k:]
        mask = np.zeros(X.shape[1], dtype=bool)
        mask[top_k_indices] = True

        return X[:, mask], mask

    @staticmethod
    def _f_score(X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute F-score for each feature
        """
        classes = np.unique(y)
        n_classes = len(classes)
        n_samples, n_features = X.shape

        scores = np.zeros(n_features)

        for i in range(n_features):
            feature = X[:, i]

            # Overall mean
            overall_mean = np.mean(feature)

            # Between-class variance
            between_var = 0
            for c in classes:
                class_mask = y == c
                class_mean = np.mean(feature[class_mask])
                class_size = np.sum(class_mask)
                between_var += class_size * (class_mean - overall_mean) ** 2
            between_var /= n_classes - 1

            # Within-class variance
            within_var = 0
            for c in classes:
                class_mask = y == c
                class_feature = feature[class_mask]
                class_mean = np.mean(class_feature)
                within_var += np.sum((class_feature - class_mean) ** 2)
            within_var /= n_samples - n_classes

            # F-score
            if within_var > 0:
                scores[i] = between_var / within_var
            else:
                scores[i] = 0

        return scores

    @staticmethod
    def _mutual_info(X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute mutual information for each feature (simplified version)
        """
        n_features = X.shape[1]
        scores = np.zeros(n_features)

        for i in range(n_features):
            feature = X[:, i]

            # Discretize continuous features
            n_bins = min(10, len(np.unique(feature)))
            feature_binned = np.digitize(
                feature, np.linspace(feature.min(), feature.max(), n_bins)
            )

            # Calculate mutual information
            mi = 0
            for f_val in np.unique(feature_binned):
                for y_val in np.unique(y):
                    # Joint probability
                    p_xy = np.mean((feature_binned == f_val) & (y == y_val))

                    # Marginal probabilities
                    p_x = np.mean(feature_binned == f_val)
                    p_y = np.mean(y == y_val)

                    if p_xy > 0 and p_x > 0 and p_y > 0:
                        mi += p_xy * np.log(p_xy / (p_x * p_y))

            scores[i] = mi

        return scores


class DataValidator:
    """
    Data validation utilities
    """

    @staticmethod
    def check_data_quality(
        X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive data quality check
        """
        report = {
            "shape": X.shape,
            "dtype": str(X.dtype),
            "missing_values": np.sum(np.isnan(X)),
            "missing_percentage": np.sum(np.isnan(X)) / X.size * 100,
            "infinite_values": np.sum(np.isinf(X)),
            "unique_values_per_feature": [],
            "feature_statistics": {},
        }

        # Per-feature statistics
        for i in range(X.shape[1]):
            feature = X[:, i]
            feature_clean = feature[~np.isnan(feature)]

            stats = {
                "mean": np.mean(feature_clean) if len(feature_clean) > 0 else np.nan,
                "std": np.std(feature_clean) if len(feature_clean) > 0 else np.nan,
                "min": np.min(feature_clean) if len(feature_clean) > 0 else np.nan,
                "max": np.max(feature_clean) if len(feature_clean) > 0 else np.nan,
                "unique_count": len(np.unique(feature_clean)),
            }

            report["feature_statistics"][f"feature_{i}"] = stats
            report["unique_values_per_feature"].append(stats["unique_count"])

        # Label statistics if provided
        if y is not None:
            y_clean = y[~np.isnan(y)]
            unique_labels, counts = np.unique(y_clean, return_counts=True)

            report["label_statistics"] = {
                "unique_labels": unique_labels.tolist(),
                "label_counts": counts.tolist(),
                "class_balance": (counts / len(y_clean)).tolist(),
                "most_frequent_class": unique_labels[np.argmax(counts)],
                "least_frequent_class": unique_labels[np.argmin(counts)],
            }

        return report

    @staticmethod
    def detect_data_drift(
        X_train: np.ndarray, X_test: np.ndarray, threshold: float = 0.05
    ) -> Dict[str, Any]:
        """
        Detect data drift between training and test sets
        """
        from scipy import stats

        drift_report = {
            "features_with_drift": [],
            "drift_scores": [],
            "threshold": threshold,
        }

        n_features = X_train.shape[1]

        for i in range(n_features):
            train_feature = X_train[:, i]
            test_feature = X_test[:, i]

            # Remove NaN values
            train_clean = train_feature[~np.isnan(train_feature)]
            test_clean = test_feature[~np.isnan(test_feature)]

            if len(train_clean) > 0 and len(test_clean) > 0:
                # Kolmogorov-Smirnov test
                ks_stat, p_value = stats.ks_2samp(train_clean, test_clean)

                drift_report["drift_scores"].append(
                    {
                        "feature_index": i,
                        "ks_statistic": ks_stat,
                        "p_value": p_value,
                        "has_drift": p_value < threshold,
                    }
                )

                if p_value < threshold:
                    drift_report["features_with_drift"].append(i)

        return drift_report


# Convenience functions
def quick_load_dataset(
    name: str, cache_dir: str = "./data", **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Quick load common datasets
    """
    downloader = DatasetDownloader(cache_dir)

    if name.lower() == "iris":
        return downloader.load_iris()
    elif name.lower() == "wine":
        return downloader.load_wine()
    elif name.lower() == "boston":
        return downloader.load_boston_housing()
    else:
        raise ValueError(f"Unknown dataset: {name}")


def prepare_data_for_training(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    val_size: float = 0.1,
    standardize: bool = True,
    remove_outliers: bool = False,
    feature_selection: Optional[str] = None,
    random_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    One-stop function to prepare data for training
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Remove outliers if requested
    if remove_outliers:
        X, outlier_mask = DataProcessor.remove_outliers(X)
        y = y[~outlier_mask]

    # Feature selection
    if feature_selection:
        if feature_selection == "variance":
            X, feature_mask = FeatureSelector.variance_threshold(X)
        elif feature_selection == "correlation":
            X, feature_mask = FeatureSelector.correlation_filter(X)
        elif feature_selection.startswith("univariate"):
            k = int(feature_selection.split("_")[1]) if "_" in feature_selection else 10
            X, feature_mask = FeatureSelector.univariate_selection(X, y, k=k)

    # Split data
    n_samples = len(X)
    indices = np.random.permutation(n_samples)

    test_split = int(n_samples * test_size)
    val_split = int(n_samples * val_size)

    test_indices = indices[:test_split]
    val_indices = indices[test_split : test_split + val_split]
    train_indices = indices[test_split + val_split :]

    X_train, X_val, X_test = X[train_indices], X[val_indices], X[test_indices]
    y_train, y_val, y_test = y[train_indices], y[val_indices], y[test_indices]

    # Standardize if requested
    stats = None
    if standardize:
        X_train, stats = DataProcessor.standardize(X_train, return_stats=True)
        X_val = (X_val - stats["mean"]) / stats["std"]
        X_test = (X_test - stats["mean"]) / stats["std"]

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "preprocessing_stats": stats,
        "feature_mask": feature_mask if feature_selection else None,
    }
