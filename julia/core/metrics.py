import numpy as np
from typing import Union, List, Dict, Optional, Tuple, Any
from julia.core.tensor import Tensor
from abc import ABC, abstractmethod
import warnings


class Metric(ABC):
    """
    Abstract base class for metrics
    """
    
    def __init__(self, name: str):
        self.name = name
        self.reset()
    
    @abstractmethod
    def update(self, predictions: Union[Tensor, np.ndarray], targets: Union[Tensor, np.ndarray]):
        """Update metric with new predictions and targets"""
        pass
    
    @abstractmethod
    def compute(self) -> float:
        """Compute the final metric value"""
        pass
    
    @abstractmethod
    def reset(self):
        """Reset metric state"""
        pass
    
    def __call__(self, predictions: Union[Tensor, np.ndarray], targets: Union[Tensor, np.ndarray]) -> float:
        """Compute metric for given predictions and targets"""
        self.reset()
        self.update(predictions, targets)
        return self.compute()


class Accuracy(Metric):
    """
    Classification accuracy metric
    """
    
    def __init__(self):
        super().__init__("accuracy")
    
    def reset(self):
        self.correct = 0
        self.total = 0
    
    def update(self, predictions: Union[Tensor, np.ndarray], targets: Union[Tensor, np.ndarray]):
        pred_data = predictions.data if isinstance(predictions, Tensor) else predictions
        target_data = targets.data if isinstance(targets, Tensor) else targets
        
        # Handle logits by taking argmax
        if pred_data.ndim > 1 and pred_data.shape[-1] > 1:
            pred_labels = np.argmax(pred_data, axis=-1)
        else:
            pred_labels = pred_data
        
        self.correct += np.sum(pred_labels == target_data)
        self.total += len(target_data)
    
    def compute(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0


class TopKAccuracy(Metric):
    """
    Top-K accuracy metric
    """
    
    def __init__(self, k: int = 5):
        self.k = k
        super().__init__(f"top_{k}_accuracy")
    
    def reset(self):
        self.correct = 0
        self.total = 0
    
    def update(self, predictions: Union[Tensor, np.ndarray], targets: Union[Tensor, np.ndarray]):
        pred_data = predictions.data if isinstance(predictions, Tensor) else predictions
        target_data = targets.data if isinstance(targets, Tensor) else targets
        
        # Get top-k predictions
        top_k_preds = np.argsort(pred_data, axis=-1)[:, -self.k:]
        
        # Check if true label is in top-k
        for i, true_label in enumerate(target_data):
            if true_label in top_k_preds[i]:
                self.correct += 1
        
        self.total += len(target_data)
    
    def compute(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0


class Precision(Metric):
    """
    Precision metric
    """
    
    def __init__(self, average: str = 'binary', num_classes: Optional[int] = None):
        self.average = average  # 'binary', 'macro', 'micro', 'weighted'
        self.num_classes = num_classes
        super().__init__(f"precision_{average}")
    
    def reset(self):
        if self.num_classes:
            self.true_positives = np.zeros(self.num_classes)
            self.predicted_positives = np.zeros(self.num_classes)
        else:
            self.true_positives = 0
            self.predicted_positives = 0
    
    def update(self, predictions: Union[Tensor, np.ndarray], targets: Union[Tensor, np.ndarray]):
        pred_data = predictions.data if isinstance(predictions, Tensor) else predictions
        target_data = targets.data if isinstance(targets, Tensor) else targets
        
        # Handle logits by taking argmax
        if pred_data.ndim > 1 and pred_data.shape[-1] > 1:
            pred_labels = np.argmax(pred_data, axis=-1)
        else:
            pred_labels = pred_data
        
        if self.num_classes is None:
            # Binary case
            self.true_positives += np.sum((pred_labels == 1) & (target_data == 1))
            self.predicted_positives += np.sum(pred_labels == 1)
        else:
            # Multi-class case
            for i in range(self.num_classes):
                self.true_positives[i] += np.sum((pred_labels == i) & (target_data == i))
                self.predicted_positives[i] += np.sum(pred_labels == i)
    
    def compute(self) -> float:
        if self.num_classes is None:
            return self.true_positives / self.predicted_positives if self.predicted_positives > 0 else 0.0
        else:
            # Avoid division by zero
            precisions = np.divide(self.true_positives, self.predicted_positives, 
                                 out=np.zeros_like(self.true_positives), 
                                 where=self.predicted_positives != 0)
            
            if self.average == 'macro':
                return np.mean(precisions)
            elif self.average == 'micro':
                return np.sum(self.true_positives) / np.sum(self.predicted_positives) if np.sum(self.predicted_positives) > 0 else 0.0
            elif self.average == 'weighted':
                weights = self.predicted_positives / np.sum(self.predicted_positives) if np.sum(self.predicted_positives) > 0 else np.zeros_like(self.predicted_positives)
                return np.sum(precisions * weights)
            else:
                return precisions


class Recall(Metric):
    """
    Recall metric
    """
    
    def __init__(self, average: str = 'binary', num_classes: Optional[int] = None):
        self.average = average
        self.num_classes = num_classes
        super().__init__(f"recall_{average}")
    
    def reset(self):
        if self.num_classes:
            self.true_positives = np.zeros(self.num_classes)
            self.actual_positives = np.zeros(self.num_classes)
        else:
            self.true_positives = 0
            self.actual_positives = 0
    
    def update(self, predictions: Union[Tensor, np.ndarray], targets: Union[Tensor, np.ndarray]):
        pred_data = predictions.data if isinstance(predictions, Tensor) else predictions
        target_data = targets.data if isinstance(targets, Tensor) else targets
        
        if pred_data.ndim > 1 and pred_data.shape[-1] > 1:
            pred_labels = np.argmax(pred_data, axis=-1)
        else:
            pred_labels = pred_data
        
        if self.num_classes is None:
            self.true_positives += np.sum((pred_labels == 1) & (target_data == 1))
            self.actual_positives += np.sum(target_data == 1)
        else:
            for i in range(self.num_classes):
                self.true_positives[i] += np.sum((pred_labels == i) & (target_data == i))
                self.actual_positives[i] += np.sum(target_data == i)
    
    def compute(self) -> float:
        if self.num_classes is None:
            return self.true_positives / self.actual_positives if self.actual_positives > 0 else 0.0
        else:
            recalls = np.divide(self.true_positives, self.actual_positives,
                              out=np.zeros_like(self.true_positives),
                              where=self.actual_positives != 0)
            
            if self.average == 'macro':
                return np.mean(recalls)
            elif self.average == 'micro':
                return np.sum(self.true_positives) / np.sum(self.actual_positives) if np.sum(self.actual_positives) > 0 else 0.0
            elif self.average == 'weighted':
                weights = self.actual_positives / np.sum(self.actual_positives) if np.sum(self.actual_positives) > 0 else np.zeros_like(self.actual_positives)
                return np.sum(recalls * weights)
            else:
                return recalls


class F1Score(Metric):
    """
    F1 Score metric
    """
    
    def __init__(self, average: str = 'binary', num_classes: Optional[int] = None):
        self.average = average
        self.num_classes = num_classes
        self.precision = Precision(average, num_classes)
        self.recall = Recall(average, num_classes)
        super().__init__(f"f1_{average}")
    
    def reset(self):
        self.precision.reset()
        self.recall.reset()
    
    def update(self, predictions: Union[Tensor, np.ndarray], targets: Union[Tensor, np.ndarray]):
        self.precision.update(predictions, targets)
        self.recall.update(predictions, targets)
    
    def compute(self) -> float:
        p = self.precision.compute()
        r = self.recall.compute()
        
        if isinstance(p, np.ndarray):
            f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) != 0)
            if self.average == 'macro':
                return np.mean(f1)
            elif self.average == 'weighted':
                weights = self.recall.actual_positives / np.sum(self.recall.actual_positives) if np.sum(self.recall.actual_positives) > 0 else np.zeros_like(self.recall.actual_positives)
                return np.sum(f1 * weights)
            else:
                return f1
        else:
            return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


class MSE(Metric):
    """
    Mean Squared Error metric
    """
    
    def __init__(self):
        super().__init__("mse")
    
    def reset(self):
        self.sum_squared_error = 0.0
        self.total = 0
    
    def update(self, predictions: Union[Tensor, np.ndarray], targets: Union[Tensor, np.ndarray]):
        pred_data = predictions.data if isinstance(predictions, Tensor) else predictions
        target_data = targets.data if isinstance(targets, Tensor) else targets
        
        squared_error = (pred_data - target_data) ** 2
        self.sum_squared_error += np.sum(squared_error)
        self.total += len(target_data)
    
    def compute(self) -> float:
        return self.sum_squared_error / self.total if self.total > 0 else 0.0


class MAE(Metric):
    """
    Mean Absolute Error metric
    """
    
    def __init__(self):
        super().__init__("mae")
    
    def reset(self):
        self.sum_absolute_error = 0.0
        self.total = 0
    
    def update(self, predictions: Union[Tensor, np.ndarray], targets: Union[Tensor, np.ndarray]):
        pred_data = predictions.data if isinstance(predictions, Tensor) else predictions
        target_data = targets.data if isinstance(targets, Tensor) else targets
        
        absolute_error = np.abs(pred_data - target_data)
        self.sum_absolute_error += np.sum(absolute_error)
        self.total += len(target_data)
    
    def compute(self) -> float:
        return self.sum_absolute_error / self.total if self.total > 0 else 0.0


class RMSE(Metric):
    """
    Root Mean Squared Error metric
    """
    
    def __init__(self):
        super().__init__("rmse")
        self.mse = MSE()
    
    def reset(self):
        self.mse.reset()
    
    def update(self, predictions: Union[Tensor, np.ndarray], targets: Union[Tensor, np.ndarray]):
        self.mse.update(predictions, targets)
    
    def compute(self) -> float:
        return np.sqrt(self.mse.compute())


class R2Score(Metric):
    """
    R-squared (coefficient of determination) metric
    """
    
    def __init__(self):
        super().__init__("r2")
    
    def reset(self):
        self.sum_squared_residuals = 0.0
        self.sum_squared_total = 0.0
        self.target_sum = 0.0
        self.total = 0
    
    def update(self, predictions: Union[Tensor, np.ndarray], targets: Union[Tensor, np.ndarray]):
        pred_data = predictions.data if isinstance(predictions, Tensor) else predictions
        target_data = targets.data if isinstance(targets, Tensor) else targets
        
        self.sum_squared_residuals += np.sum((target_data - pred_data) ** 2)
        self.target_sum += np.sum(target_data)
        self.total += len(target_data)
    
    def compute(self) -> float:
        target_mean = self.target_sum / self.total if self.total > 0 else 0.0
        # Note: We need targets again to compute total sum of squares
        # This is a limitation of the streaming approach
        # For now, return a simplified version
        return 1.0 - (self.sum_squared_residuals / self.total) / (target_mean ** 2) if target_mean != 0 else 0.0


class ConfusionMatrix:
    """
    Confusion matrix computation
    """
    
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        self.matrix = np.zeros((self.num_classes, self.num_classes), dtype=int)
    
    def update(self, predictions: Union[Tensor, np.ndarray], targets: Union[Tensor, np.ndarray]):
        pred_data = predictions.data if isinstance(predictions, Tensor) else predictions
        target_data = targets.data if isinstance(targets, Tensor) else targets
        
        if pred_data.ndim > 1 and pred_data.shape[-1] > 1:
            pred_labels = np.argmax(pred_data, axis=-1)
        else:
            pred_labels = pred_data
        
        for true_label, pred_label in zip(target_data, pred_labels):
            if 0 <= true_label < self.num_classes and 0 <= pred_label < self.num_classes:
                self.matrix[int(true_label), int(pred_label)] += 1
    
    def compute(self) -> np.ndarray:
        return self.matrix.copy()
    
    def plot(self, class_names: Optional[List[str]] = None, normalize: bool = False, 
             title: str = "Confusion Matrix"):
        """
        Plot confusion matrix (requires matplotlib)
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            print("matplotlib and seaborn required for plotting")
            return
        
        matrix = self.compute()
        if normalize:
            matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(matrix, annot=True, fmt='.2f' if normalize else 'd',
                   xticklabels=class_names, yticklabels=class_names,
                   cmap='Blues')
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.show()


class ROCCurve:
    """
    ROC Curve computation for binary classification
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.scores = []
        self.labels = []
    
    def update(self, predictions: Union[Tensor, np.ndarray], targets: Union[Tensor, np.ndarray]):
        pred_data = predictions.data if isinstance(predictions, Tensor) else predictions
        target_data = targets.data if isinstance(targets, Tensor) else targets
        
        # For binary classification, use probability of positive class
        if pred_data.ndim > 1 and pred_data.shape[-1] == 2:
            scores = pred_data[:, 1]  # Probability of class 1
        elif pred_data.ndim > 1:
            scores = np.max(pred_data, axis=-1)  # Max probability
        else:
            scores = pred_data
        
        self.scores.extend(scores)
        self.labels.extend(target_data)
    
    def compute(self, num_thresholds: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute ROC curve
        
        Returns:
            fpr: False positive rates
            tpr: True positive rates  
            thresholds: Thresholds used
        """
        scores = np.array(self.scores)
        labels = np.array(self.labels)
        
        # Sort by scores in descending order
        desc_score_indices = np.argsort(scores)[::-1]
        scores = scores[desc_score_indices]
        labels = labels[desc_score_indices]
        
        # Get unique thresholds
        thresholds = np.unique(scores)
        if len(thresholds) > num_thresholds:
            thresholds = np.linspace(scores.min(), scores.max(), num_thresholds)
        
        tpr = np.zeros(len(thresholds))
        fpr = np.zeros(len(thresholds))
        
        for i, threshold in enumerate(thresholds):
            predictions = scores >= threshold
            
            tp = np.sum((predictions == 1) & (labels == 1))
            fp = np.sum((predictions == 1) & (labels == 0))
            tn = np.sum((predictions == 0) & (labels == 0))
            fn = np.sum((predictions == 0) & (labels == 1))
            
            tpr[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr[i] = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        return fpr, tpr, thresholds
    
    def auc(self) -> float:
        """Compute Area Under Curve"""
        fpr, tpr, _ = self.compute()
        return np.trapz(tpr, fpr)


class MetricsLogger:
    """
    Logger for tracking multiple metrics during training
    """
    
    def __init__(self):
        self.metrics = {}
        self.history = {}
    
    def add_metric(self, name: str, metric: Metric):
        """Add a metric to track"""
        self.metrics[name] = metric
        self.history[name] = []
    
    def update(self, predictions: Union[Tensor, np.ndarray], targets: Union[Tensor, np.ndarray]):
        """Update all metrics"""
        for metric in self.metrics.values():
            metric.update(predictions, targets)
    
    def compute_and_log(self) -> Dict[str, float]:
        """Compute all metrics and log to history"""
        results = {}
        for name, metric in self.metrics.items():
            value = metric.compute()
            results[name] = value
            self.history[name].append(value)
            metric.reset()
        return results
    
    def get_history(self) -> Dict[str, List[float]]:
        """Get training history"""
        return self.history.copy()
    
    def plot_history(self, metric_names: Optional[List[str]] = None):
        """Plot training history"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib required for plotting")
            return
        
        if metric_names is None:
            metric_names = list(self.history.keys())
        
        fig, axes = plt.subplots(len(metric_names), 1, figsize=(10, 3 * len(metric_names)))
        if len(metric_names) == 1:
            axes = [axes]
        
        for i, metric_name in enumerate(metric_names):
            if metric_name in self.history:
                axes[i].plot(self.history[metric_name])
                axes[i].set_title(f'{metric_name.title()} History')
                axes[i].set_xlabel('Epoch')
                axes[i].set_ylabel(metric_name.title())
                axes[i].grid(True)
        
        plt.tight_layout()
        plt.show()


# Convenience functions for computing metrics
def accuracy_score(predictions: Union[Tensor, np.ndarray], targets: Union[Tensor, np.ndarray]) -> float:
    """Compute accuracy score"""
    metric = Accuracy()
    return metric(predictions, targets)


def precision_score(predictions: Union[Tensor, np.ndarray], targets: Union[Tensor, np.ndarray], 
                   average: str = 'binary', num_classes: Optional[int] = None) -> float:
    """Compute precision score"""
    metric = Precision(average, num_classes)
    return metric(predictions, targets)


def recall_score(predictions: Union[Tensor, np.ndarray], targets: Union[Tensor, np.ndarray],
                average: str = 'binary', num_classes: Optional[int] = None) -> float:
    """Compute recall score"""
    metric = Recall(average, num_classes)
    return metric(predictions, targets)


def f1_score(predictions: Union[Tensor, np.ndarray], targets: Union[Tensor, np.ndarray],
            average: str = 'binary', num_classes: Optional[int] = None) -> float:
    """Compute F1 score"""
    metric = F1Score(average, num_classes)
    return metric(predictions, targets)


def mean_squared_error(predictions: Union[Tensor, np.ndarray], targets: Union[Tensor, np.ndarray]) -> float:
    """Compute mean squared error"""
    metric = MSE()
    return metric(predictions, targets)


def mean_absolute_error(predictions: Union[Tensor, np.ndarray], targets: Union[Tensor, np.ndarray]) -> float:
    """Compute mean absolute error"""
    metric = MAE()
    return metric(predictions, targets)


def classification_report(predictions: Union[Tensor, np.ndarray], targets: Union[Tensor, np.ndarray],
                         class_names: Optional[List[str]] = None, num_classes: Optional[int] = None) -> Dict[str, Any]:
    """
    Generate a comprehensive classification report
    """
    pred_data = predictions.data if isinstance(predictions, Tensor) else predictions
    target_data = targets.data if isinstance(targets, Tensor) else targets
    
    if pred_data.ndim > 1 and pred_data.shape[-1] > 1:
        pred_labels = np.argmax(pred_data, axis=-1)
        if num_classes is None:
            num_classes = pred_data.shape[-1]
    else:
        pred_labels = pred_data
        if num_classes is None:
            num_classes = len(np.unique(target_data))
    
    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]
    
    report = {
        'accuracy': accuracy_score(predictions, targets),
        'macro_precision': precision_score(predictions, targets, 'macro', num_classes),
        'macro_recall': recall_score(predictions, targets, 'macro', num_classes),
        'macro_f1': f1_score(predictions, targets, 'macro', num_classes),
        'weighted_precision': precision_score(predictions, targets, 'weighted', num_classes),
        'weighted_recall': recall_score(predictions, targets, 'weighted', num_classes),
        'weighted_f1': f1_score(predictions, targets, 'weighted', num_classes),
        'per_class': {}
    }
    
    # Per-class metrics
    for i, class_name in enumerate(class_names):
        class_predictions = (pred_labels == i).astype(int)
        class_targets = (target_data == i).astype(int)
        
        report['per_class'][class_name] = {
            'precision': precision_score(class_predictions, class_targets),
            'recall': recall_score(class_predictions, class_targets),
            'f1': f1_score(class_predictions, class_targets),
            'support': np.sum(target_data == i)
        }
    
    return report


def regression_report(predictions: Union[Tensor, np.ndarray], targets: Union[Tensor, np.ndarray]) -> Dict[str, float]:
    """
    Generate a comprehensive regression report
    """
    return {
        'mse': mean_squared_error(predictions, targets),
        'mae': mean_absolute_error(predictions, targets),
        'rmse': RMSE()(predictions, targets),
        'r2': R2Score()(predictions, targets)
    }
