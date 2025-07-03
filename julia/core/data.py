import numpy as np
import time
import random
from typing import Union, List, Tuple, Optional, Callable, Iterator, Any, Dict, Sequence
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from julia.core.tensor import Tensor


class Dataset(ABC):
    """
    Abstract base class for all datasets
    """
    @abstractmethod
    def __len__(self) -> int:
        pass
    
    @abstractmethod
    def __getitem__(self, index: int) -> Any:
        pass


class TensorDataset(Dataset):
    """
    Each sample is retrieved by indexing tensors along the first dimension
    """
    
    def __init__(self, *tensors: Tensor):
        assert all(tensors[0].shape[0] == tensor.shape[0] for tensor in tensors), \
            "All tensors must have same size in first dimension"
        self.tensors = tensors
    
    def __len__(self) -> int:
        return self.tensors[0].shape[0]
    
    def __getitem__(self, index: int) -> Tuple[Tensor, ...]:
        if isinstance(index, slice):
            return tuple(Tensor(tensor.data[index]) for tensor in self.tensors)
        return tuple(Tensor(tensor.data[index]) for tensor in self.tensors)

class ArrayDataset(Dataset):
    """
    Dataset wrapping numpy arrays
    """
    
    def __init__(self, *arrays: np.ndarray):
        assert all(arrays[0].shape[0] == array.shape[0] for array in arrays), \
            "All arrays must have same size in first dimension"
        self.arrays = arrays
    
    def __len__(self) -> int:
        return self.arrays[0].shape[0]
    
    def __getitem__(self, index: int) -> Tuple[np.ndarray, ...]:
        if isinstance(index, slice):
            return tuple(array[index] for array in self.arrays)
        return tuple(array[index] for array in self.arrays)


class ListDataset(Dataset):
    def __init__(self, samples: List[Any]):
        self.samples = samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, index: int) -> Any:
        return self.samples[index]


class CSVDataset(Dataset):
    """
    Dataset for loading CSV files
    """
    
    def __init__(self, file_path: str, header: bool = True, delimiter: str = ',',
                 target_column: Optional[Union[str, int]] = None,
                 feature_columns: Optional[List[Union[str, int]]] = None,
                 dtype: np.dtype = np.float32):
        self.file_path = file_path
        self.header = header
        self.delimiter = delimiter
        self.dtype = dtype
        
        self.data = np.loadtxt(file_path, delimiter=delimiter, skiprows=1 if header else 0, dtype=dtype)
        
        if header:
            with open(file_path, 'r') as f:
                header_line = f.readline().strip()
                self.column_names = header_line.split(delimiter)
        else:
            self.column_names = [f"col_{i}" for i in range(self.data.shape[1])]
        
        # Handle target and feature selection
        if target_column is not None:
            if isinstance(target_column, str):
                self.target_idx = self.column_names.index(target_column)
            else:
                self.target_idx = target_column
            self.targets = self.data[:, self.target_idx]
        else:
            self.targets = None
            self.target_idx = None
        
        if feature_columns is not None:
            if isinstance(feature_columns[0], str):
                self.feature_indices = [self.column_names.index(col) for col in feature_columns]
            else:
                self.feature_indices = feature_columns
            self.features = self.data[:, self.feature_indices]
        else:
            # Use all columns except target
            if self.target_idx is not None:
                self.feature_indices = [i for i in range(self.data.shape[1]) if i != self.target_idx]
                self.features = self.data[:, self.feature_indices]
            else:
                self.features = self.data
                self.feature_indices = list(range(self.data.shape[1]))
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index: int) -> Tuple[np.ndarray, ...]:
        if self.targets is not None:
            return self.features[index], self.targets[index]
        return self.features[index]


class Transform(ABC):
    """
    Abstract base class for data transformations
    """
    
    @abstractmethod
    def __call__(self, sample: Any) -> Any:
        pass


class Compose(Transform):
    """
    Compose multiple transforms together
    """
    
    def __init__(self, transforms: List[Transform]):
        self.transforms = transforms
    
    def __call__(self, sample: Any) -> Any:
        for transform in self.transforms:
            sample = transform(sample)
        return sample


class ToTensor(Transform):
    """
    Convert numpy arrays or lists to Tensors
    """
    
    def __init__(self, dtype: Optional[np.dtype] = None):
        self.dtype = dtype
    
    def __call__(self, sample: Any) -> Any:
        if isinstance(sample, (list, tuple)):
            return tuple(Tensor(np.array(item, dtype=self.dtype)) if not isinstance(item, Tensor) 
                        else item for item in sample)
        elif isinstance(sample, np.ndarray):
            return Tensor(sample.astype(self.dtype) if self.dtype else sample)
        elif isinstance(sample, (int, float)):
            return Tensor(np.array(sample, dtype=self.dtype or np.float32))
        return sample


class Normalize(Transform):
    """
    Normalize tensor with mean and std
    """
    
    def __init__(self, mean: Union[float, List[float]], std: Union[float, List[float]]):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
    
    def __call__(self, sample: Any) -> Any:
        if isinstance(sample, (list, tuple)):
            result = []
            for item in sample:
                if isinstance(item, (Tensor, np.ndarray)):
                    data = item.data if isinstance(item, Tensor) else item
                    normalized = (data - self.mean) / self.std
                    result.append(Tensor(normalized) if isinstance(item, Tensor) else normalized)
                else:
                    result.append(item)
            return tuple(result)
        elif isinstance(sample, (Tensor, np.ndarray)):
            data = sample.data if isinstance(sample, Tensor) else sample
            normalized = (data - self.mean) / self.std
            return Tensor(normalized) if isinstance(sample, Tensor) else normalized
        return sample


class RandomHorizontalFlip(Transform):
    """
    Randomly flip tensor horizontally
    """
    
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, sample: Any) -> Any:
        if random.random() < self.p:
            if isinstance(sample, (list, tuple)):
                result = []
                for item in sample:
                    if isinstance(item, (Tensor, np.ndarray)):
                        data = item.data if isinstance(item, Tensor) else item
                        if len(data.shape) >= 2:
                            flipped = np.flip(data, axis=-1)
                            result.append(Tensor(flipped) if isinstance(item, Tensor) else flipped)
                        else:
                            result.append(item)
                    else:
                        result.append(item)
                return tuple(result)
            elif isinstance(sample, (Tensor, np.ndarray)):
                data = sample.data if isinstance(sample, Tensor) else sample
                if len(data.shape) >= 2:
                    flipped = np.flip(data, axis=-1)
                    return Tensor(flipped) if isinstance(sample, Tensor) else flipped
        return sample


class RandomRotation(Transform):
    """
    Randomly rotate 2D tensor
    """
    
    def __init__(self, degrees: Union[float, Tuple[float, float]]):
        if isinstance(degrees, (int, float)):
            self.degrees = (-degrees, degrees)
        else:
            self.degrees = degrees
    
    def __call__(self, sample: Any) -> Any:
        angle = random.uniform(self.degrees[0], self.degrees[1])
        # Simple rotation for 2D data - would need more sophisticated implementation for images
        if isinstance(sample, (Tensor, np.ndarray)):
            data = sample.data if isinstance(sample, Tensor) else sample
            if len(data.shape) == 2:
                # Simple 90-degree rotations only for now
                k = int(angle // 90)
                rotated = np.rot90(data, k)
                return Tensor(rotated) if isinstance(sample, Tensor) else rotated
        return sample


class AddNoise(Transform):
    """
    Add Gaussian noise to tensor
    """
    
    def __init__(self, mean: float = 0.0, std: float = 0.1):
        self.mean = mean
        self.std = std
    
    def __call__(self, sample: Any) -> Any:
        if isinstance(sample, (Tensor, np.ndarray)):
            data = sample.data if isinstance(sample, Tensor) else sample
            noise = np.random.normal(self.mean, self.std, data.shape).astype(data.dtype)
            noisy_data = data + noise
            return Tensor(noisy_data) if isinstance(sample, Tensor) else noisy_data
        return sample


class Sampler(ABC):
    """
    Base class for data samplers
    """
    
    def __init__(self, dataset: Dataset):
        self.dataset = dataset
    
    @abstractmethod
    def __iter__(self) -> Iterator[int]:
        pass
    
    def __len__(self) -> int:
        return len(self.dataset)


class SequentialSampler(Sampler):
    """
    Sequential sampler
    """
    
    def __iter__(self) -> Iterator[int]:
        return iter(range(len(self.dataset)))


class RandomSampler(Sampler):
    """
    Random sampler
    """
    
    def __init__(self, dataset: Dataset, replacement: bool = False, num_samples: Optional[int] = None):
        super().__init__(dataset)
        self.replacement = replacement
        self.num_samples = num_samples or len(dataset)
    
    def __iter__(self) -> Iterator[int]:
        if self.replacement:
            return iter(np.random.choice(len(self.dataset), self.num_samples, replace=True))
        else:
            indices = list(range(len(self.dataset)))
            random.shuffle(indices)
            return iter(indices[:self.num_samples])


class BatchSampler(Sampler):
    """
    Batch sampler that yields batches of indices
    """
    
    def __init__(self, sampler: Sampler, batch_size: int, drop_last: bool = False):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
    
    def __iter__(self) -> Iterator[List[int]]:
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch and not self.drop_last:
            yield batch
    
    def __len__(self) -> int:
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size


class SubsetRandomSampler(Sampler):
    """
    Sample randomly from a subset of indices
    """
    
    def __init__(self, indices: Sequence[int]):
        self.indices = list(indices)
    
    def __iter__(self) -> Iterator[int]:
        indices = self.indices.copy()
        random.shuffle(indices)
        return iter(indices)
    
    def __len__(self) -> int:
        return len(self.indices)


def collate_fn_default(batch: List[Any]) -> Any:
    """
    Default collate function that stacks samples into batches
    """
    if isinstance(batch[0], (list, tuple)):
        return tuple(collate_fn_default([sample[i] for sample in batch]) 
                    for i in range(len(batch[0])))
    
    elif isinstance(batch[0], Tensor):
        # Stack tensors
        return Tensor(np.stack([sample.data for sample in batch])) # TODO built in stack funciton
    
    elif isinstance(batch[0], np.ndarray):
        # Stack numpy arrays
        return np.stack(batch)
    
    elif isinstance(batch[0], (int, float)):
        # Convert to numpy array
        return np.array(batch)
    
    else:
        # Return as list for other types
        return batch


class DataLoader:
    """
    Data loader for batching and iterating over datasets
    """
    
    def __init__(self, 
                 dataset: Dataset,
                 batch_size: int = 1,
                 shuffle: bool = False,
                 sampler: Optional[Sampler] = None,
                 batch_sampler: Optional[BatchSampler] = None,
                 num_workers: int = 0,
                 collate_fn: Optional[Callable] = None,
                 pin_memory: bool = False,
                 drop_last: bool = False,
                 timeout: float = 0.0,
                 transform: Optional[Transform] = None):
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collate_fn = collate_fn or collate_fn_default
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.timeout = timeout
        self.transform = transform
        
        if batch_sampler is not None:
            if batch_size != 1 or shuffle or sampler is not None or drop_last:
                raise ValueError("batch_sampler option is mutually exclusive with "
                               "batch_size, shuffle, sampler, and drop_last")
            self.batch_sampler = batch_sampler
        else:
            if sampler is None:
                if shuffle:
                    sampler = RandomSampler(dataset)
                else:
                    sampler = SequentialSampler(dataset)
            self.batch_sampler = BatchSampler(sampler, batch_size, drop_last)
        
        self._worker_pool = None
        self._shutdown_workers = False
    
    def __iter__(self):
        if self.num_workers == 0:
            return self._single_worker_iter()
        else:
            return self._multi_worker_iter()
    
    def __len__(self) -> int:
        return len(self.batch_sampler)
    
    def _single_worker_iter(self):
        """Single-threaded iteration"""
        for batch_indices in self.batch_sampler:
            batch = []
            for idx in batch_indices:
                sample = self.dataset[idx]
                if self.transform:
                    sample = self.transform(sample)
                batch.append(sample)
            yield self.collate_fn(batch)
    
    def _multi_worker_iter(self):
        """Multi-threaded iteration"""
        if self._worker_pool is None:
            self._worker_pool = ThreadPoolExecutor(max_workers=self.num_workers)
        
        def fetch_batch(batch_indices):
            batch = []
            for idx in batch_indices:
                sample = self.dataset[idx]
                if self.transform:
                    sample = self.transform(sample)
                batch.append(sample)
            return self.collate_fn(batch)
        
        try:
            # Submit batches to thread pool
            futures = []
            for batch_indices in self.batch_sampler:
                future = self._worker_pool.submit(fetch_batch, batch_indices)
                futures.append(future)
            
            # Yield results as they complete
            for future in futures:
                yield future.result(timeout=self.timeout if self.timeout > 0 else None)
                
        except Exception as e:
            self._shutdown_workers = True
            raise e
    
    def __del__(self):
        if self._worker_pool is not None:
            self._worker_pool.shutdown(wait=False)


class DataSplitter:
    """
    Utility class for splitting datasets
    """
    
    @staticmethod
    def train_val_split(dataset: Dataset, 
                       val_ratio: float = 0.2, 
                       shuffle: bool = True,
                       random_seed: Optional[int] = None) -> Tuple[Dataset, Dataset]:
        """
        Split dataset into train and validation sets
        """
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        
        if shuffle:
            random.shuffle(indices)
        
        val_size = int(dataset_size * val_ratio)
        train_indices = indices[val_size:]
        val_indices = indices[:val_size]
        
        train_dataset = SubsetDataset(dataset, train_indices)
        val_dataset = SubsetDataset(dataset, val_indices)
        
        return train_dataset, val_dataset
    
    @staticmethod
    def train_val_test_split(dataset: Dataset,
                           val_ratio: float = 0.15,
                           test_ratio: float = 0.15,
                           shuffle: bool = True,
                           random_seed: Optional[int] = None) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Split dataset into train, validation, and test sets
        """
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        
        if shuffle:
            random.shuffle(indices)
        
        test_size = int(dataset_size * test_ratio)
        val_size = int(dataset_size * val_ratio)
        
        test_indices = indices[:test_size]
        val_indices = indices[test_size:test_size + val_size]
        train_indices = indices[test_size + val_size:]
        
        train_dataset = SubsetDataset(dataset, train_indices)
        val_dataset = SubsetDataset(dataset, val_indices)
        test_dataset = SubsetDataset(dataset, test_indices)
        
        return train_dataset, val_dataset, test_dataset


class SubsetDataset(Dataset):
    """
    Subset of a dataset at specified indices
    """
    
    def __init__(self, dataset: Dataset, indices: Sequence[int]):
        self.dataset = dataset
        self.indices = list(indices)
    
    def __getitem__(self, index: int) -> Any:
        return self.dataset[self.indices[index]]
    
    def __len__(self) -> int:
        return len(self.indices)


class ConcatDataset(Dataset):
    """
    Dataset to concatenate multiple datasets
    """
    
    def __init__(self, datasets: Sequence[Dataset]):
        self.datasets = list(datasets)
        self.cumulative_sizes = self._get_cumulative_sizes()
    
    def _get_cumulative_sizes(self) -> List[int]:
        cumulative_sizes = [0]
        for dataset in self.datasets:
            cumulative_sizes.append(cumulative_sizes[-1] + len(dataset))
        return cumulative_sizes[1:]
    
    def __len__(self) -> int:
        return self.cumulative_sizes[-1]
    
    def __getitem__(self, index: int) -> Any:
        if index < 0:
            if -index > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            index = len(self) + index
        dataset_idx = 0
        for cumulative_size in self.cumulative_sizes:
            if index < cumulative_size:
                break
            dataset_idx += 1
        
        if dataset_idx == 0:
            sample_idx = index
        else:
            sample_idx = index - self.cumulative_sizes[dataset_idx - 1]
        
        return self.datasets[dataset_idx][sample_idx]


class InMemoryDataset(Dataset):
    """
    Dataset that loads all data into memory for fast access
    """
    
    def __init__(self, dataset: Dataset, transform: Optional[Transform] = None):
        self.transform = transform
        print(f"Loading {len(dataset)} samples into memory...")
        self.data = []
        for i in range(len(dataset)):
            sample = dataset[i]
            if transform:
                sample = transform(sample)
            self.data.append(sample)
        print("Loading complete!")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index: int) -> Any:
        return self.data[index]


class DatasetCache:
    """
    LRU Cache for dataset samples
    """
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache = {}
        self.access_order = []
    
    def get(self, key: Any) -> Optional[Any]:
        if key in self.cache:
            # Move to end (most recently used)
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None
    
    def put(self, key: Any, value: Any):
        if key in self.cache:
            # Update existing
            self.cache[key] = value
            self.access_order.remove(key)
            self.access_order.append(key)
        else:
            # Add new
            if len(self.cache) >= self.max_size:
                # Remove least recently used
                oldest = self.access_order.pop(0)
                del self.cache[oldest]
            
            self.cache[key] = value
            self.access_order.append(key)


class CachedDataset(Dataset):
    """
    Dataset with LRU caching
    """
    
    def __init__(self, dataset: Dataset, cache_size: int = 1000, 
                 transform: Optional[Transform] = None):
        self.dataset = dataset
        self.transform = transform
        self.cache = DatasetCache(cache_size)
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, index: int) -> Any:
        cached = self.cache.get(index)
        if cached is not None:
            return cached
        
        sample = self.dataset[index]
        if self.transform:
            sample = self.transform(sample)
        
        self.cache.put(index, sample)
        return sample


class DataLoaderProfiler:
    """
    Profiler for DataLoader performance
    """
    
    def __init__(self, dataloader: DataLoader):
        self.dataloader = dataloader
        self.batch_times = []
        self.total_time = 0
        self.num_batches = 0
    
    def profile(self, num_epochs: int = 1) -> Dict[str, float]:
        """
        Profile the DataLoader for specified epochs
        """
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            for batch_idx, batch in enumerate(self.dataloader):
                batch_time = time.time() - epoch_start
                self.batch_times.append(batch_time)
                self.num_batches += 1
                epoch_start = time.time()
        
        self.total_time = time.time() - start_time
        
        return {
            'total_time': self.total_time,
            'avg_batch_time': np.mean(self.batch_times),
            'min_batch_time': np.min(self.batch_times),
            'max_batch_time': np.max(self.batch_times),
            'std_batch_time': np.std(self.batch_times),
            'batches_per_second': self.num_batches / self.total_time,
            'samples_per_second': (self.num_batches * self.dataloader.batch_size) / self.total_time
        }


# Utility functions for common data tasks

def create_synthetic_regression_data(n_samples: int = 1000, 
                                   n_features: int = 10,
                                   noise: float = 0.1,
                                   random_seed: Optional[int] = None) -> TensorDataset:
    """
    Create synthetic regression dataset
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    true_coef = np.random.randn(n_features).astype(np.float32)
    y = X @ true_coef + noise * np.random.randn(n_samples).astype(np.float32)
    
    return TensorDataset(Tensor(X), Tensor(y))


def create_synthetic_classification_data(n_samples: int = 1000,
                                        n_features: int = 10,
                                        n_classes: int = 2,
                                        random_seed: Optional[int] = None) -> TensorDataset:
    """
    Create synthetic classification dataset
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    
    # Create class centroids
    centroids = np.random.randn(n_classes, n_features).astype(np.float32) * 3
    
    # Assign samples to closest centroid
    distances = np.array([[np.linalg.norm(x - c) for c in centroids] for x in X])
    y = np.argmin(distances, axis=1).astype(np.int64)
    
    return TensorDataset(Tensor(X), Tensor(y))


def load_csv_dataset(file_path: str, 
                    target_column: Optional[Union[str, int]] = None,
                    feature_columns: Optional[List[Union[str, int]]] = None,
                    train_ratio: float = 0.8,
                    val_ratio: float = 0.1,
                    test_ratio: float = 0.1,
                    transform: Optional[Transform] = None,
                    random_seed: Optional[int] = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load CSV dataset and create train/val/test data loaders
    """
    # Load dataset
    dataset = CSVDataset(file_path, target_column=target_column, 
                        feature_columns=feature_columns)
    
    # Apply transforms
    if transform is None:
        transform = ToTensor()
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = DataSplitter.train_val_test_split(
        dataset, val_ratio=val_ratio, test_ratio=test_ratio, 
        shuffle=True, random_seed=random_seed
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, transform=transform)
    
    return train_loader, val_loader, test_loader
