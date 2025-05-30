import numpy as np 
import weakref
from typing import Dict, Optional, Set, Tuple 
from threading import Lock

"""
Memory pools and buffer management 
"""

class MemoryPool:
    """
    Simple memory pool for tensor allocations
    """

    def __init__(self, device: str = "cpu"):
        self.device = device 
        self.free_blocks: Dict[Tuple[int, str], list] = {}
        self.allocated_blocks: Set[int] = set()
        self.lock = Lock()
        self.total_allocated = 0
        self.peak_allocated = 0 

    def allocate(self, shape: Tuple[int, ...], dtype: np.dtype) -> np.ndarray:
       """Allocated array from pool or create a new pool""" 
       with self.lock:
            size = np.prod(shape)
            key = (size, str(dtype))

            if key in self.free_blocks and self.free_blocks[key]:
                array = self.free_blocks[key].pop()
                array = array.reshape(shape)
                self.allocated_blocks.add(id(array))
                return array

            array = np.empty(shape, dtype=dtype)
            self.allocated_blocks.add(id(array))
            self.total_allocated += array.nbytes
            self.peak_allocated = max(self.peak_allocated, self.total_allocated)
            return array 

    def deallocate(self, array: np.ndarray):
        """Return array to pool"""
        with self.lock:
            array_id = id(array)
            if array_id not in self.allocated_blocks:
                return

            self.allocated_blocks.remove(array_id)

            size = array.size
            if size < 1024 * 1024: # Pooling arrays < 1MB 
                key = (size, str(array.dtype))
                if key not in self.free_blocks:
                    self.free_blocks[key] = []

                flat_array = array.flatten()
                self.free_blocks[key].append(flat_array)

                if len(self.free_blocks[key]) > 10:
                    self.free_blocks[key] = self.free_blocks[key][-10:]

    def clear(self):
        """Clear all pooled memory"""
        with self.lock:
            self.free_blocks.clear()
            self.allocated_blocks.clear()
            self.total_allocated = 0

    def stats(self) -> Dict:
        with self.lock:
            return {
                    'total_allocated': self.total_allocated,
                    'peak_allocated': self.peak_allocated,
                    'active_blocks': len(self.allocated_blocks),
                    'pooled_blocks': sum(len(blocks) for blocks in self.free_blocks.values()),
                    'pool_keys': len(self.free_blocks)
            }

class DeviceManager:
    """Management for different devices"""

    def __init__(self):
        self.pools: Dict[str, MemoryPool] = {}
        self.default_device = "cpu"

    def get_pool(self, device: str = None) -> MemoryPool:
        device = device or self.default_device
        if device not in self.pools:
            self.pools[device] = MemoryPool(device)
        return self.pools[device]

    def allocate(self, shape: Tuple[int, ...], dtype: np.dtype, device: str = None) -> np.ndarray:
        pool = self.get_pool(device)
        return pool.allocate(shape, dtype)

    def deallocate(self, array: np.ndarray, device: str = None):
        pool = self.get_pool(device)
        return pool.deallocate(array)

    def clear_all(self):
        for pool in self.pools.values():
            pool.clear()

    def stats(self) -> Dict[str, Dict]:
        return {device: pool.stats() for device, pool in self.pools.items()}

device_manager = DeviceManager()

# TODO auto memory management with default tensors
