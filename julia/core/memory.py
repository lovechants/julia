import numpy as np 
import weakref
from typing import Dict, Optional, Set, Tuple, List 
from threading import Lock

"""
Memory pools and buffer management 
"""

class MemoryPool:
    """
    Improved memory pool for tensor allocations
    """

    def __init__(self, device: str = "cpu"):
        self.device = device 
        # Enhanced free blocks structure: (size, dtype) -> list of arrays
        self.free_blocks: Dict[Tuple[int, str], List[np.ndarray]] = {}
        self.allocated_blocks: Set[int] = set()
        self.lock = Lock()
        self.total_allocated = 0
        self.peak_allocated = 0
        
        # Pool statistics for profiler integration
        self.pool_hits = 0
        self.pool_misses = 0
        
        # Size classes for better pooling efficiency
        self.size_classes = self._compute_size_classes()
        
        # Cleanup tracking
        self.tensor_cleanup_refs: Dict[str, weakref.ref] = {}

    def _compute_size_classes(self) -> List[int]:
        """Compute efficient size classes for common tensor allocations"""
        size_classes = []
        
        # Small tensors: 64B to 8KB (common for small networks)
        size = 64
        while size <= 8192:
            size_classes.append(size)
            size *= 2
        
        # Medium tensors: 16KB to 1MB (common for layers)
        size = 16384
        while size <= 1024 * 1024:
            size_classes.append(size)
            size *= 2
            
        # Large tensors: 2MB to 64MB (common for large models)
        size = 2 * 1024 * 1024
        while size <= 64 * 1024 * 1024:
            size_classes.append(size)
            size *= 2
        
        return size_classes

    def _find_size_class(self, size: int) -> int:
        """Find appropriate size class for pooling"""
        for size_class in self.size_classes:
            if size_class >= size:
                return size_class
        # For very large allocations, use exact size
        return size

    def _should_pool(self, size: int, dtype: str) -> bool:
        """Decide whether to pool arrays of this size/type"""
        # Pool arrays between 64 bytes and 16MB
        # Don't pool very large arrays to avoid memory bloat
        return 64 <= size <= 16 * 1024 * 1024

    def allocate(self, shape: Tuple[int, ...], dtype: np.dtype) -> np.ndarray:
        """Allocate array from pool or create new"""
        size = np.prod(shape)
        itemsize = np.dtype(dtype).itemsize
        byte_size = size * itemsize
        
        with self.lock:
            dtype_str = str(dtype)
            
            if self._should_pool(byte_size, dtype_str):
                # Try to find existing array in pool
                size_class = self._find_size_class(size)
                key = (size_class, dtype_str)
                
                if key in self.free_blocks and self.free_blocks[key]:
                    # Reuse existing array
                    array = self.free_blocks[key].pop()
                    
                    # Reshape to required shape
                    if array.size >= size:
                        try:
                            array = array.flat[:size].reshape(shape)
                            self.allocated_blocks.add(id(array))
                            self.pool_hits += 1
                            return array
                        except (ValueError, AttributeError):
                            # Array was corrupted, fall through to create new
                            pass

            # Create new array
            array = np.empty(shape, dtype=dtype)
            self.allocated_blocks.add(id(array))
            self.total_allocated += array.nbytes
            self.peak_allocated = max(self.peak_allocated, self.total_allocated)
            self.pool_misses += 1
            
            return array

    def deallocate(self, array: np.ndarray):
        """Return array to pool"""
        with self.lock:
            array_id = id(array)
            if array_id not in self.allocated_blocks:
                return

            self.allocated_blocks.remove(array_id)
            
            size = array.size
            byte_size = array.nbytes
            dtype_str = str(array.dtype)
            
            if self._should_pool(byte_size, dtype_str):
                size_class = self._find_size_class(size)
                key = (size_class, dtype_str)
                
                if key not in self.free_blocks:
                    self.free_blocks[key] = []

                # Limit pool size per key to prevent unbounded growth
                if len(self.free_blocks[key]) < 10:  # Max 10 arrays per size class
                    # Flatten array for reuse flexibility
                    flat_array = array.flatten()
                    self.free_blocks[key].append(flat_array)

                # Clean up empty lists
                if len(self.free_blocks[key]) > 20:
                    self.free_blocks[key] = self.free_blocks[key][-10:]  # Keep newest 10

    def register_tensor_cleanup(self, tensor_id: str, tensor_obj):
        """Register tensor for automatic cleanup via weak reference"""
        def cleanup_callback(ref):
            self.tensor_cleanup_refs.pop(tensor_id, None)
        
        self.tensor_cleanup_refs[tensor_id] = weakref.ref(tensor_obj, cleanup_callback)

    def clear(self):
        """Clear all pooled memory"""
        with self.lock:
            self.free_blocks.clear()
            self.allocated_blocks.clear()
            self.total_allocated = 0

    def stats(self) -> Dict:
        """Get pool statistics - compatible with existing profiler"""
        with self.lock:
            pooled_memory = sum(
                sum(arr.nbytes for arr in arr_list) 
                for arr_list in self.free_blocks.values()
            )
            pooled_arrays = sum(len(arr_list) for arr_list in self.free_blocks.values())
            
            hit_rate = 0.0
            if self.pool_hits + self.pool_misses > 0:
                hit_rate = self.pool_hits / (self.pool_hits + self.pool_misses) * 100
            
            return {
                'total_allocated': self.total_allocated,
                'peak_allocated': self.peak_allocated,
                'active_blocks': len(self.allocated_blocks),
                'pooled_blocks': pooled_arrays,
                'pool_keys': len(self.free_blocks),
                'pooled_memory': pooled_memory,
                'pool_hit_rate': hit_rate,
                'pool_hits': self.pool_hits,
                'pool_misses': self.pool_misses,
                'registered_tensors': len(self.tensor_cleanup_refs)
            }


class DeviceManager:
    """Enhanced device manager - direct improvement of existing implementation"""

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
        """Get stats for all devices - compatible with existing profiler"""
        return {device: pool.stats() for device, pool in self.pools.items()}


# Update the global device manager instance
device_manager = DeviceManager()

def get_memory_pool_info() -> Dict:
    """
    Get memory pool information for profiler integration
    Returns data compatible with existing profiler structure
    """
    all_stats = device_manager.stats()
    
    if not all_stats:
        return {}
    
    total_allocated = sum(s['total_allocated'] for s in all_stats.values())
    total_pooled = sum(s['pooled_memory'] for s in all_stats.values())
    total_hits = sum(s['pool_hits'] for s in all_stats.values())
    total_requests = sum(s['pool_hits'] + s['pool_misses'] for s in all_stats.values())
    
    overall_hit_rate = 0.0
    if total_requests > 0:
        overall_hit_rate = total_hits / total_requests * 100
    
    return {
        'pool_allocated_mb': total_allocated / (1024 * 1024),
        'pool_memory_mb': total_pooled / (1024 * 1024), 
        'pool_hit_rate': overall_hit_rate,
        'pool_devices': list(all_stats.keys()),
        'pool_active_tensors': sum(s['active_blocks'] for s in all_stats.values()),
        'pool_efficiency': (total_pooled / max(1, total_allocated)) * 100 if total_allocated > 0 else 0
    }
