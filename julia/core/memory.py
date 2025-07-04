import numpy as np
import weakref
from typing import Dict, Optional, Tuple, List
from threading import Lock, RLock
import ctypes
import atexit


class MemoryPool:
    """
    Thread-safe memory pool for tensor allocations with proper cleanup
    """

    _instance = None
    _lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized"):
            return
        self._pool_lock = RLock()  # Use RLock for nested locking
        self.free_blocks: Dict[int, List[ctypes.c_void_p]] = {}
        self.allocated_blocks: Dict[
            int, Tuple[ctypes.c_void_p, int]
        ] = {}  # ptr_value -> (ptr, size)
        self.max_pool_size = 100  # Increased for better reuse
        self.total_allocated = 0
        self.total_freed = 0
        self.peak_memory = 0

        atexit.register(self.cleanup_all)
        self._initialized = True

    def allocate(self, byte_size: int) -> Optional[ctypes.c_void_p]:
        """Thread-safe allocation of raw memory block"""
        if byte_size <= 0:
            return None

        with self._pool_lock:
            try:
                # Try to reuse from pool
                if byte_size in self.free_blocks and self.free_blocks[byte_size]:
                    ptr = self.free_blocks[byte_size].pop()
                    self.allocated_blocks[ptr.value] = (ptr, byte_size)
                    return ptr

                raw_memory = (ctypes.c_byte * byte_size)()
                ptr = ctypes.cast(raw_memory, ctypes.c_void_p)

                # Keep reference to prevent GC
                ptr._raw_memory = raw_memory

                self.allocated_blocks[ptr.value] = (ptr, byte_size)
                self.total_allocated += byte_size
                self.peak_memory = max(
                    self.peak_memory, self.total_allocated - self.total_freed
                )

                return ptr

            except Exception as e:
                print(f"Memory allocation failed: {e}")
                return None

    def deallocate(self, ptr: ctypes.c_void_p, byte_size: int):
        if not ptr or ptr.value == 0:
            return

        with self._pool_lock:
            try:
                if ptr.value not in self.allocated_blocks:
                    return  # Already freed or not allocated by us

                stored_ptr, stored_size = self.allocated_blocks[ptr.value]
                if stored_size != byte_size:
                    print(
                        f"Warning: Size mismatch on deallocation: {stored_size} vs {byte_size}"
                    )

                del self.allocated_blocks[ptr.value]
                self.total_freed += byte_size

                if byte_size not in self.free_blocks:
                    self.free_blocks[byte_size] = []

                if len(self.free_blocks[byte_size]) < self.max_pool_size:
                    self.free_blocks[byte_size].append(ptr)
                else:
                    del ptr._raw_memory

            except Exception as e:
                print(f"Memory deallocation failed: {e}")

    def cleanup_all(self):
        with self._pool_lock:
            try:
                for block_list in self.free_blocks.values():
                    for ptr in block_list:
                        if hasattr(ptr, "_raw_memory"):
                            del ptr._raw_memory
                self.free_blocks.clear()

                for ptr, size in self.allocated_blocks.values():
                    if hasattr(ptr, "_raw_memory"):
                        del ptr._raw_memory
                self.allocated_blocks.clear()

            except Exception as e:
                print(f"Memory cleanup failed: {e}")

    def get_stats(self):
        with self._pool_lock:
            total_free = sum(len(blocks) for blocks in self.free_blocks.values())
            current_allocated = len(self.allocated_blocks)

            return {
                "active_blocks": current_allocated,
                "free_blocks": total_free,
                "total_allocated_bytes": self.total_allocated,
                "total_freed_bytes": self.total_freed,
                "current_usage_bytes": self.total_allocated - self.total_freed,
                "peak_memory_bytes": self.peak_memory,
                "pool_sizes": {
                    size: len(blocks) for size, blocks in self.free_blocks.items()
                },
            }


_memory_pool = MemoryPool()


def try_allocate_raw_backed_array(source_array, device="cpu"):
    """
    Thread-safe creation of numpy array backed by raw memory
    Returns: (numpy_array, raw_ptr, byte_size) or (source_array, None, 0)
    """
    if device != "cpu":
        return source_array, None, 0

    if not isinstance(source_array, np.ndarray):
        source_array = np.asarray(source_array)

    try:
        byte_size = source_array.nbytes
        if byte_size == 0:
            return source_array, None, 0

        # Skip memory pool for very small arrays (scalars, small vectors)
        if byte_size < 64:  # Less than 64 bytes, just use regular copy
            return source_array.copy(), None, 0

        raw_ptr = _memory_pool.allocate(byte_size)

        if raw_ptr is not None:
            try:
                buffer = ctypes.string_at(raw_ptr, byte_size)
                backed_array = np.frombuffer(buffer, dtype=source_array.dtype).reshape(
                    source_array.shape
                )

                # Make it writable
                backed_array = backed_array.copy()

                # Handle scalar and array assignment differently
                if source_array.ndim == 0:
                    # Scalar case
                    backed_array.fill(source_array.item())
                else:
                    # Array case
                    backed_array[:] = source_array

                return backed_array, raw_ptr, byte_size
            except Exception as e:
                # If numpy array creation fails, deallocate and fallback
                _memory_pool.deallocate(raw_ptr, byte_size)
                # Don't print error for expected cases like scalars
                if "too many indices" not in str(e):
                    print(f"Failed to create backed array: {e}")

    except Exception as e:
        if "too many indices" not in str(e):
            print(f"Memory allocation error: {e}")

    return source_array.copy(), None, 0


def cleanup_raw_memory(raw_ptr, byte_size):
    try:
        if raw_ptr is not None and byte_size > 0:
            _memory_pool.deallocate(raw_ptr, byte_size)
    except Exception as e:
        print(f"Memory cleanup error: {e}")


def get_pool_stats():
    return _memory_pool.get_stats()


def cleanup_memory_pool():
    _memory_pool.cleanup_all()


# Weak reference cleanup for automatic memory management
class TensorMemoryManager:
    def __init__(self):
        self._cleanup_lock = Lock()
        self._cleanup_registry = weakref.WeakKeyDictionary()

    def register_tensor(self, tensor, raw_ptr, byte_size):
        if raw_ptr is None or byte_size <= 0:
            return

        with self._cleanup_lock:
            # Create cleanup callback
            def cleanup_callback():
                cleanup_raw_memory(raw_ptr, byte_size)

            # Register with weak reference
            try:
                weakref.finalize(tensor, cleanup_callback)
                self._cleanup_registry[tensor] = (raw_ptr, byte_size)
            except Exception as e:
                print(f"Failed to register tensor cleanup: {e}")
                cleanup_raw_memory(raw_ptr, byte_size)

    def manual_cleanup(self, tensor):
        with self._cleanup_lock:
            if tensor in self._cleanup_registry:
                raw_ptr, byte_size = self._cleanup_registry[tensor]
                cleanup_raw_memory(raw_ptr, byte_size)
                del self._cleanup_registry[tensor]


_tensor_memory_manager = TensorMemoryManager()


def register_tensor_memory(tensor, raw_ptr, byte_size):
    """Register tensor for automatic memory cleanup"""
    _tensor_memory_manager.register_tensor(tensor, raw_ptr, byte_size)


def manual_cleanup_tensor(tensor):
    """Manually cleanup tensor memory"""
    _tensor_memory_manager.manual_cleanup(tensor)
