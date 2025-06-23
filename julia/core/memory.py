import numpy as np 
import weakref
from typing import Dict, Optional, Set, Tuple, List 
from threading import Lock
import ctypes

"""
Memory pools and buffer management 
"""

class MemoryPool:
    """
    Improved memory pool for tensor allocations
    """
    def __init__(self):
        self.free_blocks: Dict[int, List[ctypes.c_void_p]] = {}
        self.allocated_blocks = set()
        self.max_pool_size = 20  # Max blocks per size
    
    def allocate(self, byte_size: int) -> Optional[ctypes.c_void_p]:
        """Allocate raw memory block"""
        try:
            # Try to reuse from pool
            if byte_size in self.free_blocks and self.free_blocks[byte_size]:
                ptr = self.free_blocks[byte_size].pop()
                self.allocated_blocks.add(ptr.value)
                return ptr
            
            # Allocate new block
            raw_memory = (ctypes.c_byte * byte_size)()
            ptr = ctypes.cast(raw_memory, ctypes.c_void_p)
            self.allocated_blocks.add(ptr.value)
            
            # Keep reference to prevent GC
            ptr._raw_memory = raw_memory
            return ptr
            
        except:
            return None
    
    def deallocate(self, ptr: ctypes.c_void_p, byte_size: int):
        """Return raw memory to pool"""
        try:
            if ptr and ptr.value in self.allocated_blocks:
                self.allocated_blocks.discard(ptr.value)
                
                # Return to pool (limit pool size)
                if byte_size not in self.free_blocks:
                    self.free_blocks[byte_size] = []
                
                if len(self.free_blocks[byte_size]) < self.max_pool_size:
                    self.free_blocks[byte_size].append(ptr)
        except:
            pass

# Global pool instance
_memory_pool = MemoryPool()

def try_allocate_raw_backed_array(source_array, device="cpu"):
    """
    Try to create a numpy array backed by raw memory
    Returns: (numpy_array, raw_ptr, byte_size) or (source_array, None, 0)
    """
    if device != "cpu":
        return source_array, None, 0
    
    try:
        import numpy as np
        
        byte_size = source_array.nbytes
        raw_ptr = _memory_pool.allocate(byte_size)
        
        if raw_ptr is not None:
            # Create numpy array from raw memory
            backed_array = np.frombuffer(
                ctypes.string_at(raw_ptr, byte_size), 
                dtype=source_array.dtype
            ).reshape(source_array.shape)
            
            # Copy source data
            backed_array[:] = source_array
            
            return backed_array, raw_ptr, byte_size
        
    except:
        pass
    
    # Fallback to original array
    return source_array, None, 0

def cleanup_raw_memory(raw_ptr, byte_size):
    """Clean up raw memory allocation"""
    try:
        if raw_ptr is not None:
            _memory_pool.deallocate(raw_ptr, byte_size)
    except:
        pass

def get_pool_stats():
    """Get memory pool statistics"""
    try:
        total_free = sum(len(blocks) for blocks in _memory_pool.free_blocks.values())
        return {
            'active_blocks': len(_memory_pool.allocated_blocks),
            'free_blocks': total_free,
            'pool_sizes': {size: len(blocks) for size, blocks in _memory_pool.free_blocks.items()}
        }
    except:
        return {'error': 'Could not get stats'}
