import time
import threading
import json
import numpy as np
from dataclasses import dataclass, field
from collections import defaultdict, deque
from typing import Dict, List, Optional, Any, ContextManager, Callable, Union, Tuple
from contextlib import contextmanager
import functools
import weakref
import gc
import psutil
import os

@dataclass
class OperationStats:
    """Statistics for a single operation execution"""
    op_name: str
    op_type: str = "unknown"
    start_time: float = 0.0
    end_time: float = 0.0
    duration: float = 0.0
    memory_before: int = 0
    memory_after: int = 0
    memory_delta: int = 0
    input_shapes: List[tuple] = field(default_factory=list)
    output_shapes: List[tuple] = field(default_factory=list)
    input_dtypes: List[str] = field(default_factory=list)
    output_dtypes: List[str] = field(default_factory=list)
    flops: Optional[int] = None
    backward_time: Optional[float] = None
    gradient_shapes: List[tuple] = field(default_factory=list)
    device: str = "cpu"
    thread_id: int = 0
    call_stack_depth: int = 0
    custom_metrics: Dict[str, Any] = field(default_factory=dict)

@dataclass 
class AggregatedStats:
    """Aggregated statistics for multiple executions of the same operation"""
    op_name: str
    op_type: str
    call_count: int = 0
    total_time: float = 0.0
    avg_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    total_memory: int = 0
    avg_memory: float = 0.0
    total_flops: int = 0
    avg_flops: float = 0.0
    backward_time: float = 0.0
    
    def update(self, stats: OperationStats):
        """Update aggregated statistics with new operation stats"""
        self.call_count += 1
        self.total_time += stats.duration
        self.avg_time = self.total_time / self.call_count
        self.min_time = min(self.min_time, stats.duration)
        self.max_time = max(self.max_time, stats.duration)
        
        self.total_memory += stats.memory_delta
        self.avg_memory = self.total_memory / self.call_count
        
        if stats.flops:
            self.total_flops += stats.flops
            self.avg_flops = self.total_flops / self.call_count
            
        if stats.backward_time:
            self.backward_time += stats.backward_time

class ModelProfiler:
    """
    Comprehensive profiler for the Julia framework
    
    Features:
    - Operation timing and memory tracking
    - Hierarchical call stack profiling
    - Memory leak detection
    - FLOP counting for common operations
    - Hook-based integration with tensor operations
    - Export to various formats (JSON, CSV, etc.)
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern for global profiler instance"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, enabled: bool = True, memory_tracking: bool = True, 
                 flop_counting: bool = True, max_events: int = 10000):
        # Prevent re-initialization of singleton
        if hasattr(self, '_initialized'):
            return
        
        self.enabled = enabled
        self.memory_tracking = memory_tracking
        self.flop_counting = flop_counting
        self.max_events = max_events
        
        # Storage for profiling data
        self.operation_history: deque = deque(maxlen=max_events)
        self.aggregated_stats: Dict[str, AggregatedStats] = {}
        self.call_stack: List[OperationStats] = []
        self.active_timers: Dict[str, float] = {}
        
        # Memory tracking
        self.initial_memory = self._get_memory_usage()
        self.peak_memory = self.initial_memory
        self.memory_snapshots: List[tuple] = []  # (timestamp, memory)
        
        # Threading support
        self.thread_local = threading.local()
        self.lock = threading.RLock()
        
        # Hook registry for tensor operations
        self.tensor_hooks: Dict[int, Callable] = {}
        self.function_hooks: Dict[str, Callable] = {}
        
        # Performance counters
        self.total_operations = 0
        self.total_forward_time = 0.0
        self.total_backward_time = 0.0
        
        self._initialized = True
    
    def enable(self):
        """Enable profiling"""
        self.enabled = True
    
    def disable(self):
        """Disable profiling"""
        self.enabled = False
    
    def clear(self):
        """Clear all profiling data"""
        with self.lock:
            self.operation_history.clear()
            self.aggregated_stats.clear()
            self.call_stack.clear()
            self.active_timers.clear()
            self.memory_snapshots.clear()
            self.total_operations = 0
            self.total_forward_time = 0.0
            self.total_backward_time = 0.0
    
    def _get_memory_usage(self) -> int:
        """Get current memory usage in bytes"""
        if not self.memory_tracking:
            return 0
        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss
        except:
            return 0
    
    def _estimate_flops(self, op_type: str, input_shapes: List[tuple], 
                       output_shapes: List[tuple]) -> Optional[int]:
        """Estimate FLOPS for common operations"""
        if not self.flop_counting or not input_shapes:
            return None
            
        try:
            if op_type in ["Add", "Sub", "Mul", "Div"]:
                # Element-wise operations
                return max(np.prod(shape) for shape in input_shapes)
            
            elif op_type == "MatMul":
                if len(input_shapes) >= 2:
                    a_shape, b_shape = input_shapes[0], input_shapes[1]
                    if len(a_shape) == 2 and len(b_shape) == 2:
                        # Standard matrix multiplication: M*N*K
                        return a_shape[0] * a_shape[1] * b_shape[1] * 2  # 2 for mul+add
                    
            elif op_type in ["ReLU", "Sigmoid", "Tanh"]:
                # Activation functions
                return max(np.prod(shape) for shape in input_shapes)
            
            elif op_type.startswith("Conv"):
                if output_shapes and input_shapes:
                    # Rough estimate for convolution
                    output_size = np.prod(output_shapes[0])
                    # Assume 3x3 kernel as average
                    return output_size * 9
                    
        except Exception:
            pass
        
        return None
    
    @contextmanager
    def profile_operation(self, op_name: str, op_type: str = "unknown", 
                         inputs: List = None, device: str = "cpu",
                         custom_metrics: Dict[str, Any] = None):
        """
        Context manager for profiling an operation
        
        Usage:
            with profiler.profile_operation("linear_forward", "Linear", inputs=[x]):
                output = linear_layer(x)
        """
        if not self.enabled:
            yield
            return
            
        stats = OperationStats(
            op_name=op_name,
            op_type=op_type,
            device=device,
            thread_id=threading.get_ident(),
            call_stack_depth=len(self.call_stack),
            custom_metrics=custom_metrics or {}
        )
        
        # Extract input information
        if inputs:
            stats.input_shapes = [getattr(inp, 'shape', ()) for inp in inputs]
            stats.input_dtypes = [str(getattr(inp, 'dtype', 'unknown')) for inp in inputs]
        
        # Memory before operation
        if self.memory_tracking:
            stats.memory_before = self._get_memory_usage()
        
        # Start timing
        stats.start_time = time.perf_counter()
        
        # Add to call stack
        with self.lock:
            self.call_stack.append(stats)
        
        try:
            yield stats
        finally:
            # End timing
            stats.end_time = time.perf_counter()
            stats.duration = stats.end_time - stats.start_time
            
            # Memory after operation
            if self.memory_tracking:
                stats.memory_after = self._get_memory_usage()
                stats.memory_delta = stats.memory_after - stats.memory_before
                self.peak_memory = max(self.peak_memory, stats.memory_after)
                self.memory_snapshots.append((stats.end_time, stats.memory_after))
            
            # Remove from call stack
            with self.lock:
                if self.call_stack and self.call_stack[-1] is stats:
                    self.call_stack.pop()
                
                # Add to history and update aggregated stats
                self.operation_history.append(stats)
                self._update_aggregated_stats(stats)
                self.total_operations += 1
                self.total_forward_time += stats.duration
    
    def _update_aggregated_stats(self, stats: OperationStats):
        """Update aggregated statistics"""
        key = f"{stats.op_type}:{stats.op_name}"
        if key not in self.aggregated_stats:
            self.aggregated_stats[key] = AggregatedStats(
                op_name=stats.op_name,
                op_type=stats.op_type
            )
        self.aggregated_stats[key].update(stats)
    
    def profile_function(self, op_type: str = None, track_args: bool = True):
        """
        Decorator for profiling functions
        
        @profiler.profile_function("MatMul")
        def matmul_function(a, b):
            return a @ b
        """
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if not self.enabled:
                    return func(*args, **kwargs)
                
                func_name = f"{func.__module__}.{func.__name__}"
                inputs = list(args) if track_args else []
                
                with self.profile_operation(func_name, op_type or func.__name__, inputs):
                    return func(*args, **kwargs)
            return wrapper
        return decorator
    
    def register_tensor_hook(self, tensor, hook_name: str = None):
        """Register a hook on tensor operations"""
        if not hasattr(tensor, 'register_hook'):
            return
            
        hook_name = hook_name or f"tensor_{id(tensor)}"
        
        def backward_hook(grad):
            if self.enabled and self.call_stack:
                current_op = self.call_stack[-1]
                if current_op.backward_time is None:
                    current_op.backward_time = 0.0
                backward_start = time.perf_counter()
                
                # The actual backward computation happens after this hook
                def after_backward():
                    backward_end = time.perf_counter()
                    current_op.backward_time += backward_end - backward_start
                    self.total_backward_time += backward_end - backward_start
                
                # Schedule after_backward to run later
                threading.Timer(0.001, after_backward).start()
            
            return grad
        
        hook_handle = tensor.register_hook(backward_hook)
        self.tensor_hooks[id(tensor)] = hook_handle
        return hook_handle
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of profiling results"""
        with self.lock:
            total_time = sum(stats.duration for stats in self.operation_history)
            
            # Top operations by time
            top_ops_by_time = sorted(
                self.aggregated_stats.values(),
                key=lambda x: x.total_time,
                reverse=True
            )[:10]
            
            # Top operations by call count
            top_ops_by_count = sorted(
                self.aggregated_stats.values(),
                key=lambda x: x.call_count,
                reverse=True
            )[:10]
            
            from julia.core.memory import get_memory_pool_info
            pool_info = get_memory_pool_info()
            print(f"  Memory Pool:")
            print(f"    Hit Rate: {pool_info['pool_hit_rate']:.1f}%")
            print(f"    Efficiency: {pool_info['pool_efficiency']:.1f}%")

            # Memory analysis
            memory_info = {
                "initial_memory_mb": self.initial_memory / 1024 / 1024,
                "peak_memory_mb": self.peak_memory / 1024 / 1024,
                "memory_increase_mb": (self.peak_memory - self.initial_memory) / 1024 / 1024,
                "current_memory_mb": self._get_memory_usage() / 1024 / 1024,
            }
            
            return {
                "total_operations": self.total_operations,
                "total_time_seconds": total_time,
                "total_forward_time": self.total_forward_time,
                "total_backward_time": self.total_backward_time,
                "average_op_time": total_time / max(1, self.total_operations),
                "memory_info": memory_info,
                "top_operations_by_time": [
                    {
                        "name": op.op_name,
                        "type": op.op_type,
                        "total_time": op.total_time,
                        "avg_time": op.avg_time,
                        "call_count": op.call_count,
                        "percentage": (op.total_time / total_time * 100) if total_time > 0 else 0
                    }
                    for op in top_ops_by_time
                ],
                "top_operations_by_count": [
                    {
                        "name": op.op_name,
                        "type": op.op_type,
                        "call_count": op.call_count,
                        "total_time": op.total_time,
                        "avg_time": op.avg_time
                    }
                    for op in top_ops_by_count
                ]
            }
    
    def print_summary(self, top_n: int = 10):
        """Print a formatted summary of profiling results"""
        summary = self.get_summary()
        
        print("JULIA FRAMEWORK PROFILING SUMMARY")
        
        print(f"Total Operations: {summary['total_operations']}")
        print(f"Total Time: {summary['total_time_seconds']:.4f}s")
        print(f"Forward Time: {summary['total_forward_time']:.4f}s")
        print(f"Backward Time: {summary['total_backward_time']:.4f}s")
        print(f"Average Operation Time: {summary['average_op_time']:.6f}s")
        
        print("\nMemory Usage:")
        mem_info = summary['memory_info']
        print(f"  Initial: {mem_info['initial_memory_mb']:.2f} MB")
        print(f"  Peak: {mem_info['peak_memory_mb']:.2f} MB")
        print(f"  Current: {mem_info['current_memory_mb']:.2f} MB")
        print(f"  Increase: {mem_info['memory_increase_mb']:.2f} MB")
        
        print(f"\nTop {top_n} Operations by Time:")
        print("-" * 80)
        print(f"{'Operation':<30} {'Type':<15} {'Time(s)':<10} {'Count':<8} {'Avg(ms)':<10} {'%':<6}")
        print("-" * 80)
        
        for op in summary['top_operations_by_time'][:top_n]:
            print(f"{op['name'][:29]:<30} {op['type'][:14]:<15} "
                  f"{op['total_time']:<10.4f} {op['call_count']:<8} "
                  f"{op['avg_time']*1000:<10.3f} {op['percentage']:<6.1f}")
        
        print(f"\nTop {top_n} Operations by Call Count:")
        print("-" * 80)
        print(f"{'Operation':<30} {'Type':<15} {'Count':<8} {'Time(s)':<10} {'Avg(ms)':<10}")
        print("-" * 80)
        
        for op in summary['top_operations_by_count'][:top_n]:
            print(f"{op['name'][:29]:<30} {op['type'][:14]:<15} "
                  f"{op['call_count']:<8} {op['total_time']:<10.4f} "
                  f"{op['avg_time']*1000:<10.3f}")
    
    def export_to_json(self, filename: str):
        """Export profiling data to JSON file"""
        with self.lock:
            data = {
                "summary": self.get_summary(),
                "operation_history": [
                    {
                        "op_name": stats.op_name,
                        "op_type": stats.op_type,
                        "duration": stats.duration,
                        "memory_delta": stats.memory_delta,
                        "input_shapes": stats.input_shapes,
                        "output_shapes": stats.output_shapes,
                        "flops": stats.flops,
                        "thread_id": stats.thread_id,
                        "call_stack_depth": stats.call_stack_depth,
                        "custom_metrics": stats.custom_metrics
                    }
                    for stats in self.operation_history
                ],
                "memory_timeline": self.memory_snapshots
            }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    def export_to_csv(self, filename: str):
        """Export operation history to CSV file"""
        import csv
        
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'operation', 'type', 'duration_ms', 'memory_delta_mb',
                'input_shapes', 'flops', 'thread_id', 'call_stack_depth'
            ])
            
            with self.lock:
                for stats in self.operation_history:
                    writer.writerow([
                        stats.op_name,
                        stats.op_type,
                        stats.duration * 1000,  # Convert to ms
                        stats.memory_delta / 1024 / 1024,  # Convert to MB
                        str(stats.input_shapes),
                        stats.flops or 0,
                        stats.thread_id,
                        stats.call_stack_depth
                    ])
    
    def get_memory_timeline(self) -> List[Tuple[float, float]]:
        """Get memory usage timeline"""
        with self.lock:
            return [(ts, mem / 1024 / 1024) for ts, mem in self.memory_snapshots]
    
    def detect_memory_leaks(self, threshold_mb: float = 100.0) -> List[str]:
        """Detect potential memory leaks"""
        leaks = []
        
        if self.memory_snapshots:
            initial = self.memory_snapshots[0][1] if self.memory_snapshots else self.initial_memory
            current = self._get_memory_usage()
            increase_mb = (current - initial) / 1024 / 1024
            
            if increase_mb > threshold_mb:
                leaks.append(f"Total memory increase: {increase_mb:.2f} MB")
        
        # Check for operations that consistently increase memory
        with self.lock:
            for key, stats in self.aggregated_stats.items():
                if stats.avg_memory > 10 * 1024 * 1024:  # > 10MB average increase
                    leaks.append(f"Operation {key} averages {stats.avg_memory/1024/1024:.2f} MB increase")
        
        return leaks

# Global profiler instance
profiler = ModelProfiler()

# Convenience functions
def enable_profiling():
    """Enable global profiling"""
    profiler.enable()

def disable_profiling():
    """Disable global profiling"""
    profiler.disable()

def clear_profiling_data():
    """Clear all profiling data"""
    profiler.clear()

def get_profiling_summary():
    """Get profiling summary"""
    return profiler.get_summary()

def print_profiling_summary(top_n: int = 10):
    """Print profiling summary"""
    profiler.print_summary(top_n)

@contextmanager
def profile_scope(name: str, op_type: str = "scope"):
    """Context manager for profiling a code scope"""
    with profiler.profile_operation(name, op_type):
        yield
