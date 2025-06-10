"""
Custom exceptions
"""
import traceback
import sys 
import numpy as np 
import threading 
import time 
import warnings
from enum import Enum
from dataclasses import dataclass, field 
from contextlib import contextmanager 
from typing import Dict, List, Optional, Any, Union, Tuple, Callable


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ErrorContext:
    """Context information for errors"""
    operation: Optional[str] = None
    tensor_shapes: List[Tuple] = field(default_factory=list)
    tensor_dtypes: List[str] = field(default_factory=list)
    tensor_devices: List[str] = field(default_factory=list)
    tensor_requires_grad: List[bool] = field(default_factory=list)
    model_state: Optional[str] = None
    layer_name: Optional[str] = None
    batch_size: Optional[int] = None
    memory_usage: Optional[int] = None
    thread_id: Optional[int] = None
    stack_depth: Optional[int] = None
    custom_context: Dict[str, Any] = field(default_factory=dict)

class JuliaError(Exception):
    """
    Base exception class for Julia framework
    
    Provides enhanced error reporting with context, suggestions, and debugging info
    """
    
    def __init__(self, message: str, 
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 context: Optional[ErrorContext] = None,
                 suggestion: Optional[str] = None,
                 error_code: Optional[str] = None,
                 cause: Optional[Exception] = None,
                 recoverable: bool = False):
        self.message = message
        self.severity = severity
        self.context = context or ErrorContext()
        self.suggestion = suggestion
        self.error_code = error_code
        self.cause = cause
        self.recoverable = recoverable
        self.timestamp = time.time()
        self.thread_id = threading.get_ident()
        
        # Capture stack trace
        self.stack_trace = traceback.format_stack()[:-1]
        
        super().__init__(self._format_message())
    
    def _format_message(self) -> str:
        """Format comprehensive error message"""
        lines = [f"Julia {self.severity.value.upper()} Error: {self.message}"]
        
        if self.error_code:
            lines.append(f"Error Code: {self.error_code}")
        
        if self.context:
            lines.append("\nContext Information:")
            if self.context.operation:
                lines.append(f"  Operation: {self.context.operation}")
            if self.context.layer_name:
                lines.append(f"  Layer: {self.context.layer_name}")
            if self.context.tensor_shapes:
                lines.append(f"  Tensor Shapes: {self.context.tensor_shapes}")
            if self.context.tensor_dtypes:
                lines.append(f"  Tensor Dtypes: {self.context.tensor_dtypes}")
            if self.context.tensor_devices:
                lines.append(f"  Tensor Devices: {self.context.tensor_devices}")
            if self.context.batch_size:
                lines.append(f"  Batch Size: {self.context.batch_size}")
            if self.context.memory_usage:
                lines.append(f"  Memory Usage: {self.context.memory_usage / 1024 / 1024:.2f} MB")
        
        if self.suggestion:
            lines.append(f"\nSuggestion: {self.suggestion}")
        
        if self.cause:
            lines.append(f"\nCaused by: {type(self.cause).__name__}: {self.cause}")
        
        if self.recoverable:
            lines.append("\nThis error is potentially recoverable.")
        
        return "\n".join(lines)
    
    def add_context(self, **kwargs):
        """Add additional context to the error"""
        for key, value in kwargs.items():
            if hasattr(self.context, key):
                setattr(self.context, key, value)
            else:
                self.context.custom_context[key] = value
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for logging/serialization"""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "severity": self.severity.value,
            "error_code": self.error_code,
            "timestamp": self.timestamp,
            "thread_id": self.thread_id,
            "recoverable": self.recoverable,
            "context": {
                "operation": self.context.operation,
                "tensor_shapes": self.context.tensor_shapes,
                "tensor_dtypes": self.context.tensor_dtypes,
                "layer_name": self.context.layer_name,
                "batch_size": self.context.batch_size,
                "memory_usage": self.context.memory_usage,
                "custom_context": self.context.custom_context
            },
            "suggestion": self.suggestion,
            "cause": str(self.cause) if self.cause else None
        }

# Tensor-related exceptions
class TensorError(JuliaError):
    """Base class for tensor-related errors"""
    
    def __init__(self, message: str, tensors: List = None, operation: str = None, **kwargs):
        # Extract tensor information
        context = kwargs.get('context', ErrorContext())
        if operation:
            context.operation = operation
        if tensors:
            context.tensor_shapes = [getattr(t, 'shape', None) for t in tensors]
            context.tensor_dtypes = [str(getattr(t, 'dtype', None)) for t in tensors]
            context.tensor_devices = [getattr(t, 'device', 'unknown') for t in tensors]
            context.tensor_requires_grad = [getattr(t, 'requires_grad', False) for t in tensors]
        
        kwargs['context'] = context
        super().__init__(message, **kwargs)

class ShapeError(TensorError):
    """Shape mismatch or invalid shape errors with detailed analysis"""
    
    def __init__(self, message: str, expected_shape=None, actual_shape=None, 
                 operation: str = None, **kwargs):
        self.expected_shape = expected_shape
        self.actual_shape = actual_shape
        
        # Enhanced message with shape analysis
        if expected_shape and actual_shape:
            shape_msg = f"Expected shape {expected_shape}, got {actual_shape}"
            if message:
                message = f"{message}. {shape_msg}"
            else:
                message = shape_msg
                
            # Add broadcasting suggestion
            suggestion = self._generate_shape_suggestion(expected_shape, actual_shape)
            kwargs.setdefault('suggestion', suggestion)
        
        kwargs.setdefault('error_code', 'SHAPE_MISMATCH')
        super().__init__(message, operation=operation, **kwargs)
    
    def _generate_shape_suggestion(self, expected, actual) -> str:
        """Generate helpful suggestions for shape mismatches"""
        if not expected or not actual:
            return "Check tensor dimensions and reshape if necessary"
        
        suggestions = []
        
        # Check if it's a dimension mismatch
        if len(expected) != len(actual):
            if len(actual) < len(expected):
                suggestions.append(f"Consider using unsqueeze() to add dimensions")
            else:
                suggestions.append(f"Consider using squeeze() or view() to reduce dimensions")
        
        # Check if it's broadcastable
        try:
            import numpy as np
            np.broadcast_shapes(expected, actual)
            suggestions.append("Shapes are broadcastable - this might be an internal error")
        except ValueError:
            # Check for simple fixes
            if len(expected) == len(actual):
                for i, (e, a) in enumerate(zip(expected, actual)):
                    if e != a and e == 1:
                        suggestions.append(f"Try expanding dimension {i} of expected tensor")
                    elif e != a and a == 1:
                        suggestions.append(f"Try expanding dimension {i} of actual tensor")
        
        if not suggestions:
            suggestions.append("Use .reshape() or .view() to adjust tensor shape")
        
        return "; ".join(suggestions)

class DTypeError(TensorError):
    """Data type related errors"""
    
    def __init__(self, message: str, expected_dtype=None, actual_dtype=None, operation: str = None, **kwargs):
        self.expected_dtype = expected_dtype
        self.actual_dtype = actual_dtype
        
        if expected_dtype and actual_dtype:
            dtype_msg = f"Expected dtype {expected_dtype}, got {actual_dtype}"
            if message:
                message = f"{message}. {dtype_msg}"
            else:
                message = dtype_msg
                
            # Add casting suggestion
            kwargs.setdefault('suggestion', 
                f"Use .astype({expected_dtype}) or ensure input tensors have consistent dtypes")
        
        kwargs.setdefault('error_code', 'DTYPE_MISMATCH')
        super().__init__(message, operation=operation, **kwargs)

class DeviceError(TensorError):
    """Device placement and movement errors"""
    
    def __init__(self, message: str, expected_device=None, actual_device=None, operation: str = None, **kwargs):
        self.expected_device = expected_device
        self.actual_device = actual_device
        
        if expected_device and actual_device:
            device_msg = f"Expected device {expected_device}, got {actual_device}"
            if message:
                message = f"{message}. {device_msg}"
            else:
                message = device_msg
                
            kwargs.setdefault('suggestion', 
                f"Use .to('{expected_device}') to move tensor to correct device")
        
        kwargs.setdefault('error_code', 'DEVICE_MISMATCH')
        super().__init__(message, operation=operation, **kwargs)

class GradientError(TensorError):
    """Gradient computation errors"""
    
    def __init__(self, message: str, operation: str = None, **kwargs):
        kwargs.setdefault('error_code', 'GRADIENT_ERROR')
        kwargs.setdefault('suggestion', 
            "Check that tensors require gradients and the computation graph is valid")
        super().__init__(message, operation=operation, **kwargs)

class AutogradError(GradientError):
    """Automatic differentiation errors"""
    
    def __init__(self, message: str, operation: str = None, **kwargs):
        if operation:
            message = f"Autograd error in {operation}: {message}"
        
        suggestions = [
            "Ensure all inputs requiring gradients are leaf tensors or have valid gradient functions",
            "Check for in-place operations that break the computation graph",
            "Verify that retain_graph=True if you need to backward through the graph multiple times"
        ]
        
        kwargs.setdefault('suggestion', "; ".join(suggestions))
        kwargs.setdefault('error_code', 'AUTOGRAD_ERROR')
        super().__init__(message, **kwargs)

# Backend-related exceptions
class BackendError(JuliaError):
    """Base class for backend compilation errors"""
    
    def __init__(self, message: str, backend: str = None, **kwargs):
        self.backend = backend
        if backend:
            message = f"{backend} backend error: {message}"
        
        kwargs.setdefault('error_code', f'{backend.upper()}_ERROR' if backend else 'BACKEND_ERROR')
        super().__init__(message, **kwargs)

class LLVMError(BackendError):
    """LLVM backend specific errors"""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('suggestion', 
            "Check LLVM installation and ensure llvmlite is properly installed")
        super().__init__(message, backend="LLVM", **kwargs)

class CUDAError(BackendError):
    """CUDA backend specific errors"""
    
    def __init__(self, message: str, **kwargs):
        suggestions = [
            "Check CUDA installation and driver version",
            "Verify GPU memory availability",
            "Ensure tensors are moved to GPU with .to('cuda')"
        ]
        kwargs.setdefault('suggestion', "; ".join(suggestions))
        super().__init__(message, backend="CUDA", **kwargs)

class ClangError(BackendError):
    """Clang backend specific errors"""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('suggestion', 
            "Check Clang installation and ensure clang bindings are available")
        super().__init__(message, backend="Clang", **kwargs)

class MetalError(BackendError):
    """Metal backend specific errors"""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('suggestion', 
            "Metal backend only available on macOS with compatible hardware")
        super().__init__(message, backend="Metal", **kwargs)

# Model and layer exceptions
class LayerError(JuliaError):
    """Neural network layer errors"""
    
    def __init__(self, message: str, layer_name: str = None, layer_type: str = None, **kwargs):
        self.layer_name = layer_name
        self.layer_type = layer_type
        
        context = kwargs.get('context', ErrorContext())
        context.layer_name = layer_name
        kwargs['context'] = context
        
        if layer_name and layer_type:
            message = f"Error in {layer_type} layer '{layer_name}': {message}"
        elif layer_name:
            message = f"Error in layer '{layer_name}': {message}"
        elif layer_type:
            message = f"Error in {layer_type} layer: {message}"
        
        kwargs.setdefault('error_code', 'LAYER_ERROR')
        super().__init__(message, **kwargs)

class ModelError(JuliaError):
    """Model-level errors"""
    
    def __init__(self, message: str, model_name: str = None, **kwargs):
        self.model_name = model_name
        
        if model_name:
            message = f"Model '{model_name}' error: {message}"
        
        kwargs.setdefault('error_code', 'MODEL_ERROR')
        super().__init__(message, **kwargs)

class OptimizerError(JuliaError):
    """Optimizer-related errors"""
    
    def __init__(self, message: str, optimizer_type: str = None, **kwargs):
        self.optimizer_type = optimizer_type
        
        if optimizer_type:
            message = f"{optimizer_type} optimizer error: {message}"
        
        suggestions = [
            "Check learning rate and other hyperparameters",
            "Ensure model parameters are properly passed to optimizer",
            "Verify gradient computation is working correctly"
        ]
        
        kwargs.setdefault('suggestion', "; ".join(suggestions))
        kwargs.setdefault('error_code', 'OPTIMIZER_ERROR')
        super().__init__(message, **kwargs)

# Data and I/O exceptions
class DataError(JuliaError):
    """Data loading and processing errors"""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('error_code', 'DATA_ERROR')
        super().__init__(message, **kwargs)

class ONNXError(JuliaError):
    """ONNX import/export errors"""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('suggestion', 
            "Check ONNX model format and ensure onnx package is installed")
        kwargs.setdefault('error_code', 'ONNX_ERROR')
        super().__init__(message, **kwargs)

class CheckpointError(JuliaError):
    """Model checkpoint save/load errors"""
    
    def __init__(self, message: str, **kwargs):
        suggestions = [
            "Verify file path and permissions",
            "Check available disk space",
            "Ensure model state dict is serializable"
        ]
        kwargs.setdefault('suggestion', "; ".join(suggestions))
        kwargs.setdefault('error_code', 'CHECKPOINT_ERROR')
        super().__init__(message, **kwargs)

# Memory and performance exceptions
class MemoryError(JuliaError):
    """Memory-related errors"""
    
    def __init__(self, message: str, **kwargs):
        suggestions = [
            "Reduce batch size or model size",
            "Use gradient checkpointing to save memory",
            "Clear unused tensors and call garbage collection",
            "Consider using mixed precision training"
        ]
        kwargs.setdefault('suggestion', "; ".join(suggestions))
        kwargs.setdefault('error_code', 'MEMORY_ERROR')
        kwargs.setdefault('severity', ErrorSeverity.HIGH)
        super().__init__(message, **kwargs)

class PerformanceWarning(UserWarning):
    """Performance-related warnings"""
    pass

# Validation and utility functions
def validate_tensor_shape(tensor, expected_shape, operation_name="operation"):
    """Validate tensor has expected shape with enhanced error reporting"""
    if not hasattr(tensor, 'shape'):
        raise TensorError(f"Object is not a tensor", operation=operation_name)
    
    if tensor.shape != expected_shape:
        raise ShapeError(
            f"{operation_name} shape validation failed",
            expected_shape=expected_shape,
            actual_shape=tensor.shape,
            operation=operation_name,
            tensors=[tensor]
        )

def validate_tensor_dtype(tensor, expected_dtype, operation_name="operation"):
    """Validate tensor has expected dtype"""
    if not hasattr(tensor, 'dtype'):
        raise TensorError(f"Object is not a tensor", operation=operation_name)
    
    actual_dtype = getattr(tensor, 'dtype', None)
    if actual_dtype != expected_dtype:
        raise DTypeError(
            f"{operation_name} dtype validation failed",
            expected_dtype=expected_dtype,
            actual_dtype=actual_dtype,
            operation=operation_name,
            tensors=[tensor]
        )

def validate_tensor_device(tensor, expected_device, operation_name="operation"):
    """Validate tensor is on expected device"""
    if not hasattr(tensor, 'device'):
        raise TensorError(f"Object is not a tensor", operation=operation_name)
    
    actual_device = getattr(tensor, 'device', 'unknown')
    if actual_device != expected_device:
        raise DeviceError(
            f"{operation_name} device validation failed",
            expected_device=expected_device,
            actual_device=actual_device,
            operation=operation_name,
            tensors=[tensor]
        )

def validate_gradient_enabled(tensor, operation_name="operation"):
    """Validate tensor has gradients enabled"""
    if not hasattr(tensor, 'requires_grad'):
        raise TensorError(f"Object is not a tensor", operation=operation_name)
    
    if not tensor.requires_grad:
        raise GradientError(
            f"{operation_name} requires tensor with requires_grad=True",
            operation=operation_name,
            tensors=[tensor]
        )

def validate_same_device(*tensors, operation_name="operation"):
    """Validate tensors are on same device"""
    if len(tensors) < 2:
        return
    
    devices = [getattr(t, 'device', 'unknown') for t in tensors]
    if len(set(devices)) > 1:
        raise DeviceError(
            f"{operation_name} requires tensors on same device",
            operation=operation_name,
            tensors=list(tensors),
            suggestion=f"Found devices: {devices}. Use .to() to move tensors to same device"
        )

def validate_same_dtype(*tensors, operation_name="operation"):
    """Validate tensors have same dtype"""
    if len(tensors) < 2:
        return
    
    dtypes = [getattr(t, 'dtype', None) for t in tensors]
    if len(set(str(dt) for dt in dtypes)) > 1:
        raise DTypeError(
            f"{operation_name} requires tensors with same dtype",
            operation=operation_name,
            tensors=list(tensors),
            suggestion=f"Found dtypes: {dtypes}. Use .astype() to convert tensors"
        )

def validate_backend_available(backend_name, available):
    """Validate backend is available with installation instructions"""
    if not available:
        suggestions = {
            "LLVM": "pip install llvmlite",
            "CUDA": "Install CUDA toolkit and pycuda: pip install pycuda",
            "Clang": "Install clang and python bindings",
            "Metal": "Metal backend only available on macOS",
            "ONNX": "pip install onnx"
        }
        
        suggestion = suggestions.get(backend_name, f"Install {backend_name} dependencies")
        
        raise BackendError(
            f"{backend_name} backend not available",
            backend=backend_name,
            suggestion=suggestion,
            severity=ErrorSeverity.HIGH
        )

# Context managers for enhanced error handling
@contextmanager
def error_context(operation: str = None, **context_kwargs):
    """Context manager to add operation context to errors"""
    try:
        yield
    except JuliaError as e:
        # Add context to existing Julia errors
        if operation and not e.context.operation:
            e.context.operation = operation
        e.add_context(**context_kwargs)
        raise
    except Exception as e:
        # Wrap other exceptions in JuliaError
        context = ErrorContext(operation=operation, **context_kwargs)
        raise JuliaError(
            f"Unexpected error in {operation or 'operation'}: {str(e)}",
            context=context,
            cause=e,
            severity=ErrorSeverity.HIGH
        ) from e

@contextmanager
def tensor_operation_context(operation: str, *tensors, **kwargs):
    """Context manager specifically for tensor operations"""
    context_data = {
        'tensor_shapes': [getattr(t, 'shape', None) for t in tensors],
        'tensor_dtypes': [str(getattr(t, 'dtype', None)) for t in tensors],
        'tensor_devices': [getattr(t, 'device', 'unknown') for t in tensors],
    }
    context_data.update(kwargs)
    
    try:
        yield
    except JuliaError as e:
        # Add context to existing Julia errors
        if operation and not e.context.operation:
            e.context.operation = operation
        # Add tensor context to custom_context since these aren't direct attributes
        e.context.custom_context.update(context_data)
        raise
    except Exception as e:
        # Wrap other exceptions in JuliaError
        context = ErrorContext(operation=operation, custom_context=context_data)
        raise JuliaError(
            f"Unexpected error in {operation or 'operation'}: {str(e)}",
            context=context,
            cause=e,
            severity=ErrorSeverity.HIGH
        ) from e

# Warning system
def warn_performance(message: str, category=PerformanceWarning, stacklevel=2):
    """Issue performance warning with context"""
    warnings.warn(message, category=category, stacklevel=stacklevel)

def warn_deprecated(message: str, version: str = None, stacklevel=2):
    """Issue deprecation warning"""
    if version:
        message = f"{message} (deprecated since version {version})"
    warnings.warn(message, category=DeprecationWarning, stacklevel=stacklevel)

# Error recovery utilities
class ErrorRecovery:
    """Utilities for error recovery and graceful degradation"""
    
    @staticmethod
    def safe_tensor_operation(operation: Callable, *args, fallback=None, **kwargs):
        """Safely execute tensor operation with fallback"""
        try:
            return operation(*args, **kwargs)
        except JuliaError as e:
            if e.recoverable and fallback is not None:
                warn_performance(f"Using fallback for {operation.__name__}: {str(e)}")
                return fallback(*args, **kwargs)
            raise
    
    @staticmethod
    def auto_fix_shape_mismatch(tensor, target_shape):
        """Attempt to automatically fix common shape mismatches"""
        if not hasattr(tensor, 'shape'):
            return tensor
        
        current_shape = tensor.shape
        
        # If shapes match, no fix needed
        if current_shape == target_shape:
            return tensor
        
        # Try to squeeze extra dimensions
        if len(current_shape) > len(target_shape):
            squeezed = tensor
            for _ in range(len(current_shape) - len(target_shape)):
                if hasattr(squeezed, 'squeeze'):
                    squeezed = squeezed.squeeze()
            if squeezed.shape == target_shape:
                return squeezed
        
        # Try to reshape if total elements match
        if hasattr(tensor, 'reshape'):
            try:
                current_size = np.prod(current_shape) if current_shape else 1
                target_size = np.prod(target_shape) if target_shape else 1
                if current_size == target_size:
                    return tensor.reshape(target_shape)
            except:
                pass
        
        return tensor

# Global error handling configuration
class ErrorConfig:
    """Global configuration for error handling"""
    
    def __init__(self):
        self.show_suggestions = True
        self.show_context = True
        self.auto_recovery = False
        self.performance_warnings = True
        self.max_tensor_info = 5  # Max number of tensors to show info for
        
    def configure(self, **kwargs):
        """Configure error handling behavior"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

# Global instance
error_config = ErrorConfig()

def configure_error_handling(**kwargs):
    """Configure global error handling behavior"""
    error_config.configure(**kwargs)
