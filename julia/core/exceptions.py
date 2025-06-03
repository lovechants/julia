"""
Custom exceptions
"""

class JuliaError(Exception):
    """Base exception"""
    pass 

class TensorError(JuliaError):
    """Tensor Errors"""
    pass 

class ShapeError(TensorError):
    """Shape mismatch or invalid shape errors"""
    pass

class GradientError(TensorError):
    """Gradient computation errors"""
    pass

class BackendError(JuliaError):
    """Backend compilation errors"""
    pass

class LLVMError(BackendError):
    """LLVM backend specific errors"""
    pass

class CUDAError(BackendError):
    """CUDA backend specific errors"""
    pass

class ClangError(BackendError):
    """Clang backend specific errors"""
    pass

class ONNXError(JuliaError):
    """ONNX import/export errors"""
    pass

class OptimizerError(JuliaError):
    """Optimizer-related errors"""
    pass

class LayerError(JuliaError):
    """Neural network layer errors"""
    pass

def validate_tensor_shape(tensor, expected_shape, operation_name="operation"):
    """Validate tensor has expected shape"""
    if tensor.shape != expected_shape:
        raise ShapeError(
            f"{operation_name} expected shape {expected_shape}, "
            f"got {tensor.shape}"
        )

def validate_gradient_enabled(tensor, operation_name="operation"):
    """Validate tensor has gradients enabled"""
    if not tensor.requires_grad:
        raise GradientError(
            f"{operation_name} requires tensor with requires_grad=True"
        )

def validate_same_device(tensor1, tensor2, operation_name="operation"):
    """Validate tensors are on same device"""
    if tensor1.device != tensor2.device:
        raise TensorError(
            f"{operation_name} requires tensors on same device, "
            f"got {tensor1.device} and {tensor2.device}"
        )

def validate_backend_available(backend_name, available):
    """Validate backend is available"""
    if not available:
        raise BackendError(
            f"{backend_name} backend not available. "
            f"Please install required dependencies."
        )
