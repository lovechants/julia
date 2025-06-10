import pytest
import numpy as np
import threading
import time
import warnings
from unittest.mock import Mock, patch, MagicMock

import sys
import os
from julia.core.tensor import *
from julia.core.exceptions import *

class MockTensor:
    def __init__(self, shape, dtype='float32', device='cpu', requires_grad=False):
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self.requires_grad = requires_grad
        self.data = np.zeros(shape, dtype=dtype)
    
    def reshape(self, new_shape):
        return MockTensor(new_shape, self.dtype, self.device, self.requires_grad)
    
    def squeeze(self):
        squeezed_shape = tuple(d for d in self.shape if d != 1)
        return MockTensor(squeezed_shape, self.dtype, self.device, self.requires_grad)
    
    def to(self, device):
        return MockTensor(self.shape, self.dtype, device, self.requires_grad)
    
    def astype(self, dtype):
        return MockTensor(self.shape, dtype, self.device, self.requires_grad)

class TestJuliaErrorBase:
    """Test the base JuliaError class functionality"""
    
    def test_basic_error_creation(self):
        """Test basic error creation with message"""
        error = JuliaError("Test error message")
        assert "Test error message" in str(error)
        assert error.severity == ErrorSeverity.MEDIUM
        assert error.recoverable == False
        assert error.error_code is None
    
    def test_error_with_all_parameters(self):
        """Test error creation with all parameters"""
        context = ErrorContext(
            operation="test_op",
            tensor_shapes=[(3, 4), (2, 2)],
            layer_name="test_layer"
        )
        
        error = JuliaError(
            "Complex error",
            severity=ErrorSeverity.HIGH,
            context=context,
            suggestion="Try this fix",
            error_code="TEST_001",
            recoverable=True
        )
        
        assert error.severity == ErrorSeverity.HIGH
        assert "test_op" in str(error)
        assert "Try this fix" in str(error)
        assert error.error_code == "TEST_001"
        assert error.recoverable == True
        assert error.context.layer_name == "test_layer"
    
    def test_error_context_addition(self):
        """Test adding context to errors after creation"""
        error = JuliaError("Test error")
        error.add_context(batch_size=32, memory_usage=1024*1024)
        
        assert error.context.batch_size == 32
        assert error.context.memory_usage == 1024*1024
    
    def test_error_serialization(self):
        """Test error serialization to dict"""
        context = ErrorContext(operation="test_op", tensor_shapes=[(2, 2)])
        error = JuliaError("Test", context=context, error_code="TEST_001")
        
        error_dict = error.to_dict()
        assert error_dict["error_type"] == "JuliaError"
        assert error_dict["message"] == "Test"
        assert error_dict["error_code"] == "TEST_001"
        assert error_dict["context"]["operation"] == "test_op"
    
    def test_error_threading_info(self):
        """Test that thread ID is captured correctly"""
        def create_error():
            return JuliaError("Thread test")
        
        error1 = create_error()
        error2 = threading.Thread(target=create_error).start()
        
        assert error1.thread_id == threading.get_ident()
        # Thread ID should be captured at creation time


class TestTensorErrors:
    """Test tensor-specific error types"""
    
    def test_tensor_error_with_tensors(self):
        """Test TensorError with tensor information extraction"""
        tensor1 = MockTensor((3, 4), 'float32', 'cpu', True)
        tensor2 = MockTensor((2, 2), 'int32', 'cuda', False)
        
        error = TensorError("Tensor operation failed", tensors=[tensor1, tensor2])
        
        assert error.context.tensor_shapes == [(3, 4), (2, 2)]
        assert error.context.tensor_dtypes == ['float32', 'int32']
        assert error.context.tensor_devices == ['cpu', 'cuda']
        assert error.context.tensor_requires_grad == [True, False]
    
    def test_shape_error_with_suggestions(self):
        """Test ShapeError generates helpful suggestions"""
        error = ShapeError(
            "Matrix multiplication failed",
            expected_shape=(3, 4),
            actual_shape=(2, 4),
            operation="matmul"
        )
        
        assert error.expected_shape == (3, 4)
        assert error.actual_shape == (2, 4)
        assert "reshape" in error.suggestion.lower() or "view" in error.suggestion.lower()
        assert "SHAPE_MISMATCH" in str(error)
    
    def test_shape_error_broadcast_suggestion(self):
        """Test ShapeError detects broadcastable shapes"""
        # Create shapes that are NOT broadcastable but have dimension size differences
        error = ShapeError(
            "Addition failed", 
            expected_shape=(3, 1, 4),
            actual_shape=(3, 2, 4)
        )
        
        suggestion = error._generate_shape_suggestion((3, 1, 4), (3, 2, 4))
        # Should suggest expanding the dimension with size 1
        assert "expand" in suggestion.lower() or "expanding" in suggestion.lower()
    
    def test_dtype_error(self):
        """Test DTypeError functionality"""
        error = DTypeError(
            "Type mismatch",
            expected_dtype='float32',
            actual_dtype='int64'
        )
        
        assert error.expected_dtype == 'float32'
        assert error.actual_dtype == 'int64'
        assert "astype" in error.suggestion
        assert "DTYPE_MISMATCH" in str(error)
    
    def test_device_error(self):
        """Test DeviceError functionality"""
        error = DeviceError(
            "Device mismatch",
            expected_device='cuda',
            actual_device='cpu'
        )
        
        assert error.expected_device == 'cuda'
        assert error.actual_device == 'cpu'
        assert ".to('cuda')" in error.suggestion
        assert "DEVICE_MISMATCH" in str(error)
    
    def test_gradient_error(self):
        """Test GradientError functionality"""
        error = GradientError("Gradient computation failed")
        
        assert "GRADIENT_ERROR" in str(error)
        assert "require" in error.suggestion.lower()
    
    def test_autograd_error(self):
        """Test AutogradError with operation context"""
        error = AutogradError("Backward pass failed", operation="matmul")
        
        assert "matmul" in str(error)
        assert "AUTOGRAD_ERROR" in str(error)
        assert "retain_graph" in error.suggestion


class TestBackendErrors:
    """Test backend-specific error types"""
    
    def test_llvm_error(self):
        """Test LLVMError"""
        error = LLVMError("Compilation failed")
        
        assert "LLVM backend error" in str(error)
        assert "llvmlite" in error.suggestion
        assert "LLVM_ERROR" in str(error)
    
    def test_cuda_error(self):
        """Test CUDAError"""
        error = CUDAError("GPU memory exhausted")
        
        assert "CUDA backend error" in str(error)
        assert "CUDA installation" in error.suggestion
        assert "CUDA_ERROR" in str(error)
    
    def test_clang_error(self):
        """Test ClangError"""
        error = ClangError("Clang compilation failed")
        
        assert "Clang backend error" in str(error)
        assert "Clang installation" in error.suggestion
    
    def test_metal_error(self):
        """Test MetalError"""
        error = MetalError("Metal not available")
        
        assert "Metal backend error" in str(error)
        assert "macOS" in error.suggestion


class TestModelErrors:
    """Test model and layer error types"""
    
    def test_layer_error(self):
        """Test LayerError with layer context"""
        error = LayerError(
            "Forward pass failed",
            layer_name="conv1",
            layer_type="Conv2D"
        )
        
        assert error.layer_name == "conv1"
        assert error.layer_type == "Conv2D"
        assert "Conv2D layer 'conv1'" in str(error)
        assert error.context.layer_name == "conv1"
    
    def test_model_error(self):
        """Test ModelError"""
        error = ModelError("Model loading failed", model_name="ResNet50")
        
        assert error.model_name == "ResNet50"
        assert "Model 'ResNet50' error" in str(error)
    
    def test_optimizer_error(self):
        """Test OptimizerError"""
        error = OptimizerError("Invalid learning rate", optimizer_type="Adam")
        
        assert error.optimizer_type == "Adam"
        assert "Adam optimizer error" in str(error)
        assert "learning rate" in error.suggestion


class TestIOErrors:
    """Test I/O and data-related errors"""
    
    def test_onnx_error(self):
        """Test ONNXError"""
        error = ONNXError("Invalid ONNX model")
        
        assert "ONNX_ERROR" in str(error)
        assert "onnx package" in error.suggestion
    
    def test_checkpoint_error(self):
        """Test CheckpointError"""
        error = CheckpointError("Failed to save checkpoint")
        
        assert "CHECKPOINT_ERROR" in str(error)
        assert "file path" in error.suggestion
    
    def test_memory_error(self):
        """Test MemoryError with high severity"""
        error = MemoryError("Out of memory")
        
        assert error.severity == ErrorSeverity.HIGH
        assert "batch size" in error.suggestion
        assert "MEMORY_ERROR" in str(error)


class TestValidationFunctions:
    """Test validation utility functions"""
    
    def test_validate_tensor_shape_success(self):
        """Test successful shape validation"""
        tensor = MockTensor((3, 4))
        # Should not raise any exception
        validate_tensor_shape(tensor, (3, 4), "test_op")
    
    def test_validate_tensor_shape_failure(self):
        """Test shape validation failure"""
        tensor = MockTensor((3, 4))
        
        with pytest.raises(ShapeError) as exc_info:
            validate_tensor_shape(tensor, (2, 4), "test_op")
        
        error = exc_info.value
        assert error.expected_shape == (2, 4)
        assert error.actual_shape == (3, 4)
        assert "test_op" in str(error)
    
    def test_validate_tensor_dtype_success(self):
        """Test successful dtype validation"""
        tensor = MockTensor((3, 4), dtype='float32')
        validate_tensor_dtype(tensor, 'float32', "test_op")
    
    def test_validate_tensor_dtype_failure(self):
        """Test dtype validation failure"""
        tensor = MockTensor((3, 4), dtype='int32')
        
        with pytest.raises(DTypeError) as exc_info:
            validate_tensor_dtype(tensor, 'float32', "test_op")
        
        error = exc_info.value
        assert error.expected_dtype == 'float32'
        assert error.actual_dtype == 'int32'
    
    def test_validate_tensor_device_success(self):
        """Test successful device validation"""
        tensor = MockTensor((3, 4), device='cuda')
        validate_tensor_device(tensor, 'cuda', "test_op")
    
    def test_validate_tensor_device_failure(self):
        """Test device validation failure"""
        tensor = MockTensor((3, 4), device='cpu')
        
        with pytest.raises(DeviceError) as exc_info:
            validate_tensor_device(tensor, 'cuda', "test_op")
        
        error = exc_info.value
        assert error.expected_device == 'cuda'
        assert error.actual_device == 'cpu'
    
    def test_validate_gradient_enabled_success(self):
        """Test successful gradient validation"""
        tensor = MockTensor((3, 4), requires_grad=True)
        validate_gradient_enabled(tensor, "test_op")
    
    def test_validate_gradient_enabled_failure(self):
        """Test gradient validation failure"""
        tensor = MockTensor((3, 4), requires_grad=False)
        
        with pytest.raises(GradientError) as exc_info:
            validate_gradient_enabled(tensor, "test_op")
    
    def test_validate_same_device_success(self):
        """Test successful same device validation"""
        tensor1 = MockTensor((3, 4), device='cuda')
        tensor2 = MockTensor((2, 2), device='cuda')
        validate_same_device(tensor1, tensor2, operation_name="test_op")
    
    def test_validate_same_device_failure(self):
        """Test same device validation failure"""
        tensor1 = MockTensor((3, 4), device='cpu')
        tensor2 = MockTensor((2, 2), device='cuda')
        
        with pytest.raises(DeviceError) as exc_info:
            validate_same_device(tensor1, tensor2, operation_name="test_op")
        
        error = exc_info.value
        assert "same device" in str(error)
        assert "cpu" in error.suggestion and "cuda" in error.suggestion
    
    def test_validate_same_dtype_success(self):
        """Test successful same dtype validation"""
        tensor1 = MockTensor((3, 4), dtype='float32')
        tensor2 = MockTensor((2, 2), dtype='float32')
        validate_same_dtype(tensor1, tensor2, operation_name="test_op")
    
    def test_validate_same_dtype_failure(self):
        """Test same dtype validation failure"""
        tensor1 = MockTensor((3, 4), dtype='float32')
        tensor2 = MockTensor((2, 2), dtype='int64')
        
        with pytest.raises(DTypeError) as exc_info:
            validate_same_dtype(tensor1, tensor2, operation_name="test_op")
    
    def test_validate_backend_available_success(self):
        """Test successful backend validation"""
        validate_backend_available("LLVM", True)
    
    def test_validate_backend_available_failure(self):
        """Test backend validation failure"""
        with pytest.raises(BackendError) as exc_info:
            validate_backend_available("CUDA", False)
        
        error = exc_info.value
        assert "CUDA backend not available" in str(error)
        assert "pip install pycuda" in error.suggestion


class TestContextManagers:
    """Test error context managers"""
    
    def test_error_context_success(self):
        """Test error_context when no error occurs"""
        with error_context(operation="test_op", batch_size=32):
            # No error should occur
            pass
    
    def test_error_context_julia_error(self):
        """Test error_context with existing JuliaError"""
        original_error = TensorError("Original error")
        
        with pytest.raises(TensorError) as exc_info:
            with error_context(operation="test_op", batch_size=32):
                raise original_error
        
        # Should be the same error but with added context
        caught_error = exc_info.value
        assert caught_error is original_error
        assert caught_error.context.operation == "test_op"
        assert caught_error.context.custom_context["batch_size"] == 32
    
    def test_error_context_other_exception(self):
        """Test error_context wrapping non-Julia exceptions"""
        with pytest.raises(JuliaError) as exc_info:
            with error_context(operation="test_op"):
                raise ValueError("Standard Python error")
        
        error = exc_info.value
        assert isinstance(error.cause, ValueError)
        assert "Unexpected error in test_op" in str(error)
        assert error.severity == ErrorSeverity.HIGH
    
    def test_tensor_operation_context(self):
        """Test tensor_operation_context"""
        tensor1 = MockTensor((3, 4), 'float32', 'cpu')
        tensor2 = MockTensor((2, 2), 'int32', 'cuda')
        
        with pytest.raises(JuliaError) as exc_info:
            with tensor_operation_context("matmul", tensor1, tensor2):
                raise RuntimeError("Operation failed")
        
        error = exc_info.value
        assert error.context.operation == "matmul"
        assert (3, 4) in error.context.custom_context["tensor_shapes"]
        assert (2, 2) in error.context.custom_context["tensor_shapes"]
        assert 'float32' in error.context.custom_context["tensor_dtypes"]


class TestWarningSystem:
    """Test warning functionality"""
    
    def test_performance_warning(self):
        """Test performance warning emission"""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warn_performance("This operation is slow")
            
            assert len(w) == 1
            assert issubclass(w[0].category, PerformanceWarning)
            assert "slow" in str(w[0].message)
    
    def test_deprecation_warning(self):
        """Test deprecation warning with version"""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warn_deprecated("This function is deprecated", version="1.2.0")
            
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "1.2.0" in str(w[0].message)


class TestErrorRecovery:
    """Test error recovery utilities"""
    
    def test_safe_tensor_operation_success(self):
        """Test safe operation when no error occurs"""
        def successful_op(x, y):
            return x + y
        
        result = ErrorRecovery.safe_tensor_operation(successful_op, 1, 2)
        assert result == 3
    
    def test_safe_tensor_operation_recoverable_error(self):
        """Test safe operation with recoverable error and fallback"""
        def failing_op(x, y):
            raise JuliaError("Operation failed", recoverable=True)
        
        def fallback_op(x, y):
            return x * y
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = ErrorRecovery.safe_tensor_operation(
                failing_op, 2, 3, fallback=fallback_op
            )
            
            assert result == 6  # fallback was used
            assert len(w) == 1
            assert "fallback" in str(w[0].message).lower()
    
    def test_safe_tensor_operation_non_recoverable_error(self):
        """Test safe operation with non-recoverable error"""
        def failing_op(x, y):
            raise JuliaError("Critical error", recoverable=False)
        
        def fallback_op(x, y):
            return x * y
        
        with pytest.raises(JuliaError):
            ErrorRecovery.safe_tensor_operation(
                failing_op, 2, 3, fallback=fallback_op
            )
    
    def test_auto_fix_shape_mismatch_squeeze(self):
        """Test automatic shape fixing by squeezing"""
        tensor = MockTensor((3, 1, 4))
        fixed = ErrorRecovery.auto_fix_shape_mismatch(tensor, (3, 4))
        
        # Should squeeze the dimension of size 1
        assert fixed.shape == (3, 4)
    
    def test_auto_fix_shape_mismatch_reshape(self):
        """Test automatic shape fixing by reshaping"""
        tensor = MockTensor((2, 6))
        fixed = ErrorRecovery.auto_fix_shape_mismatch(tensor, (3, 4))
        
        # Should reshape since total elements match (12)
        assert fixed.shape == (3, 4)
    
    def test_auto_fix_shape_mismatch_no_fix(self):
        """Test when no automatic fix is possible"""
        tensor = MockTensor((2, 3))
        fixed = ErrorRecovery.auto_fix_shape_mismatch(tensor, (3, 4))
        
        # Should return original tensor when no fix possible
        assert fixed.shape == (2, 3)


class TestErrorConfiguration:
    """Test global error configuration"""
    
    def test_error_config_defaults(self):
        """Test default error configuration"""
        config = ErrorConfig()
        assert config.show_suggestions == True
        assert config.show_context == True
        assert config.auto_recovery == False
        assert config.performance_warnings == True
        assert config.max_tensor_info == 5
    
    def test_error_config_modification(self):
        """Test modifying error configuration"""
        config = ErrorConfig()
        config.configure(
            show_suggestions=False,
            auto_recovery=True,
            max_tensor_info=10
        )
        
        assert config.show_suggestions == False
        assert config.auto_recovery == True
        assert config.max_tensor_info == 10
    
    def test_global_config_function(self):
        """Test global configuration function"""
        configure_error_handling(
            show_context=False,
            performance_warnings=False
        )
        
        assert error_config.show_context == False
        assert error_config.performance_warnings == False


class TestIntegrationScenarios:
    """Test realistic error scenarios"""
    
    def test_complex_tensor_operation_error(self):
        """Test complex scenario with multiple validation failures"""
        tensor1 = MockTensor((3, 4), 'float32', 'cpu', True)
        tensor2 = MockTensor((2, 4), 'int32', 'cuda', False)
        
        # This should catch multiple validation errors
        with pytest.raises(DTypeError):
            validate_same_dtype(tensor1, tensor2, operation_name="complex_op")
    
    def test_nested_context_managers(self):
        """Test nested error context managers"""
        tensor = MockTensor((3, 4))
        
        with pytest.raises(JuliaError) as exc_info:
            with error_context(operation="outer_op", model_name="test_model"):
                with tensor_operation_context("inner_op", tensor):
                    raise ValueError("Deep error")
        
        error = exc_info.value
        error_str = str(error)
        # Should capture context from both managers
        assert "inner_op" in error_str  # inner context overwrites outer operation
        assert "(3, 4)" in error_str  # tensor shapes should be in the error string
    
    def test_memory_error_with_context(self):
        """Test memory error with full context"""
        context = ErrorContext(
            operation="large_matmul",
            tensor_shapes=[(1000, 1000), (1000, 1000)],
            batch_size=128,
            memory_usage=8 * 1024 * 1024 * 1024  # 8GB
        )
        
        error = MemoryError(
            "GPU out of memory during matrix multiplication",
            context=context
        )
        
        error_str = str(error)
        assert "large_matmul" in error_str
        assert "8192.00 MB" in error_str  # Memory formatted in MB
        assert "reduce batch size" in error.suggestion.lower()
    
    def test_autograd_error_chain(self):
        """Test chained autograd errors"""
        original_error = ValueError("Gradient computation failed")
        
        autograd_error = AutogradError(
            "Backward pass failed in layer conv1",
            operation="backward",
            cause=original_error
        )
        
        assert isinstance(autograd_error.cause, ValueError)
        assert "conv1" in str(autograd_error)
        assert "retain_graph" in autograd_error.suggestion
