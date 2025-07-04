"""
Pytest configuration and fixtures for Julia framework tests.
"""

import pytest
import numpy as np
from julia.core.tensor import Tensor


# Configure pytest markers
def pytest_configure(config):
    """Register custom pytest markers."""
    config.addinivalue_line("markers", "autograd: tests for autograd engine and operations")
    
    # CPU Compiler backends
    config.addinivalue_line("markers", "compiler_cpu: all CPU compiler backends")
    config.addinivalue_line("markers", "compiler_cpu_llvm: LLVM CPU backend tests")
    config.addinivalue_line("markers", "compiler_cpu_clang: Clang CPU backend tests")
    
    # GPU Compiler backends
    config.addinivalue_line("markers", "compiler_gpu: all GPU compiler backends")
    config.addinivalue_line("markers", "compiler_gpu_cuda: CUDA backend tests")
    config.addinivalue_line("markers", "compiler_gpu_triton: Triton backend tests")
    config.addinivalue_line("markers", "compiler_gpu_opencl: OpenCL backend tests")
    config.addinivalue_line("markers", "compiler_gpu_rocm: ROCm backend tests")
    config.addinivalue_line("markers", "compiler_gpu_metal: Apple Metal backend tests")
    
    # General categories
    config.addinivalue_line("markers", "neural_network: tests for neural network layers")
    config.addinivalue_line("markers", "numerical: tests for numerical accuracy")
    config.addinivalue_line("markers", "memory: tests for memory management")
    config.addinivalue_line("markers", "memory_profiling: tests requiring psutil for memory profiling")
    config.addinivalue_line("markers", "integration: integration tests with examples")
    
    # Serialization
    config.addinivalue_line("markers", "serialization: all serialization tests")
    config.addinivalue_line("markers", "serialization_onnx: ONNX export/import tests")
    config.addinivalue_line("markers", "serialization_ir: IR graph tests")
    
    # Performance and misc
    config.addinivalue_line("markers", "benchmark: performance benchmark tests")
    config.addinivalue_line("markers", "slow: tests that take a long time to run")
    config.addinivalue_line("markers", "profiling: profiler functionality tests")
    config.addinivalue_line("markers", "data: data loading and processing tests")


# Common fixtures
@pytest.fixture
def simple_tensor():
    """Simple 2D tensor for testing."""
    return Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)


@pytest.fixture
def random_tensor():
    """Random tensor for testing."""
    np.random.seed(42)
    data = np.random.randn(3, 4)
    return Tensor(data, requires_grad=True)


@pytest.fixture
def batch_tensors():
    """Batch of tensors for testing."""
    np.random.seed(42)
    return [
        Tensor(np.random.randn(2, 3), requires_grad=True),
        Tensor(np.random.randn(2, 3), requires_grad=True),
        Tensor(np.random.randn(2, 3), requires_grad=True),
    ]


@pytest.fixture
def tolerance():
    """Default numerical tolerance for testing."""
    return 1e-6


# Utility functions for tests
def assert_tensors_close(actual, expected, rtol=1e-7, atol=1e-8):
    """Assert that two tensors are close in value."""
    if isinstance(actual, Tensor):
        actual = actual.data
    if isinstance(expected, Tensor):
        expected = expected.data
    
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)


def finite_difference_gradient(func, x, h=1e-5):
    """Compute gradient using finite differences for testing."""
    grad = np.zeros_like(x.data)
    
    for i in range(x.data.size):
        flat_idx = np.unravel_index(i, x.shape)
        
        # Forward difference
        x_plus = x.data.copy()
        x_plus[flat_idx] += h
        x_tensor_plus = Tensor(x_plus, requires_grad=True)
        f_plus = func(x_tensor_plus)
        
        # Backward difference  
        x_minus = x.data.copy()
        x_minus[flat_idx] -= h
        x_tensor_minus = Tensor(x_minus, requires_grad=True)
        f_minus = func(x_tensor_minus)
        
        # Central difference
        grad.flat[i] = (f_plus.data - f_minus.data) / (2 * h)
    
    return grad


# Skip markers for missing dependencies
def pytest_runtest_setup(item):
    """Skip tests if required dependencies are missing."""
    # CPU Compiler backends
    if "compiler_cpu_llvm" in item.keywords:
        try:
            import llvmlite
        except ImportError:
            pytest.skip("llvmlite not available for LLVM backend tests")
    
    if "compiler_cpu_clang" in item.keywords:
        try:
            import subprocess
            subprocess.run(['clang', '--version'], check=True, capture_output=True)
        except (ImportError, subprocess.CalledProcessError, FileNotFoundError):
            pytest.skip("clang not available for Clang backend tests")
    
    # GPU Compiler backends
    if "compiler_gpu_cuda" in item.keywords:
        try:
            import pycuda
            import pycuda.driver as cuda
            cuda.init()
        except ImportError:
            pytest.skip("pycuda not available for CUDA backend tests")
        except Exception:
            pytest.skip("CUDA not available or no GPU detected")
    
    if "compiler_gpu_triton" in item.keywords:
        try:
            import triton
        except ImportError:
            pytest.skip("triton not available for Triton backend tests")
    
    if "compiler_gpu_opencl" in item.keywords:
        try:
            import pyopencl
        except ImportError:
            pytest.skip("pyopencl not available for OpenCL backend tests")
    
    if "compiler_gpu_rocm" in item.keywords:
        try:
            import hip
        except ImportError:
            pytest.skip("hip-python not available for ROCm backend tests")
    
    if "compiler_gpu_metal" in item.keywords:
        try:
            import Metal
        except ImportError:
            pytest.skip("Metal framework not available (macOS only)")
    
    # Serialization
    if "serialization_onnx" in item.keywords:
        try:
            import onnx
        except ImportError:
            pytest.skip("onnx not available for ONNX serialization tests")
    
    # Memory profiling
    if "memory_profiling" in item.keywords:
        try:
            import psutil
        except ImportError:
            pytest.skip("psutil not available for memory profiling tests")
