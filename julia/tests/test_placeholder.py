import pytest


@pytest.mark.compiler_cpu_llvm
def test_placeholder_compiler_cpu_llvm():
    pass


@pytest.mark.compiler_cpu_clang
def test_placeholder_compiler_cpu_clang():
    pass


@pytest.mark.compiler_gpu_cuda
def test_placeholder_compiler_gpu_cuda():
    pass


@pytest.mark.compiler_gpu_triton
def test_placeholder_compiler_gpu_triton():
    pass


@pytest.mark.compiler_gpu_opencl
def test_placeholder_compiler_gpu_opencl():
    pass


@pytest.mark.compiler_gpu_rocm
def test_placeholder_compiler_gpu_rocm():
    pass


@pytest.mark.compiler_gpu_metal
def test_placeholder_compiler_gpu_metal():
    pass


@pytest.mark.serialization_onnx
def test_placeholder_serialization_onnx():
    pass


@pytest.mark.memory_profiling
def test_placeholder_memory_profiling():
    pass


@pytest.mark.neural_network
def test_placeholder_neural_network():
    pass


@pytest.mark.autograd
def test_placeholder_autograd():
    pass


@pytest.mark.numerical
def test_placeholder_numerical():
    pass
