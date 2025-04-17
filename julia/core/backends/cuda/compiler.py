import os 
import numpy as np 
from typing import Dict, List, Tuple, Any, Optional
import pkg_resources

# TODO test in lab | linux or windows 
try: 
    import pycuda.driver as cuda 
    import pycuda.autoinit
    import pycuda.compiler 
    from pycuda.compiler import SourceModule
    CUDA_AVAILABLE = True
expect ImportError:
    CUDA_AVAILABLE = False
    print("Warning: CUDA not available")

from julia.core.ir import IRGraph, IRNode, DataType

class PTXKernelRunner:
    """
    Runner for pre-compiled kernels (add, sub, etc.)
    Simple operations we can easily speed up outside of the cuda wrapper 
    """

    def __init__(self, ptx_file_path: str, kernel_name: str):
        """
        args: 
        file_path -> path to ptx file 
        kernel_name -> name of th kernel 
        """

        if not CUDA_AVAILABLE:
            raise ImportError("CUDA required for PTX kernel")

        with open(ptx_file_path, 'r') as f:
            ptx_code = f.read()

        self.module = cuda.module_from_buffer(ptx_code.encode())
        self.kernel = self.module.get_function(kernel_name)

    def __call__(self, *args, grid=(1,1), block=(256, 1, 1)) -> Any:
        self.kernel(*args, grid=grid, block=block)

# TODO do this in the lab
"""
Example 
check the operation 
see if there is a ptx kernel associated 
build the kernel 
if not kernel -> cuda 

"""
class CudaCompiler: 
    pass
