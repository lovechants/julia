import os 
import numpy as np 
from typing import Dict, List, Tuple, Union, Any, Optional

try:
    import pycuda.driver as cuda 
    import pycuda.autoinit 
    from pycuda.compiler import SourceModule 
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    print("Cuda not available")



