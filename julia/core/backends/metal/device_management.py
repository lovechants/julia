import os 
import sys 
import platform
import ctypes 
import numpy as np 
from typing import Dict, List, Tuple, Union, Any, Optional

try:
    if platform.system() == "Darwin":
        import objc
        from Foundation import NSBundle
        from Metal import (
            MTLCreateSystemDefaultDevice,
            MTLCommandQueue,
            MTLBuffer,
            MTLComputePipelineState,
            MTLLibrary,
            MTLFunction,
            MTLCommandBuffer,
            MTLComputeCommandEncoder,
            MTLResourceStorageModeShared,
            MTLResourceStorageModePrivate
        )
        METAL_AVAILABLE = True
    else:
        METAL_AVAILABLE = False
except ImportError:
    METAL_AVAILABLE = False

class MetalBuffer:
    pass

class MetalDeviceManager:
    pass 
