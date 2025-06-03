import os
import tempfile
import subprocess
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from julia.core.ir import IRGraph, IRNode, DataType
from julia.core.backends.metal.device_management import MetalDeviceManager, METAL_AVAILABLE

if METAL_AVAILABLE:
    from Metal import (
        MTLLibrary,
        MTLFunction,
        MTLComputePipelineState,
        MTLSize,
        MTLCreateSystemDefaultDevice
    )

class MetalKernalRunner:
    pass 

class MetalCompiler:
    pass
