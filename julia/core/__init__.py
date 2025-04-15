from julia.core.tensor import Tensor, Function
from julia.core.ir import IRGraph, IRNode, DataType
from julia.core.ir_bridge import trace, execute_graph

# Import registries to ensure operations are registered
import julia.core.utils.ops_registry
try:
    import julia.core.utils.onnx_registry
except ImportError:
    pass  # ONNX not available
