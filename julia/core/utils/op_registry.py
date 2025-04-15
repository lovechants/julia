"""
Registration system so operations can register themselves 
For IR -> ONNX seperate  
"""
from typing import Dict, Callable, Type, Any, Optional, List, Tuple
import inspect

class OpRegistry:
    """
    Registry for operations that maps between Tensor ops, IR nodes, and shape inference
    """
    _instance = None  # Singleton instance
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        # Only initialize if this is the singleton instance
        if self.__class__._instance is not None:
            return
            
        # Maps from function class to op_type string
        self.fn_to_op_type: Dict[Type, str] = {}
        
        # Maps from op_type string to function class
        self.op_type_to_fn: Dict[str, Type] = {}
        
        # Maps from op_type to shape inference function
        self.shape_infer_funcs: Dict[str, Callable] = {}
        
    def register(self, op_type: str, fn_cls: Type = None):
        """
        Register an operation type
        
        Can be used as a decorator:
        @registry.register("Add")
        class Add(Function):
            ...
            
        Or directly:
        registry.register("Add", Add)
        """
        def decorator(fn_cls):
            self.fn_to_op_type[fn_cls] = op_type
            self.op_type_to_fn[op_type] = fn_cls
            return fn_cls
            
        if fn_cls is not None:
            return decorator(fn_cls)
        return decorator
        
    def register_shape_inference(self, op_type: str, func: Callable = None):
        """
        Register a shape inference function for an operation type
        
        Can be used as a decorator:
        @registry.register_shape_inference("Add")
        def infer_add_shape(node, input_shapes):
            ...
            
        Or directly:
        registry.register_shape_inference("Add", infer_add_shape)
        """
        def decorator(func):
            self.shape_infer_funcs[op_type] = func
            return func
            
        if func is not None:
            return decorator(func)
        return decorator
        
    def get_op_type(self, fn_cls: Type) -> str:
        """Get the operation type string for a function class"""
        return self.fn_to_op_type.get(fn_cls, fn_cls.__name__)
        
    def get_function_class(self, op_type: str) -> Optional[Type]:
        """Get the function class for an operation type"""
        return self.op_type_to_fn.get(op_type)
        
    def get_shape_inference(self, op_type: str) -> Optional[Callable]:
        """Get the shape inference function for an operation type"""
        return self.shape_infer_funcs.get(op_type)
        
    def infer_shape(self, node, input_shapes: List[Tuple[int, ...]]) -> Optional[Tuple[int, ...]]:
        """
        Infer the output shape for a node given input shapes
        
        Args:
            node: The IR node
            input_shapes: List of shapes of the input nodes
            
        Returns:
            The inferred output shape, or None if shape inference fails
        """
        infer_func = self.get_shape_inference(node.op_type)
        if infer_func:
            return infer_func(node, input_shapes)
        return None

# Create singleton instance
registry = OpRegistry.get_instance()


# Register standard shape inference functions

@registry.register_shape_inference("Add")
def infer_add_shape(node, input_shapes):
    """Shape inference for Add: inputs must have compatible shapes"""
    if len(input_shapes) != 2:
        return None
        
    # Handle broadcasting
    if len(input_shapes[0]) == 0 or len(input_shapes[1]) == 0:
        # One input is a scalar
        return input_shapes[0] if len(input_shapes[0]) > 0 else input_shapes[1]
        
    if input_shapes[0] == input_shapes[1]:
        # Same shape
        return input_shapes[0]
        
    
    try:
        
        import numpy as np
        a = np.empty(input_shapes[0])
        b = np.empty(input_shapes[1])
        return np.broadcast(a, b).shape
    except Exception:
        return None


@registry.register_shape_inference("Sub")
def infer_sub_shape(node, input_shapes):
    """Shape inference for Sub: same as Add"""
    return infer_add_shape(node, input_shapes)


@registry.register_shape_inference("Mul")
def infer_mul_shape(node, input_shapes):
    """Shape inference for Mul: same as Add"""
    return infer_add_shape(node, input_shapes)


@registry.register_shape_inference("Div")
def infer_div_shape(node, input_shapes):
    """Shape inference for Div: same as Add"""
    return infer_add_shape(node, input_shapes)


@registry.register_shape_inference("MatMul")
def infer_matmul_shape(node, input_shapes):
    """Shape inference for MatMul"""
    if len(input_shapes) != 2:
        return None
        
    a_shape = input_shapes[0]
    b_shape = input_shapes[1]
    
    if a_shape is None or b_shape is None:
        return None
    
    # Vector-vector: (n) x (n) -> scalar
    if len(a_shape) == 1 and len(b_shape) == 1:
        if a_shape[0] != b_shape[0]:
            return None
        return ()
        
    # Matrix-vector: (m, n) x (n) -> (m)
    if len(a_shape) == 2 and len(b_shape) == 1:
        if a_shape[1] != b_shape[0]:
            return None
        return (a_shape[0],)
        
    # Vector-matrix: (m) x (m, n) -> (n)
    if len(a_shape) == 1 and len(b_shape) == 2:
        if a_shape[0] != b_shape[0]:
            return None
        return (b_shape[1],)
        
    # Matrix-matrix: (m, n) x (n, p) -> (m, p)
    if len(a_shape) == 2 and len(b_shape) == 2:
        if a_shape[1] != b_shape[0]:
            return None
        return (a_shape[0], b_shape[1])
        
    # Batched matrix multiplication with matrix: (b..., m, n) x (n, p) -> (b..., m, p)
    if len(a_shape) >= 3 and len(b_shape) == 2:
        if a_shape[-1] != b_shape[0]:
            return None
        return a_shape[:-1] + (b_shape[1],)
        
    # Matrix with batched matrix: (m, n) x (b..., n, p) -> (b..., m, p)
    if len(a_shape) == 2 and len(b_shape) >= 3:
        if a_shape[1] != b_shape[-2]:
            return None
        return b_shape[:-2] + (a_shape[0], b_shape[-1])
        
    # Batched matrix multiplication: (b..., m, n) x (b..., n, p) -> (b..., m, p)
    if len(a_shape) >= 3 and len(b_shape) >= 3:
        # Check matrix dimensions
        if a_shape[-1] != b_shape[-2]:
            return None
            
        # If batch dimensions match exactly, use them
        if a_shape[:-2] == b_shape[:-2]:
            return a_shape[:-2] + (a_shape[-2], b_shape[-1])
            
        # Try to broadcast batch dimensions
        try:
            import numpy as np
            a_batch = np.empty(a_shape[:-2])
            b_batch = np.empty(b_shape[:-2])
            batch_shape = np.broadcast(a_batch, b_batch).shape
            return batch_shape + (a_shape[-2], b_shape[-1])
        except Exception:
            # Just combine the batch dimensions if broadcasting fails
            return a_shape[:-2] + b_shape[:-2] + (a_shape[-2], b_shape[-1])
    
    return None


@registry.register_shape_inference("ReLU")
def infer_relu_shape(node, input_shapes):
    """Shape inference for ReLU: same as input"""
    if len(input_shapes) != 1:
        return None
    return input_shapes[0]


@registry.register_shape_inference("Sigmoid")
def infer_sigmoid_shape(node, input_shapes):
    """Shape inference for Sigmoid: same as input"""
    if len(input_shapes) != 1:
        return None
    return input_shapes[0]


@registry.register_shape_inference("Tanh")
def infer_tanh_shape(node, input_shapes):
    """Shape inference for Tanh: same as input"""
    if len(input_shapes) != 1:
        return None
    return input_shapes[0]


@registry.register_shape_inference("Reshape")
def infer_reshape_shape(node, input_shapes):
    """Shape inference for Reshape"""
    if len(input_shapes) < 1:
        return None
        
    # Get the new shape from attributes or second input
    new_shape = None
    
    if len(input_shapes) > 1:
        # Shape comes from second input
        if hasattr(node.inputs[1], 'attributes') and 'value' in node.inputs[1].attributes:
            new_shape = tuple(int(x) for x in node.inputs[1].attributes['value'].flatten())
        
    if new_shape is None and 'shape' in node.attributes:
        # Shape is in attributes
        shape_attr = node.attributes['shape']
        if isinstance(shape_attr, (list, tuple)):
            new_shape = tuple(int(x) for x in shape_attr)
        
    if new_shape is None:
        return None
        
    # Handle dimension of -1 (infer from input shape)
    input_size = 1
    for dim in input_shapes[0]:
        input_size *= dim
        
    output_size = 1
    infer_idx = -1
    
    for i, dim in enumerate(new_shape):
        if dim == -1:
            if infer_idx >= 0:
                # Multiple -1 dimensions not allowed
                return None
            infer_idx = i
        else:
            output_size *= dim
    
    if infer_idx >= 0:
        # Infer the size of dimension with -1
        if output_size == 0:
            inferred_dim = 0
        else:
            if input_size % output_size != 0:
                return None
            inferred_dim = input_size // output_size
            
        new_shape = list(new_shape)
        new_shape[infer_idx] = inferred_dim
        new_shape = tuple(new_shape)
    elif input_size != output_size:
        # Input and output sizes must match
        return None
        
    return new_shape


@registry.register_shape_inference("Concat")
def infer_concat_shape(node, input_shapes):
    """Shape inference for Concat"""
    if len(input_shapes) < 1:
        return None
        
    # Get the axis from attributes
    axis = node.attributes.get('axis', 0)
    
    # Handle negative axis
    ndim = len(input_shapes[0])
    if axis < 0:
        axis += ndim
        
    if axis < 0 or axis >= ndim:
        return None
        
    # Check all inputs have same shape except at concat axis
    for shape in input_shapes[1:]:
        if len(shape) != ndim:
            return None
            
        for i in range(ndim):
            if i != axis and shape[i] != input_shapes[0][i]:
                return None
                
    # Compute concat dimension
    concat_dim = sum(shape[axis] for shape in input_shapes)
    
    # Construct output shape
    output_shape = list(input_shapes[0])
    output_shape[axis] = concat_dim
    
    return tuple(output_shape)


@registry.register_shape_inference("Transpose")
def infer_transpose_shape(node, input_shapes):
    """Shape inference for Transpose"""
    if len(input_shapes) != 1:
        return None
    
    input_shape = input_shapes[0]
    
    # Get permutation from attributes
    perm = node.attributes.get('perm', None)
    
    if perm is None:
        # Default: reverse dimensions
        perm = list(range(len(input_shape) - 1, -1, -1))
    
    # Validate permutation
    if len(perm) != len(input_shape):
        return None
    
    # Apply permutation
    output_shape = tuple(input_shape[i] for i in perm)
    
    return output_shape


@registry.register_shape_inference("Constant")
def infer_constant_shape(node, input_shapes):
    """Shape inference for Constant: use the shape of the constant value"""
    if 'value' in node.attributes:
        return node.attributes['value'].shape
    return None


@registry.register_shape_inference("Variable")
def infer_variable_shape(node, input_shapes):
    """Shape inference for Variable: use the shape attribute"""
    return node.shape


@registry.register_shape_inference("Placeholder")
def infer_placeholder_shape(node, input_shapes):
    """Shape inference for Placeholder: use the shape attribute"""
    return node.shape
