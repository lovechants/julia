import numpy as np
import weakref
import threading
from enum import Enum

class ExecutionMode(Enum):
    INFERENCE = "inference"
    TRAINING = "training"

class AutogradEngine:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        
        self.mode = ExecutionMode.TRAINING
        self.enabled = True
        self._initialized = True
    
    def set_mode(self, mode: ExecutionMode):
        self.mode = mode
    
    def is_training(self) -> bool:
        return self.mode == ExecutionMode.TRAINING and self.enabled
    
    def no_grad(self):
        return NoGradContext()
    
    def enable_grad(self):
        return EnableGradContext()
    
    def backward(self, root_tensor, gradient=None, retain_graph=False, create_graph=False):
        if not self.is_training() or not root_tensor.requires_grad:
            return
        
        if gradient is None:
            if root_tensor.shape == () or root_tensor.shape == (1,):
                gradient = np.ones_like(root_tensor.data)
            else:
                raise ValueError("Gradient must be specified for non-scalar tensors")
        elif hasattr(gradient, 'data'):
            gradient = gradient.data
        else:
            gradient = np.asarray(gradient)
        
        topo_order = self._build_topological_order(root_tensor)
        gradients = {id(root_tensor): gradient}
        for node in reversed(topo_order):
            self._execute_node_backward(node, gradients, create_graph)
        
        self._assign_leaf_gradients(topo_order, gradients, create_graph)
        
        if not retain_graph:
            self._cleanup_graph(root_tensor)
    
    def _build_topological_order(self, root_tensor):

        visited = set()
        topo_order = []
        
        def visit(tensor):
            if not isinstance(tensor, Tensor) or id(tensor) in visited:
                return
            
            visited.add(id(tensor))
            
            if hasattr(tensor, '_backward_node') and tensor._backward_node is not None:
                node = tensor._backward_node
                for input_tensor in node.inputs:
                    if isinstance(input_tensor, Tensor):
                        visit(input_tensor)
            
            topo_order.append(tensor)
        
        visit(root_tensor)
        return topo_order
    
    def _execute_node_backward(self, tensor, gradients, create_graph):
        if not isinstance(tensor, Tensor) or not hasattr(tensor, '_backward_node'):
            return
        
        node = tensor._backward_node
        if node is None:
            return
        
        tensor_id = id(tensor)
        if tensor_id not in gradients:
            return
        
        grad_output = gradients[tensor_id]
        
        try:
            from julia.core.tensor import Tensor as TensorClass
            grad_tensor = TensorClass(grad_output, requires_grad=create_graph)
            
            input_grads = node.fn_cls.backward(node.ctx, grad_tensor)
            
            if not isinstance(input_grads, tuple):
                input_grads = (input_grads,)
            
            if len(input_grads) != len(node.inputs):
                raise RuntimeError(
                    f"Backward function {node.fn_cls.__name__} returned {len(input_grads)} "
                    f"gradients but forward pass had {len(node.inputs)} inputs"
                )
            
            for input_tensor, input_grad in zip(node.inputs, input_grads):
                if isinstance(input_tensor, TensorClass) and input_tensor.requires_grad and input_grad is not None:
                    input_id = id(input_tensor)
                    grad_data = input_grad.data if hasattr(input_grad, 'data') else input_grad
                    
                    if input_id in gradients:
                        gradients[input_id] = gradients[input_id] + grad_data
                    else:
                        gradients[input_id] = grad_data.copy()
        
        except Exception as e:
            print(f"Error in backward pass for {node.fn_cls.__name__}: {e}")
            # Continue with other nodes rather than crashing
    
    def _assign_leaf_gradients(self, topo_order, gradients, create_graph):

        from julia.core.tensor import Tensor as TensorClass
        
        for tensor in topo_order:
            if not isinstance(tensor, TensorClass) or not tensor.requires_grad:
                continue
            
            tensor_id = id(tensor)
            if tensor_id not in gradients:
                continue
            
            # Only assign gradients to leaf tensors or tensors that retain gradients
            should_retain = (getattr(tensor, '_is_leaf', True) or 
                           getattr(tensor, '_retain_grad', False))
            
            if should_retain:
                grad_data = gradients[tensor_id]
                
                if tensor.grad is None:
                    tensor.grad = TensorClass(grad_data.copy(), requires_grad=create_graph)
                else:
                    if create_graph and not tensor.grad.requires_grad:
                        # Need to recreate grad tensor with requires_grad=True
                        tensor.grad = TensorClass(
                            tensor.grad.data + grad_data, 
                            requires_grad=True
                        )
                    else:
                        tensor.grad.data = tensor.grad.data + grad_data
    
    def _cleanup_graph(self, root_tensor):

        visited = set()
        
        def cleanup_recursive(tensor):
            if not isinstance(tensor, Tensor) or id(tensor) in visited:
                return
            
            visited.add(id(tensor))
            
            if hasattr(tensor, '_backward_node') and tensor._backward_node is not None:
                node = tensor._backward_node
                
                # Recursively cleanup input tensors
                for input_tensor in node.inputs:
                    if isinstance(input_tensor, Tensor):
                        cleanup_recursive(input_tensor)
                
                tensor._backward_node = None
        
        cleanup_recursive(root_tensor)

class BackwardNode:
    def __init__(self, fn_cls=None, ctx=None, inputs=None):
        self.fn_cls = fn_cls
        self.ctx = ctx
        self.inputs = inputs or []
        
        # Use weak references to prevent circular references
        self.input_refs = []
        for inp in self.inputs:
            if isinstance(inp, Tensor):
                self.input_refs.append(weakref.ref(inp))
            else:
                self.input_refs.append(lambda: inp)  # Non-tensor inputs
    
    def get_inputs(self):
        inputs = []
        for ref in self.input_refs:
            obj = ref()
            if obj is not None:
                inputs.append(obj)
        return inputs

class NoGradContext:
    """Context manager to disable gradient computation"""
    
    def __init__(self):
        self.prev_enabled = None
    
    def __enter__(self):
        engine = AutogradEngine()
        self.prev_enabled = engine.enabled
        engine.enabled = False
        return self
    
    def __exit__(self, *args):
        engine = AutogradEngine()
        engine.enabled = self.prev_enabled

class EnableGradContext:
    """Context manager to enable gradient computation"""
    
    def __init__(self):
        self.prev_enabled = None
    
    def __enter__(self):
        engine = AutogradEngine()
        self.prev_enabled = engine.enabled
        engine.enabled = True
        return self
    
    def __exit__(self, *args):
        engine = AutogradEngine()
        engine.enabled = self.prev_enabled

_engine = AutogradEngine()

def no_grad():
    return _engine.no_grad()

def enable_grad():
    return _engine.enable_grad()

def set_grad_enabled(enabled: bool):
    _engine.enabled = enabled

def is_grad_enabled() -> bool:
    return _engine.enabled

def Tensor(*args, **kwargs):
    """Import Tensor class when needed to avoid circular imports"""
    from julia.core.tensor import Tensor as TensorClass
    return TensorClass(*args, **kwargs)
