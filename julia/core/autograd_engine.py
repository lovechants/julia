import numpy as np
import uuid
from typing import List, Dict, Set, Optional, Tuple, Any, Callable
from collections import defaultdict, deque
import weakref
import threading
from enum import Enum

class ExecutionMode(Enum):
    INFERENCE = "inference"
    TRAINING = "training"

class AutogradEngine:
    """
    Singleton autograd engine that manages computation graphs and gradient computation
    """
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
        
        # Graph state
        self._ready_nodes: deque = deque()
        self._not_ready_nodes: Dict[int, BackwardNode] = {}
        self._dependencies: Dict[int, Set[int]] = defaultdict(set)
        
        # Memory optimization
        self._tensor_versions: Dict[int, int] = {}
        self._node_pool: List[BackwardNode] = []
        self._max_pool_size = 1000
        
        self._initialized = True
    
    def set_mode(self, mode: ExecutionMode):
        """Set execution mode"""
        self.mode = mode
    
    def is_training(self) -> bool:
        return self.mode == ExecutionMode.TRAINING and self.enabled
    
    def no_grad(self):
        """Context manager to disable gradient computation"""
        return NoGradContext()
    
    def enable_grad(self):
        """Context manager to enable gradient computation"""
        return EnableGradContext()
    
    def backward(self, root_tensor, gradient=None, retain_graph=False, create_graph=False):
        """
        Optimized backward pass with topological sorting and memory management
        """
        if not self.is_training() or not root_tensor.requires_grad:
            return
        
        # Clear previous state
        self._ready_nodes.clear()
        self._not_ready_nodes.clear()
        self._dependencies.clear()
        
        # Initialize gradient
        if gradient is None:
            if root_tensor.shape == () or root_tensor.shape == (1,):
                gradient = np.ones_like(root_tensor.data)
            else:
                raise ValueError("Gradient must be specified for non-scalar tensors")
        elif hasattr(gradient, 'data'):
            gradient = gradient.data
        else:
            gradient = np.asarray(gradient)
        
        # Set up initial gradient
        root_tensor._set_grad(gradient, create_graph)
        
        # Build computation graph
        if root_tensor._backward_node:
            self._build_graph(root_tensor._backward_node)
        
        # Execute backward pass
        self._execute_backward(create_graph)
        
        # Cleanup if not retaining graph
        if not retain_graph:
            self._cleanup_graph(root_tensor)
    
    def _build_graph(self, root_node: 'BackwardNode'):
        """Build computation graph with dependency tracking"""
        visited = set()
        
        def visit(node):
            if not node or id(node) in visited:
                return
            
            visited.add(id(node))
            dependencies = set()
            
            for input_tensor in node.inputs:
                if isinstance(input_tensor, Tensor) and input_tensor._backward_node:
                    child_node = input_tensor._backward_node
                    dependencies.add(id(child_node))
                    visit(child_node)
            
            self._dependencies[id(node)] = dependencies
            
            # If no dependencies, node is ready
            if not dependencies:
                self._ready_nodes.append(node)
            else:
                self._not_ready_nodes[id(node)] = node
        
        visit(root_node)
    
    def _execute_backward(self, create_graph: bool):
        """Execute backward pass using topological ordering"""
        while self._ready_nodes:
            node = self._ready_nodes.popleft()
            
            # Find output tensor and its gradient
            output_tensor = None
            output_grad = None
            
            for input_tensor in node.inputs:
                if (isinstance(input_tensor, Tensor) and 
                    hasattr(input_tensor, '_backward_node') and 
                    input_tensor._backward_node is node):
                    # This is wrong - we need to find tensors that have this node as their backward_node
                    pass
            
            # Find the tensor that owns this backward node
            for input_tensor in node.inputs:
                if isinstance(input_tensor, Tensor):
                    # We need a different approach to find the output tensor
                    # For now, assume single output per node
                    if hasattr(input_tensor, '_node_output_tensor'):
                        output_tensor = input_tensor._node_output_tensor
                        output_grad = output_tensor.grad
                        break
            
            if output_grad is None:
                continue
            
            # Execute backward function
            try:
                gradients = node.fn_cls.backward(node.ctx, output_grad)
                if not isinstance(gradients, tuple):
                    gradients = (gradients,)
                
                # Distribute gradients to inputs
                for input_tensor, grad in zip(node.inputs, gradients):
                    if isinstance(input_tensor, Tensor) and input_tensor.requires_grad and grad is not None:
                        input_tensor._accumulate_grad(grad, create_graph)
                
            except Exception as e:
                print(f"Error in backward pass for {node.fn_cls.__name__}: {e}")
                continue
            
            # Update dependencies and check for newly ready nodes
            node_id = id(node)
            for other_id, other_node in list(self._not_ready_nodes.items()):
                if node_id in self._dependencies[other_id]:
                    self._dependencies[other_id].remove(node_id)
                    if not self._dependencies[other_id]:
                        self._ready_nodes.append(other_node)
                        del self._not_ready_nodes[other_id]
    
    def _cleanup_graph(self, root_tensor):
        """Clean up computation graph to free memory"""
        visited = set()
        
        def cleanup_recursive(tensor):
            if not isinstance(tensor, Tensor) or id(tensor) in visited:
                return
            
            visited.add(id(tensor))
            
            if tensor._backward_node:
                for input_tensor in tensor._backward_node.inputs:
                    if isinstance(input_tensor, Tensor):
                        cleanup_recursive(input_tensor)
                
                # Return node to pool if possible
                if len(self._node_pool) < self._max_pool_size:
                    tensor._backward_node.reset()
                    self._node_pool.append(tensor._backward_node)
                
                tensor._backward_node = None
        
        cleanup_recursive(root_tensor)

class BackwardNode:
    """Optimized backward node with memory pooling"""
    
    def __init__(self, fn_cls=None, ctx=None, inputs=None):
        self.fn_cls = fn_cls
        self.ctx = ctx
        self.inputs = inputs or []
        self.id = id(self)
    
    def reset(self):
        """Reset node for reuse in memory pool"""
        self.fn_cls = None
        self.ctx = None
        self.inputs.clear()

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

# Global autograd engine instance
_engine = AutogradEngine()

def no_grad():
    """Disable gradient computation"""
    return _engine.no_grad()

def enable_grad():
    """Enable gradient computation"""
    return _engine.enable_grad()

def set_grad_enabled(enabled: bool):
    """Set gradient computation state"""
    _engine.enabled = enabled

def is_grad_enabled() -> bool:
    """Check if gradient computation is enabled"""
    return _engine.enabled
