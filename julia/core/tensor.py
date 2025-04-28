import numpy as np
import uuid
"""
Core Tensor class (with autograd)
Inspired by Torch
"""

class Context:
    def __init__(self):
        self.saved_tensors = ()
        self.saved_data = {}

    def save_for_backwards(self, *tensors):
        self.saved_tensors = tensors

    def save_data(self, **kwargs):
        self.saved_data.update(kwargs)


class Function:
    @staticmethod
    def forward(ctx, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError

    @classmethod
    def apply(cls, *args, **kwargs):
        ctx = Context()
        # Forward pass produces the result tensor(s)
        result = cls.forward(ctx, *args, **kwargs) # Assume forward returns Tensor or tuple of Tensors

        # Determine if any input requires grad
        op_requires_grad = any(
            isinstance(arg, Tensor) and arg.requires_grad for arg in args
        )

        if op_requires_grad:
            # Create the backward node using the original inputs
            backward_node = BackwardNode(cls, ctx, args)

            # Assign the node AND set requires_grad flag on the output tensor(s)
            if isinstance(result, Tensor):
                result._backward_node = backward_node
                result.requires_grad = True # <--- Explicitly set the flag
            elif isinstance(result, tuple):
                # Ensure all tensor outputs are marked
                processed_result = []
                for r in result:
                    if isinstance(r, Tensor):
                        r._backward_node = backward_node
                        r.requires_grad = True # <--- Explicitly set the flag
                        processed_result.append(r)
                    else:
                        processed_result.append(r) # Keep non-tensors as is
                result = tuple(processed_result)
            # Handle other potential return types if needed

        # If op_requires_grad is False, result tensors keep their original flag
        return result


class BackwardNode:
    def __init__(self, fn_cls, ctx, inputs):
        self.fn_cls = fn_cls
        self.ctx = ctx
        self.inputs = inputs
        self.next_functions = [] 
        for inp in inputs:
            if isinstance(inp, Tensor) and inp.requires_grad and inp._backward_node:
                self.next_functions.append(inp._backward_node)

class Tensor:
    def __init__(self, data, requires_grad=False, device=None, dtype=None):
        self.id = str(uuid.uuid4())
        if isinstance(data, Tensor):
            self.data = data.data
            self.requires_grad = requires_grad
            self.grad = None 
            self._backward_node = None
        elif isinstance(data, np.ndarray):
            self.data = data
            self.requires_grad = requires_grad
            self.grad = None 
            self._backward_node = None 
        else:
            self.data = np.array(data, dtype=dtype)
            self.requires_grad = requires_grad
            self.grad = None
            self._backward_node = None

        self.device = device or "cpu"
        self.shape = self.data.shape

    def zero_grad(self):
        self.grad = None

    def dropout(self, p: float, training: bool):
        from julia.core.ops import Dropout
        return Dropout.apply(self, p, training)

    def backward(self, gradient=None):
        if not self.requires_grad:
            return # Silently return if no grad required

        # --- Graph Traversal ---
        visited_nodes = set()
        topo_order_nodes = []
        def build_topo(node):
            if node and node not in visited_nodes:
                visited_nodes.add(node)
                for inp in node.inputs:
                     if isinstance(inp, Tensor) and inp._backward_node:
                         build_topo(inp._backward_node)
                topo_order_nodes.append(node)

        if self._backward_node:
             build_topo(self._backward_node)

        # --- Gradient Initialization ---
        if gradient is None:
            if self.shape == () or self.shape == (1,):
                gradient_data = np.ones_like(self.data)
            else:
                raise ValueError("Must specify gradient for non-scalar tensors used as root of backward()")
        elif isinstance(gradient, Tensor):
            gradient_data = gradient.data.copy() # Use copy
        else:
            gradient_data = np.array(gradient).copy() # Convert scalar/list/etc. and copy

        # --- Gradient Accumulation Dictionary (Tensor object -> Accumulated Grad Data) ---
        # Initialize ONCE before the loop
        grads_tensor_keyed = {self: gradient_data}

        # Assign initial gradient to self.grad attribute
        if self.grad is None:
            self.grad = Tensor(gradient_data.copy()) # Use copy
        else:
            self.grad.data += gradient_data # Accumulate if backward called multiple times

        # --- Backpropagation Loop (SINGLE loop) ---
        for node in reversed(topo_order_nodes):
             # Find the output tensor 't' where t._backward_node == node
             output_tensor = None
             output_grad_data = None
             for t, grad_d in grads_tensor_keyed.items():
                 # Check if the tensor 't' has a backward node and if it's the current node
                 if hasattr(t, '_backward_node') and t._backward_node is node:
                     output_tensor = t
                     output_grad_data = grad_d
                     break # Assume one primary output for simplicity

             if output_tensor is None or output_grad_data is None:
                 # This node's output gradient wasn't computed or needed? Skip.
                 # This can happen if a branch of the graph doesn't lead to the final output
                 # print(f"Skipping node {node.fn_cls.__name__} - output gradient not found in grads dict.")
                 continue

             # Compute gradients for the inputs of this operation
             # Pass a Tensor wrapper for the output gradient to backward function
             grad_out_tensor = Tensor(output_grad_data)
             grads_in = node.fn_cls.backward(node.ctx, grad_out_tensor)

             if not isinstance(grads_in, tuple):
                 grads_in = (grads_in,)

             # Distribute gradients to the input tensors
             if len(node.inputs) != len(grads_in):
                 raise RuntimeError(f"Backward function {node.fn_cls.__name__} returned {len(grads_in)} gradients, but forward took {len(node.inputs)} inputs.")

             for inp, grad_in_tensor in zip(node.inputs, grads_in):
                 # Check if input is a Tensor that requires grad and received a gradient
                 if isinstance(inp, Tensor) and inp.requires_grad:
                     if grad_in_tensor is not None:
                         grad_in_data = grad_in_tensor.data
                         # Accumulate in grads dictionary (for further propagation)
                         if inp not in grads_tensor_keyed:
                             grads_tensor_keyed[inp] = grad_in_data.copy() # Use copy
                         else:
                             grads_tensor_keyed[inp] += grad_in_data
                         # Accumulate in .grad attribute (for user access)
                         if inp.grad is None:
                             inp.grad = Tensor(grad_in_data.copy()) # Use copy
                         else:
                             inp.grad.data += grad_in_data


        # Operator overloads 

    def __add__(self, other):
        from julia.core.ops import Add
        return Add.apply(self, _ensure_tensor(other))

    def __mul__(self, other):
        from julia.core.ops import Mul
        return Mul.apply(self, _ensure_tensor(other))

    def __sub__(self, other):
        from julia.core.ops import Sub
        return Sub.apply(self, _ensure_tensor(other))

    def __truediv__(self, other):
        from julia.core.ops import Div
        return Div.apply(self, _ensure_tensor(other))

    def matmul(self, other):
        from julia.core.ops import MatMul
        return MatMul.apply(self, _ensure_tensor(other))

    def relu(self):
        from julia.core.ops import ReLU
        return ReLU.apply(self)

    def sigmoid(self):
        from julia.core.ops import Sigmoid
        return Sigmoid.apply(self)

    def reshape(self, *shape):
        from julia.core.ops import Reshape
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = shape[0]
        return Reshape.apply(self, shape)

    def transpose(self):
        from julia.core.ops import Transpose
        return Transpose.apply(self)


def _ensure_tensor(data):
    if isinstance(data, Tensor):
        return data
    return Tensor(data)
