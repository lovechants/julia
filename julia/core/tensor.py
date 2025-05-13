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
                result._is_leaf = False 
                result._grad_fn = cls.__name__
            elif isinstance(result, tuple):
                # multiple output 
                processed_result = []
                for i, r in enumerate(result):
                    if isinstance(r, Tensor):
                        r._backward_node = backward_node
                        r.requires_grad = True 
                        r._is_leaf = False 
                        r._grad_fn = cls.__name__
                        r._output_index = i 
                        processed_result.append(r)
                    else:
                        processed_result.append(r)
        # If op_requires_grad is False, result tensors keep their original flag
        return result


class BackwardNode:
    def __init__(self, fn_cls, ctx, inputs):
        self.fn_cls = fn_cls
        self.ctx = ctx
        self.inputs = inputs
        self.next_functions = []
        self.grad_outputs = {}
        self.grad_output_count = 0 
        self.num_expect_grads = None 
        for inp in inputs:
            if isinstance(inp, Tensor) and inp.requires_grad and inp._backward_node:
                self.next_functions.append(inp._backward_node)

    def accmulate_grad(self, grad, idx=0):
        """
        Accumlate gradient for a specfic output tensor 
        Args:
            frad: the gradient to accmulate 
            idx: the index of the output tensor 
        """
        if idx in self.grad_outputs:
            self.grad_outputs[idx] = self.grad_outputs[idx] + grad 
        else:
            self.grad_outputs[idx] = grad 
            self.grad_output_count += 1 

        if self.num_expect_grads is not None and self.grad_output_count >=self.num_expect_grads:
            return True 
        return False

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

        # For hooks 
        self._is_leaf = True # user & parameter created tensors 
        self._retain_grad = False 
        self._grad_fn = None 
        self._backward_hooks = {} 
        self._next_hook_id = 0 
        # ^^^ Think about tensor bindings to be added for the compiled autograd engine 

    def zero_grad(self):
        self.grad = None

    def dropout(self, p: float, training: bool):
        from julia.core.ops import Dropout
        return Dropout.apply(self, p, training)

    # Tensor Detach stuff 

    def detach(self):
        """
        Return a new tensor detached from computation graph 
        Same data but does not require gradients 

        Returns: 
            Tensor: A new Tensor with the same data but no grad history 
        """

        return Tensor(self.data.copy(), requires_grad=False, device=self.device)

    def detach_(self):
        """
        Inplace version 
        Clears gradient and backward_node 

        Returns: 
            self: The same Tensor but it is detached from the computation graph 
        """

        self.requires_grad = False
        self._backward_node = None 
        self.grad = None 

        return self

    def clone(self):
       """
       Clone -> returns a new tensor with the same data and grad requirements 
       Tracks gradients independely from the original tensor it cloned 

       Returns:
            Tensor: new tensor with a the same data and grad requirements 
       """

       return Tensor(self.data.copy(), requires_grad=self.requires_grad, device=self.device)

    def retain_grad(self):
        """
        Gradient retention for non-leaf Tensors 
        Default behavior: leaf Tensors created by the user or model params retain their gradients after computing the backward gradient.
        This forces a Tensor to retain their gradient 
        """
        self._retain_grad = True
        return self
    
    def backward(self, gradient=None, retain_graph=False, create_graph=False):
        """
        Compute gradient of tensor w.r.t 
        Args:
            gradient: Gradient of the current tensor (default=None, treated as 1.0 (implied))
            retain_graph: If True, computation graph is kept for future backward calls (default=False)
            create_graph: If True, graph of the derivative is constructed for higher order derivatives
        """

        if not self.requires_grad:
            return 

        visited_nodes = set()
        topo_order_nodes = []

        def build_topo(node):
            if node and node not in visited_nodes:
                visited_nodes.add(node)
                for inp in node.inputs:
                    if isinstance(inp, Tensor) and inp.requires_grad and inp._backward_node:
                        build_topo(inp._backward_node)
                topo_order_nodes.append(node)


        if self._backward_node:
            build_topo(self._backward_node)


        # Init Gradient 
        if gradient is None: 
            if self.shape == () or self.shape == (1,):
                gradient_data = np.ones_like(self.data)
            else:
                raise ValueError("Must specify gradient for non-scalr tensors used as roots")

        elif isinstance(gradient, Tensor):
            gradient_data = gradient.data.copy()
        else:
            gradient_data = np.array(gradient).copy()

        grads_tensor_keyed = {self: gradient_data}

        # Assign init gradient to self.grad 
        if self.grad is None:
            self.grad = Tensor(gradient_data.copy(), requires_grad=create_graph)
        else:
            if self.grad.requires_grad != create_graph:
                self.grad = Tensor(self.grad.data + gradient_data, requires_grad=create_graph)
            else:
                self.grad.data += gradient_data

        # Backprop 
        for node in reversed(topo_order_nodes):
            output_tensor = None 
            output_grad_data = None 
            for t, grad_d in grads_tensor_keyed.items():
                if hasattr(t, '_backward_node') and t._backward_node is node:
                    output_tensor = t 
                    output_grad_data = grad_d 
                    break 

            if output_tensor is None or output_grad_data is None:
                continue 

            grad_out_tensor = Tensor(output_grad_data, requires_grad=create_graph)
            
            # Apply hooks 
            if hasattr(output_tensor, '_backward_hooks') and output_tensor._backward_hooks:
                for hook in output_tensor._backward_hooks.values():
                    hook_result = hook(grad_out_tensor)
                    if hook_result is not None:
                        if not isinstance(hook_result, Tensor):
                            raise TypeError(f"Hook returned invalid type: {type(hook_result)}")
                        grad_out_tensor = hook_result

            grads_in = node.fn_cls.backward(node.ctx, grad_out_tensor)

            if not isinstance(grads_in, tuple):
                grads_in = (grads_in, )

            # Distribute gradients 
            if len(node.inputs) != len(grads_in):
                raise RuntimeError(f"Backwards function {node.fn_cls.__name__} returned {len(grads_in)} " 
                                   f"gradients, forward took {len(node.inputs)} inputs")

            for inp, grads_in_tensor in zip(node.inputs, grads_in):
                if isinstance(inp, Tensor) and inp.requires_grad:
                    if grads_in_tensor is not None:
                        grad_in_data = grads_in_tensor.data 

                        if inp not in grads_tensor_keyed:
                            grads_tensor_keyed[inp] = grad_in_data.copy()
                        else:
                            grads_tensor_keyed[inp] += grad_in_data

                        should_retain_grad = inp._is_leaf or hasattr(inp, '_retain_grad')

                        if should_retain_grad:
                            if inp.grad is None:
                                inp.grad = Tensor(grad_in_data.copy(), requires_grad=create_graph)
                            else:
                                if inp.grad.requires_grad != create_graph:
                                    inp.grad = Tensor(inp.grad.data + grad_in_data, requires_grad=create_graph)
                                else:
                                    inp.grad.data += grad_in_data

        if not retain_graph:
            # Clear backward node for the output tensor 
            self._backward_node = None 


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

    def sum(self):
        from julia.core.ops import Sum
        return Sum.apply(self)


def _ensure_tensor(data):
    if isinstance(data, Tensor):
        return data
    return Tensor(data)
