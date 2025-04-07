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
        result = cls.forward(ctx, *args, **kwargs)
        requires_grad = any(
            isinstance(arg, Tensor) and arg.requires_grad for arg in args
        )
        if requires_grad:
            backward_node = BackwardNode(cls, ctx, args)
            if isinstance(result, Tensor):
                result._backward_node = backward_node
            elif isinstance(result, tuple):
                for r in result:
                    if isinstance(r, Tensor):
                        r._backward_node = backward_node
        return result


class BackwardNode:
    def __init__(self, fn_cls, ctx, inputs):
        self.fn_cls = fn_cls
        self.ctx = ctx
        self.inputs = inputs
        self.next_functions = []
        for inp in inputs:
            if isinstance(inp, Tensor) and inp.requires_grad:
                if inp._backward_node is not None:
                    self.next_functions.append(inp._backward_node)


class Tensor:
    def __init__(self, data, requires_grad=False, device=None, dtype=None):
        self.id = str(uuid.uuid4())
        if isinstance(data, Tensor):
            self.data = data.data
        elif isinstance(data, np.ndarray):
            self.data = data
        else:
            self.data = np.array(data, dtype=dtype)
        self.requires_grad = requires_grad
        self.grad = None
        self._backward_node = None
        self.device = device or "cpu"
        self.shape = self.data.shape

    def zero_grad(self):
        self.grad = None

    def backward(self, gradient=None):
        if gradient is None:
            if self.shape == () or self.shape == (1,):
                gradient = np.ones_like(self.data)
            else:
                raise ValueError("Must specify gradient for non-scalar tensors")
        if isinstance(gradient, Tensor):
            gradient = gradient.data
        self.grad = Tensor(gradient)

        visited = set()
        topo_order = []

        def build_topo(node):
            if node and node not in visited:
                visited.add(node)
                for next_node in getattr(node, 'next_functions', []):
                    build_topo(next_node)
                topo_order.append(node)

        build_topo(self._backward_node)

        grads = {self._backward_node: self.grad}

        for node in reversed(topo_order):
            grad_out = grads[node]
            grads_in = node.fn_cls.backward(node.ctx, grad_out)
            if not isinstance(grads_in, tuple):
                grads_in = (grads_in,)
            for inp, grad_in in zip(node.inputs, grads_in):
                if isinstance(inp, Tensor) and inp.requires_grad:
                    if inp.grad is None:
                        inp.grad = grad_in
                    else:
                        inp.grad.data += grad_in.data
                    if inp._backward_node:
                        grads[inp._backward_node] = grad_in

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

def _ensure_tensor(data):
    if isinstance(data, Tensor):
        return data
    return Tensor(data)


