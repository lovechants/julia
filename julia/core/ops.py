import numpy as np
from julia.core.tensor import Function, Tensor
"""
Operations
"""

class Add(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backwards(a,b)
        return Tensor(a.data + b.data, requires_grad=a.requires_grad or b.requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        return grad_output, grad_output

class Sub(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backwards(a, b)
        return Tensor(a.data - b.data, requires_grad=a.requires_grad or b.requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, Tensor(-grad_output.data)


class Mul(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backwards(a, b)
        return Tensor(a.data * b.data, requires_grad=a.requires_grad or b.requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        return Tensor(grad_output.data * b.data), Tensor(grad_output.data * a.data)


class Div(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backwards(a, b)
        return Tensor(a.data / b.data, requires_grad=a.requires_grad or b.requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        return (
            Tensor(grad_output.data / b.data),
            Tensor(-grad_output.data * a.data / (b.data ** 2)),
        )


class MatMul(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backwards(a, b)
        return Tensor(np.matmul(a.data, b.data), requires_grad=a.requires_grad or b.requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_a = np.matmul(grad_output.data, b.data.T)
        grad_b = np.matmul(a.data.T, grad_output.data)
        return Tensor(grad_a), Tensor(grad_b)


class ReLU(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backwards(a)
        return Tensor(np.maximum(0, a.data), requires_grad=a.requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        a, = ctx.saved_tensors
        grad = grad_output.data * (a.data > 0)
        return Tensor(grad)


class Sigmoid(Function):
    @staticmethod
    def forward(ctx, a):
        sig = 1 / (1 + np.exp(-a.data))
        ctx.save_data(sig=sig)
        return Tensor(sig, requires_grad=a.requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        sig = ctx.saved_data['sig']
        grad = grad_output.data * sig * (1 - sig)
        return Tensor(grad)


class Reshape(Function):
    @staticmethod
    def forward(ctx, tensor, shape):
        ctx.save_for_backwards(tensor)
        ctx.save_data(original_shape=tensor.shape)
        return Tensor(tensor.data.reshape(shape), requires_grad=tensor.requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        original_shape = ctx.saved_data['original_shape']
        return Tensor(grad_output.data.reshape(original_shape)), None
