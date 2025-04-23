import numpy as np
from julia.core.tensor import Function, Tensor, _ensure_tensor
"""
Operations
"""

class Add(Function):
    @staticmethod
    def forward(ctx, a, b):
        a = _ensure_tensor(a)
        b= _ensure_tensor(b)
        ctx.save_for_backwards(a,b)
        return Tensor(a.data + b.data, requires_grad=a.requires_grad or b.requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        # Start with the incoming gradient data
        grad_a_data = grad_output.data
        grad_b_data = grad_output.data

        # Handle broadcasting for input 'a' if its shape differs
        if a.shape != grad_output.shape:
            # Identify axes present in grad_output but not (or size 1) in 'a'
            axes_to_sum_a = tuple(i for i, dim_out in enumerate(grad_output.shape)
                                if i >= len(a.shape) or a.shape[i] == 1 and dim_out > 1)
            if axes_to_sum_a:
                 # Sum along the broadcasted axes
                 summed_grad_a = np.sum(grad_a_data, axis=axes_to_sum_a, keepdims=True)
                 # Reshape the summed gradient to match the original shape of 'a'
                 grad_a_data = np.reshape(summed_grad_a, a.shape)
            # Handle case where 'a' was scalar expanded
            elif not a.shape and grad_output.shape:
                 grad_a_data = np.array(np.sum(grad_a_data)) # Sum all, ensure scalar shape


        # Handle broadcasting for input 'b' if its shape differs
        if b.shape != grad_output.shape:
             # Identify axes present in grad_output but not (or size 1) in 'b'
             axes_to_sum_b = tuple(i for i, dim_out in enumerate(grad_output.shape)
                                 if i >= len(b.shape) or b.shape[i] == 1 and dim_out > 1)
             if axes_to_sum_b:
                  # Sum along the broadcasted axes
                  summed_grad_b = np.sum(grad_b_data, axis=axes_to_sum_b, keepdims=True)
                  # Reshape the summed gradient to match the original shape of 'b'
                  grad_b_data = np.reshape(summed_grad_b, b.shape)
             # Handle case where 'b' was scalar expanded
             elif not b.shape and grad_output.shape:
                  grad_b_data = np.array(np.sum(grad_b_data)) # Sum all, ensure scalar shape


        # Return Tensor or None based on requires_grad
        grad_a = Tensor(grad_a_data) if a.requires_grad else None
        grad_b = Tensor(grad_b_data) if b.requires_grad else None

        return grad_a, grad_b

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


class LeakyReLU(Function):
    """
    Forward: f(x) = x if x > 0 else alpha * x
    Backward: f'(x) = 1 if x > 0 else alpha
    """
    @staticmethod
    def forward(ctx, a, alpha=0.01):
        ctx.save_for_backwards(a)
        ctx.save_data(alpha=alpha)
        result = a.data.copy()
        result[result < 0] *= alpha
        return Tensor(result, requires_grad=a.requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        a, = ctx.saved_tensors
        alpha = ctx.saved_data['alpha']
        grad = grad_output.data.copy()
        grad[a.data <= 0] *= alpha
        return Tensor(grad)


class PReLU(Function):
    """
    Forward: f(x) = x if x > 0 else alpha * x
    Backward: Computes gradients for both input and the parameter alpha
    """
    @staticmethod
    def forward(ctx, a, alpha):
        ctx.save_for_backwards(a, alpha)
        result = a.data.copy()
        result[result < 0] *= alpha.data
        return Tensor(result, requires_grad=a.requires_grad or alpha.requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        a, alpha = ctx.saved_tensors
        grad_a = grad_output.data.copy()
        mask = a.data <= 0
        grad_a[mask] *= alpha.data
        
        # Gradient for alpha
        grad_alpha = (grad_output.data * a.data * (a.data <= 0)).sum()
        
        return Tensor(grad_a), Tensor(np.array([grad_alpha]))


class ELU(Function):
    """
    Forward: f(x) = x if x > 0 else alpha * (exp(x) - 1)
    Backward: f'(x) = 1 if x > 0 else alpha * exp(x)
    """
    @staticmethod
    def forward(ctx, a, alpha=1.0):
        ctx.save_for_backwards(a)
        ctx.save_data(alpha=alpha)
        result = a.data.copy()
        mask = result <= 0
        result[mask] = alpha * (np.exp(result[mask]) - 1)
        return Tensor(result, requires_grad=a.requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        a, = ctx.saved_tensors
        alpha = ctx.saved_data['alpha']
        grad = grad_output.data.copy()
        mask = a.data <= 0
        grad[mask] *= alpha * np.exp(a.data[mask])
        return Tensor(grad)


class SELU(Function):
    """    
    Parameters fixed to ensure self-normalizing property.
    
    Forward: f(x) = scale * (x if x > 0 else alpha * (exp(x) - 1))
    Backward: f'(x) = scale if x > 0 else scale * alpha * exp(x)
    """
    @staticmethod
    def forward(ctx, a):
        # SELU parameters
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        
        ctx.save_for_backwards(a)
        ctx.save_data(alpha=alpha, scale=scale)
        
        result = a.data.copy()
        mask = result <= 0
        result[mask] = alpha * (np.exp(result[mask]) - 1)
        result *= scale
        
        return Tensor(result, requires_grad=a.requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        a, = ctx.saved_tensors
        alpha = ctx.saved_data['alpha']
        scale = ctx.saved_data['scale']
        
        grad = grad_output.data.copy()
        mask = a.data <= 0
        grad[mask] *= scale * alpha * np.exp(a.data[mask])
        grad[~mask] *= scale
        
        return Tensor(grad)

class Sigmoid(Function):
    """
    Forward: f(x) = 1 / (1 + exp(-x))
    Backward: f'(x) = f(x) * (1 - f(x))
    """
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


class Tanh(Function):
    """
    Forward: f(x) = tanh(x)
    Backward: f'(x) = 1 - tanh(x)^2
    """
    @staticmethod
    def forward(ctx, a):
        tanh_val = np.tanh(a.data)
        ctx.save_data(tanh_val=tanh_val)
        return Tensor(tanh_val, requires_grad=a.requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        tanh_val = ctx.saved_data['tanh_val']
        grad = grad_output.data * (1 - tanh_val ** 2)
        return Tensor(grad)


class Softmax(Function):
    """
    Forward: f(x_i) = exp(x_i) / sum(exp(x_j))
    Backward: Computes full Jacobian for softmax
    """
    @staticmethod
    def forward(ctx, a, dim=-1):
        # Apply max subtraction for numerical stability
        x = a.data - np.max(a.data, axis=dim, keepdims=True)
        exp_x = np.exp(x)
        softmax_output = exp_x / np.sum(exp_x, axis=dim, keepdims=True)
        
        ctx.save_data(softmax_output=softmax_output, dim=dim)
        return Tensor(softmax_output, requires_grad=a.requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        softmax_output = ctx.saved_data['softmax_output']
        dim = ctx.saved_data['dim']
        
        # For each sample in the batch
        if softmax_output.ndim > 1:
            grad = np.zeros_like(softmax_output)
            for i in range(softmax_output.shape[0]):
                # Compute Jacobian
                s = softmax_output[i]
                s_reshape = s.reshape(-1, 1)
                jacobian = np.diagflat(s) - np.dot(s_reshape, s_reshape.T)
                
                # Apply chain rule
                grad[i] = np.dot(jacobian, grad_output.data[i])
        else:
            # Single sample case
            s = softmax_output
            s_reshape = s.reshape(-1, 1)
            jacobian = np.diagflat(s) - np.dot(s_reshape, s_reshape.T)
            grad = np.dot(jacobian, grad_output.data)
            
        return Tensor(grad)


class LogSoftmax(Function):
    """
    Forward: f(x_i) = log(exp(x_i) / sum(exp(x_j))) = x_i - log(sum(exp(x_j)))
    Backward: Uses more numerically stable approach
    """
    @staticmethod
    def forward(ctx, a, dim=-1):
        # Apply max subtraction for numerical stability
        x = a.data - np.max(a.data, axis=dim, keepdims=True)
        exp_x = np.exp(x)
        sum_exp_x = np.sum(exp_x, axis=dim, keepdims=True)
        log_softmax = x - np.log(sum_exp_x)
        
        ctx.save_data(log_softmax=log_softmax, dim=dim)
        return Tensor(log_softmax, requires_grad=a.requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        log_softmax = ctx.saved_data['log_softmax']
        dim = ctx.saved_data['dim']
        
        softmax = np.exp(log_softmax)
        grad = grad_output.data - np.sum(grad_output.data * softmax, axis=dim, keepdims=True) * softmax
        
        return Tensor(grad)


class GELU(Function):
    """
    Forward: f(x) = x * Φ(x), where Φ is the standard Gaussian CDF
    Backward: Gradient calculated using approximation
    """
    @staticmethod
    def forward(ctx, a):
        # Exact computation of GELU (slower but more accurate)
        # sqrt_2_over_pi = np.sqrt(2 / np.pi)
        # cdf = 0.5 * (1 + np.tanh(sqrt_2_over_pi * (a.data + 0.044715 * a.data**3)))
        # gelu = a.data * cdf
        
        # Faster approximation (used in most frameworks)
        x = a.data
        cdf = 0.5 * (1 + np.tanh((0.7978845608 * (x + 0.044715 * x**3))))
        gelu = x * cdf
        
        ctx.save_for_backwards(a)
        ctx.save_data(gelu=gelu, cdf=cdf)
        return Tensor(gelu, requires_grad=a.requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        a, = ctx.saved_tensors
        gelu = ctx.saved_data['gelu']
        cdf = ctx.saved_data['cdf']
        
        x = a.data
        # Derivative of tanh(x) is 1 - tanh(x)²
        tanh_deriv = 1 - np.tanh((0.7978845608 * (x + 0.044715 * x**3)))**2
        # Chain rule for derivative of GELU
        pdf = 0.5 * 0.7978845608 * (1 + 0.134145 * x**2) * tanh_deriv
        grad = grad_output.data * (cdf + x * pdf)
        
        return Tensor(grad)


class Swish(Function):
    """
    Forward: f(x) = x * sigmoid(x)
    Backward: f'(x) = f(x) + sigmoid(x) * (1 - f(x))
    """
    @staticmethod
    def forward(ctx, a, beta=1.0):
        sigmoid_x = 1 / (1 + np.exp(-beta * a.data))
        swish = a.data * sigmoid_x
        
        ctx.save_for_backwards(a)
        ctx.save_data(swish=swish, sigmoid_x=sigmoid_x, beta=beta)
        return Tensor(swish, requires_grad=a.requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        a, = ctx.saved_tensors
        swish = ctx.saved_data['swish']
        sigmoid_x = ctx.saved_data['sigmoid_x']
        beta = ctx.saved_data['beta']
        
        x = a.data
        grad = grad_output.data * (beta * swish + sigmoid_x * (1 - beta * x * sigmoid_x))
        return Tensor(grad)

def extend_tensor_with_activations():
    """Add activation methods to the Tensor class"""
    # Already defined in tensor.py
    # Tensor.relu = lambda self: ReLU.apply(self)
    # Tensor.sigmoid = lambda self: Sigmoid.apply(self)
    Tensor.leaky_relu = lambda self, alpha=0.01: LeakyReLU.apply(self, alpha)
    Tensor.prelu = lambda self, alpha: PReLU.apply(self, alpha)
    Tensor.elu = lambda self, alpha=1.0: ELU.apply(self, alpha)
    Tensor.selu = lambda self: SELU.apply(self)
    Tensor.tanh = lambda self: Tanh.apply(self)
    Tensor.softmax = lambda self, dim=-1: Softmax.apply(self, dim)
    Tensor.log_softmax = lambda self, dim=-1: LogSoftmax.apply(self, dim)
    Tensor.gelu = lambda self: GELU.apply(self)
    Tensor.swish = lambda self, beta=1.0: Swish.apply(self, beta)

# Add methods to Tensor
extend_tensor_with_activations()
