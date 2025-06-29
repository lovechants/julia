import numpy as np
from julia.core.tensor import Function, Tensor, _ensure_tensor, Context
from typing import Tuple, List
"""
Operations
"""

class Transpose(Function):
    @staticmethod
    def forward(ctx: Context, tensor: Tensor) -> Tensor:
        if tensor.data.ndim < 2:
            # Handle 0D or 1D tensors if necessary, maybe return unchanged?
            # Or raise error depending on desired behavior.
            # For typical matmul, we expect at least 2D.
             return tensor # Or raise error
        ctx.save_data(shape=tensor.shape) # Save original shape if needed for backward
        # No need to save tensor itself if backward only needs grad_output
        result_data = tensor.data.T.copy() # Ensure it's a copy
        return Tensor(result_data) # Let Function.apply handle requires_grad and _backward_node

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        # The gradient of transpose is just the transpose of the gradient
        if grad_output.data.ndim < 2:
             return grad_output 
        # No need to use ctx if only grad_output is needed
        result_grad_data = grad_output.data.T.copy()
        return Tensor(result_grad_data)

class Dropout(Function):
    @staticmethod
    def forward(ctx, tensor: Tensor, p: float, training: bool) -> Tensor:
        mask = None; keep_prob = 1.0 - p
        output_data = tensor.data
        if training and p > 0:
            mask = (np.random.rand(*tensor.shape) < keep_prob).astype(tensor.data.dtype)
            output_data = (tensor.data * mask) / keep_prob
        ctx.save_data(mask=mask, p=p, training=training, keep_prob=keep_prob)
        return Tensor(output_data)
    
    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor | None, None, None]:
        mask = ctx.saved_data['mask']
        p = ctx.saved_data['p']
        training = ctx.saved_data['training']
        keep_prob = ctx.saved_data['keep_prob']
        grad_input_data = grad_output.data
        if training and p > 0:
            grad_input_data = (grad_output.data * mask) / keep_prob
        return Tensor(grad_input_data), None, None

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
            # Handle broadcasting across dimensions
            if len(a.shape) < len(grad_output.shape):
                # Need to sum over the extra dimensions
                extra_dims = len(grad_output.shape) - len(a.shape)
                sum_dims = tuple(range(extra_dims))
                grad_a_data = np.sum(grad_a_data, axis=sum_dims, keepdims=True)
                
                # Now remove the extra singleton dimensions
                grad_a_data = grad_a_data.reshape(a.shape)
            else:
                # Same number of dimensions but some are 1
                sum_dims = tuple(i for i, dim in enumerate(a.shape) 
                                if dim == 1 and grad_output.shape[i] > 1)
                if sum_dims:
                    grad_a_data = np.sum(grad_a_data, axis=sum_dims, keepdims=True)
                    
                    # Reshape to match original shape
                    grad_a_data = np.reshape(grad_a_data, a.shape)

        # Handle broadcasting for input 'b' if its shape differs
        if b.shape != grad_output.shape:
            # Handle broadcasting across dimensions
            if len(b.shape) < len(grad_output.shape):
                # Need to sum over the extra dimensions
                extra_dims = len(grad_output.shape) - len(b.shape)
                sum_dims = tuple(range(extra_dims))
                grad_b_data = np.sum(grad_b_data, axis=sum_dims, keepdims=True)
                
                # Now remove the extra singleton dimensions
                grad_b_data = grad_b_data.reshape(b.shape)
            else:
                # Same number of dimensions but some are 1
                sum_dims = tuple(i for i, dim in enumerate(b.shape) 
                                if dim == 1 and grad_output.shape[i] > 1)
                if sum_dims:
                    grad_b_data = np.sum(grad_b_data, axis=sum_dims, keepdims=True)
                    
                    # Reshape to match original shape
                    grad_b_data = np.reshape(grad_b_data, b.shape)

        # Return Tensor or None based on requires_grad
        grad_a = Tensor(grad_a_data) if a.requires_grad else None
        grad_b = Tensor(grad_b_data) if b.requires_grad else None

        return grad_a, grad_b

class Sum(Function):
    @staticmethod
    def forward(ctx, tensor):
        """Summation"""
        ctx.save_for_backwards(tensor)
        result = np.sum(tensor.data)
        return Tensor(result)

    @staticmethod
    def backward(ctx, grad_output):
        """Backward Pass"""
        tensor,= ctx.saved_tensors
        grad_input = np.ones_like(tensor.data) * grad_output.data 
        return Tensor(grad_input)

class Exp(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backwards(a)
        result = np.exp(a.data)
        return Tensor(result, requires_grad=a.requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        a, = ctx.saved_tensors
        grad = grad_output.data * np.exp(a.data)
        return Tensor(grad)

class Log(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backwards(a)
        result = np.log(a.data + 1e-8) # Epsilon for stability #TODO pure math functions for calculations 
        return Tensor(result, requires_grad=a.requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        a, = ctx.saved_tensors
        grad = grad_output.data / (a.data + 1e-8)
        return Tensor(grad)

"""
TODO: Some other core operations to finish 
POW 
Mean
Variance (Var)
Abs
Max | Min 
(Extend these to tensor class + add to op registry)
"""

class Pow(Function):
    @staticmethod
    def forward(ctx, a, power):
        ctx.save_for_backwards(a)
        ctx.save_data(power=power)
        # Fix: Handle power as scalar or tensor properly
        if isinstance(power, Tensor):
            power_val = power.data
        else:
            power_val = power
        result = np.power(a.data, power_val)
        return Tensor(result, requires_grad=a.requires_grad)
    
    @staticmethod
    def backward(ctx, grad_output):
        a, = ctx.saved_tensors
        power = ctx.saved_data['power']
        if isinstance(power, Tensor):
            power_val = power.data
        else:
            power_val = power
            
        grad = grad_output.data * power_val * np.power(a.data, power_val - 1)
        return Tensor(grad), None

class Mean(Function):
    @staticmethod
    def forward(ctx, a, dim=None, keepdim=False):
        ctx.save_for_backwards(a)
        ctx.save_data(dim=dim, keepdim=keepdim, input_shape=a.shape)
        result = np.mean(a.data, axis=dim, keepdims=keepdim)
        return Tensor(result, requires_grad=a.requires_grad)
    
    @staticmethod
    def backward(ctx, grad_output):
        a, = ctx.saved_tensors
        dim = ctx.saved_data['dim']
        keepdim = ctx.saved_data['keepdim']
        input_shape = ctx.saved_data['input_shape']
        
        # Calculate the number of elements that were averaged
        if dim is None:
            numel = np.prod(input_shape)
        else:
            if isinstance(dim, int):
                numel = input_shape[dim]
            else:
                numel = np.prod([input_shape[d] for d in dim])
        
        # Expand gradient back to original shape
        grad = grad_output.data / numel
        if not keepdim and dim is not None:
            grad = np.expand_dims(grad, axis=dim)
        
        # Broadcast to original shape
        grad = np.broadcast_to(grad, input_shape)
        return Tensor(grad)

class Var(Function):
    @staticmethod
    def forward(ctx, a, dim=None, keepdim=False, unbiased=True):
        ctx.save_for_backwards(a)
        ctx.save_data(dim=dim, keepdim=keepdim, unbiased=unbiased)
        
        ddof = 1 if unbiased else 0
        result = np.var(a.data, axis=dim, keepdims=keepdim, ddof=ddof)
        return Tensor(result, requires_grad=a.requires_grad)
    
    @staticmethod
    def backward(ctx, grad_output):
        a, = ctx.saved_tensors
        dim = ctx.saved_data['dim']
        keepdim = ctx.saved_data['keepdim']
        unbiased = ctx.saved_data['unbiased']
        
        # Compute mean
        mean_val = np.mean(a.data, axis=dim, keepdims=True)
        
        # Calculate N (number of elements)
        if dim is None:
            N = a.data.size
        else:
            if isinstance(dim, int):
                N = a.shape[dim]
            else:
                N = np.prod([a.shape[d] for d in dim])
        
        if unbiased and N > 1:
            N = N - 1
        
        # Gradient computation
        diff = a.data - mean_val
        grad = 2 * diff / N
        
        # Expand gradient dimensions if needed
        if not keepdim and dim is not None:
            grad_output_expanded = np.expand_dims(grad_output.data, axis=dim)
        else:
            grad_output_expanded = grad_output.data
        
        grad = grad * np.broadcast_to(grad_output_expanded, grad.shape)
        return Tensor(grad)

class Abs(Function):
    @staticmethod  
    def forward(ctx, a):
        ctx.save_for_backwards(a)
        result = np.abs(a.data)
        return Tensor(result, requires_grad=a.requires_grad)
    
    @staticmethod
    def backward(ctx, grad_output):
        a, = ctx.saved_tensors
        grad = grad_output.data * np.sign(a.data)
        return Tensor(grad)

class Max(Function):
    @staticmethod
    def forward(ctx, a, dim=None, keepdim=False):
        ctx.save_for_backwards(a)
        ctx.save_data(dim=dim, keepdim=keepdim)
        
        if dim is None:
            result = np.max(a.data)
            ctx.save_data(max_indices=np.unravel_index(np.argmax(a.data), a.shape))
        else:
            result = np.max(a.data, axis=dim, keepdims=keepdim)
            ctx.save_data(max_indices=np.argmax(a.data, axis=dim))
        
        return Tensor(result, requires_grad=a.requires_grad)
    
    @staticmethod
    def backward(ctx, grad_output):
        a, = ctx.saved_tensors
        dim = ctx.saved_data['dim']
        keepdim = ctx.saved_data['keepdim']
        max_indices = ctx.saved_data['max_indices']
        
        grad = np.zeros_like(a.data)
        
        if dim is None:
            # Global max
            grad[max_indices] = grad_output.data
        else:
            # Max along axis
            if not keepdim:
                grad_output_expanded = np.expand_dims(grad_output.data, axis=dim)
            else:
                grad_output_expanded = grad_output.data
            
            # Create indices for advanced indexing
            indices = [slice(None)] * a.data.ndim
            indices[dim] = max_indices
            grad[tuple(indices)] = grad_output_expanded
        
        return Tensor(grad)

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
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor | None, Tensor | None]:
        a, b = ctx.saved_tensors
        grad_output_data = grad_output.data

        # grad_a = grad_output @ b.T
        # Use swapaxes for transpose that works on batches
        b_T_data = np.swapaxes(b.data, -1, -2)
        grad_a_data = np.matmul(grad_output_data, b_T_data)

        # grad_b = a.T @ grad_output
        # Use swapaxes for transpose that works on batches
        a_T_data = np.swapaxes(a.data, -1, -2)
        grad_b_data = np.matmul(a_T_data, grad_output_data)

        # If broadcasting occurred in the forward pass (e.g., batch dim added),
        # the gradients might have extra dimensions that need summing out.
        # Example: If a=(M,N) and b=(N,P) -> output=(M,P), grad_output=(M,P)
        #          grad_a=(M,N), grad_b=(N,P) -> Shapes match, no sum needed.
        # Example: If a=(B,M,N) and b=(N,P) -> output=(B,M,P), grad_output=(B,M,P)
        #          grad_a = (B,M,P) @ (P,N) -> (B,M,N) -> Matches a, no sum needed.
        #          grad_b = (B,N,M) @ (B,M,P) -> (B,N,P) -> Needs sum over B dim for b!
        # We need to sum grad_a/grad_b if their ndim > original a/b ndim

        def sum_leading_dims_if_needed(grad_data, original_tensor):
             if grad_data.ndim > original_tensor.data.ndim:
                 dims_to_sum = tuple(range(grad_data.ndim - original_tensor.data.ndim))
                 return np.sum(grad_data, axis=dims_to_sum, keepdims=False)
             # Handle cases where a dimension was broadcasted (e.g., size 1)
             # This is more complex, but start with summing leading dims.
             # A more robust solution might check original_tensor.shape vs grad_data.shape
             # element-wise and sum where original dim was 1 but grad dim > 1.
             # For now, let's assume the main issue is the extra batch dims.
             return grad_data

        grad_a_data = sum_leading_dims_if_needed(grad_a_data, a)
        grad_b_data = sum_leading_dims_if_needed(grad_b_data, b)

        # --- Final Shape Check (Optional Sanity Check) ---
        # It's good practice but shouldn't be strictly necessary if logic is right
        if grad_a_data.shape != a.shape:
             # This might still happen if a dimension was size 1 and got broadcast
             # Add more sophisticated summing/reshaping if needed, but the
             # core matmuls above should be the primary calculation.
             # Example: if a.shape[i] == 1 and grad_a_data.shape[i] > 1: sum axis i
             print(f"Warning: MatMul grad_a shape mismatch {grad_a_data.shape} vs {a.shape}")
             # Attempt a reshape if product matches, otherwise error likely
             if np.prod(grad_a_data.shape) == np.prod(a.shape):
                 grad_a_data = grad_a_data.reshape(a.shape)


        if grad_b_data.shape != b.shape:
             print(f"Warning: MatMul grad_b shape mismatch {grad_b_data.shape} vs {b.shape}")
             if np.prod(grad_b_data.shape) == np.prod(b.shape):
                 grad_b_data = grad_b_data.reshape(b.shape)


        grad_a = Tensor(grad_a_data) if a.requires_grad else None
        grad_b = Tensor(grad_b_data) if b.requires_grad else None
        return grad_a, grad_b

class MatMulTranspose(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backwards(a, b)
        # Transpose b for matrix multiplication
        b_t = b.data.T
        return Tensor(np.matmul(a.data, b_t), requires_grad=a.requires_grad or b.requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        
        # For y = a @ b.T:
        # grad_a = grad_output @ b  (without transpose)
        # grad_b = a.T @ grad_output  (with tranpose)
        
        # Handle gradient for a
        grad_a = None
        if a.requires_grad:
            grad_a_data = np.matmul(grad_output.data, b.data)
            grad_a = Tensor(grad_a_data)
        
        # Handle gradient for b
        grad_b = None
        if b.requires_grad:
            grad_b_data = np.matmul(a.data.T, grad_output.data)
            grad_b = Tensor(grad_b_data)
        
        return grad_a, grad_b

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

class Slice(Function):
    @staticmethod
    def forward(ctx, tensor: Tensor, slice_obj) -> Tensor:
        ctx.save_data(original_shape=tensor.shape, slice_obj=slice_obj)
        result_data = tensor.data[slice_obj]
        return Tensor(result_data)
    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor | None, None]:
        original_shape = ctx.saved_data['original_shape']
        slice_obj = ctx.saved_data['slice_obj']
        grad_input_data = np.zeros(original_shape, dtype=grad_output.data.dtype)
        grad_input_data[slice_obj] = grad_output.data
        return Tensor(grad_input_data), None

class Stack(Function):
    @staticmethod
    def forward(ctx, tensors: List[Tensor], axis: int) -> Tensor:
        tensor_list = [_ensure_tensor(t) for t in tensors]
        if not tensor_list: raise ValueError("Cannot stack empty list")
        ctx.save_data(axis=axis, num_tensors=len(tensor_list))
        data_list = [t.data for t in tensor_list]
        result_data = np.stack(data_list, axis=axis)
        return Tensor(result_data)
    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[List[Tensor | None], None]:
        axis = ctx.saved_data['axis']
        num_tensors = ctx.saved_data['num_tensors']
        grad_inputs_data = []
        slices = [slice(None)] * grad_output.data.ndim
        for i in range(num_tensors):
            slices[axis] = i
            grad_inputs_data.append(grad_output.data[tuple(slices)])
        grad_inputs = [Tensor(grad_data) for grad_data in grad_inputs_data]
        return tuple(grad_inputs) + (None,)

class Concatenate(Function):
    @staticmethod
    def forward(ctx, tensors: List[Tensor], axis: int) -> Tensor:
        tensor_list = [_ensure_tensor(t) for t in tensors]
        if not tensor_list: raise ValueError("Cannot concatenate empty list")
        shapes = [t.shape[axis] for t in tensor_list]
        ctx.save_data(axis=axis, shapes=shapes)
        data_list = [t.data for t in tensor_list]
        result_data = np.concatenate(data_list, axis=axis)
        return Tensor(result_data)
    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[List[Tensor | None], None]:
        axis = ctx.saved_data['axis']
        shapes = ctx.saved_data['shapes']
        split_indices = np.cumsum(shapes)[:-1]
        grad_inputs_data = np.split(grad_output.data, split_indices, axis=axis)
        grad_inputs = [Tensor(grad_data) for grad_data in grad_inputs_data]
        return tuple(grad_inputs) + (None,)

class ReLU(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backwards(a)
        return Tensor(np.maximum(0, a.data), requires_grad=a.requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        a, = ctx.saved_tensors
        grad = grad_output.data * (a.data > 0)
        return (Tensor(grad),)


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
        return (Tensor(grad), None)


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
        return (Tensor(grad), None)


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
        return (Tensor(grad),)


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
        return (Tensor(grad),)


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
        return (Tensor(grad), None)

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

def extend_tensor_with_missing_ops():
    """Add missing operations to Tensor class"""
    Tensor.exp = lambda self: Exp.apply(self)
    Tensor.log = lambda self: Log.apply(self)
    Tensor.pow = lambda self, power: Pow.apply(self, power)
    Tensor.mean = lambda self, dim=None, keepdim=False: Mean.apply(self, dim, keepdim)
    Tensor.var = lambda self, dim=None, keepdim=False, unbiased=True: Var.apply(self, dim, keepdim, unbiased)
    Tensor.abs = lambda self: Abs.apply(self)
    Tensor.max = lambda self, dim=None, keepdim=False: Max.apply(self, dim, keepdim)

from julia.core.utils.op_registry import registry

registry.register("Exp", Exp)
registry.register("Log", Log) 
registry.register("Pow", Pow)
registry.register("Mean", Mean)
registry.register("Var", Var)
registry.register("Abs", Abs)
registry.register("Max", Max)
