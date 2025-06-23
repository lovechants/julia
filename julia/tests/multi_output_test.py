import numpy as np
from julia.core.tensor import Tensor, Function, Context

class Split(Function):
    """
    Example multi-output function that splits a tensor along an axis
    """
    @staticmethod
    def forward(ctx, input_tensor, split_size_or_sections, axis=0):
        """
        Split tensor along axis
        Args:
            input_tensor: Input tensor to split
            split_size_or_sections: Size of each split or list of section sizes
            axis: Axis along which to split
        """
        ctx.save_data(axis=axis, input_shape=input_tensor.shape)
        
        if isinstance(split_size_or_sections, int):
            # Split into equal sections of given size
            sections = []
            start = 0
            size = split_size_or_sections
            while start < input_tensor.shape[axis]:
                end = min(start + size, input_tensor.shape[axis])
                sections.append((start, end))
                start = end
        else:
            # Split according to given section sizes
            sections = []
            start = 0
            for size in split_size_or_sections:
                sections.append((start, start + size))
                start += size
        
        ctx.save_data(sections=sections)
        
        # Create output tensors
        outputs = []
        for start, end in sections:
            # Create slice along axis
            slice_obj = [slice(None)] * input_tensor.data.ndim
            slice_obj[axis] = slice(start, end)
            slice_data = input_tensor.data[tuple(slice_obj)]
            outputs.append(Tensor(slice_data.copy()))
        
        return tuple(outputs)
    
    @staticmethod
    def backward(ctx, *grad_outputs):
        """
        Backward pass for split - concatenate gradients back together
        """
        axis = ctx.saved_data['axis']
        input_shape = ctx.saved_data['input_shape']
        sections = ctx.saved_data['sections']
        
        # Initialize gradient for input
        grad_input = np.zeros(input_shape)
        
        # Fill in gradients from each output
        for i, (grad_output, (start, end)) in enumerate(zip(grad_outputs, sections)):
            if grad_output is not None:
                slice_obj = [slice(None)] * len(input_shape)
                slice_obj[axis] = slice(start, end)
                grad_input[tuple(slice_obj)] = grad_output.data
        
        return Tensor(grad_input)


class DivMod(Function):
    """
    Another multi-output function that returns quotient and remainder
    """
    @staticmethod
    def forward(ctx, dividend, divisor):
        """
        Compute quotient and remainder
        Args:
            dividend: Dividend tensor
            divisor: Divisor tensor
        """
        ctx.save_for_backwards(dividend, divisor)
        
        quotient_data = np.floor_divide(dividend.data, divisor.data)
        remainder_data = np.remainder(dividend.data, divisor.data)
        
        quotient = Tensor(quotient_data)
        remainder = Tensor(remainder_data)
        
        return quotient, remainder
    
    @staticmethod
    def backward(ctx, grad_quotient, grad_remainder):
        """
        Backward pass for divmod
        """
        dividend, divisor = ctx.saved_tensors
        
        # Gradients for quotient part (floor division)
        # d(floor(a/b))/da ≈ 1/b (approximation since floor is not differentiable)
        # d(floor(a/b))/db ≈ -a/b²
        grad_dividend_from_quotient = None
        grad_divisor_from_quotient = None
        
        if grad_quotient is not None:
            grad_dividend_from_quotient = grad_quotient.data / divisor.data
            grad_divisor_from_quotient = -grad_quotient.data * dividend.data / (divisor.data ** 2)
        
        # Gradients for remainder part
        # d(a mod b)/da = 1
        # d(a mod b)/db = -floor(a/b)
        grad_dividend_from_remainder = None
        grad_divisor_from_remainder = None
        
        if grad_remainder is not None:
            grad_dividend_from_remainder = grad_remainder.data
            grad_divisor_from_remainder = -grad_remainder.data * np.floor_divide(dividend.data, divisor.data)
        
        # Combine gradients
        grad_dividend_total = np.zeros_like(dividend.data)
        grad_divisor_total = np.zeros_like(divisor.data)
        
        if grad_dividend_from_quotient is not None:
            grad_dividend_total += grad_dividend_from_quotient
        if grad_dividend_from_remainder is not None:
            grad_dividend_total += grad_dividend_from_remainder
            
        if grad_divisor_from_quotient is not None:
            grad_divisor_total += grad_divisor_from_quotient
        if grad_divisor_from_remainder is not None:
            grad_divisor_total += grad_divisor_from_remainder
        
        return Tensor(grad_dividend_total), Tensor(grad_divisor_total)


# Test the multi-output functions
def test_multi_output_autograd():
    """Test that multi-output functions work correctly with autograd"""
    
    print("Testing Split function:")
    
    # Test Split function
    x = Tensor(np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32), requires_grad=True)
    print(f"Input tensor: {x.data}")
    
    # Split into 2 parts along axis 1
    out1, out2 = Split.apply(x, 2, axis=1)
    print(f"Split output 1: {out1.data}")
    print(f"Split output 2: {out2.data}")
    
    # Compute a loss using both outputs
    loss = (out1.sum() + out2.sum() * 2)
    print(f"Loss: {loss.data}")
    
    # Backward pass
    loss.backward()
    print(f"Gradient of input: {x.grad.data}")
    
    print("\nTesting DivMod function:")
    
    # Test DivMod function
    a = Tensor(np.array([10, 15, 7], dtype=np.float32), requires_grad=True)
    b = Tensor(np.array([3, 4, 2], dtype=np.float32), requires_grad=True)
    
    quotient, remainder = DivMod.apply(a, b)
    print(f"Quotient: {quotient.data}")
    print(f"Remainder: {remainder.data}")
    
    # Compute loss using both outputs
    loss2 = quotient.sum() + remainder.sum()
    print(f"Loss: {loss2.data}")
    
    # Backward pass
    loss2.backward()
    print(f"Gradient of a: {a.grad.data}")
    print(f"Gradient of b: {b.grad.data}")

if __name__ == "__main__":
    test_multi_output_autograd()
