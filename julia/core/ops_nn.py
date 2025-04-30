import numpy as np
from julia.core.tensor import Tensor, _ensure_tensor, Context, Function
from typing import Dict, Tuple, List, Union, Any, Optional

# Starting with Conv functions 
class Conv1DFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding):
        """
        Foward pass for Conv1DFunction

        Args:
            input: Input tensor fo shape (batch_size, in_channels, seq_len)
            weight: Weight tensor of shape (out_channel, in_channels, kernel_size)
            bias: Bias tensor of shape (out_channel) or None 
            stride: Conv stride
            padding: Conv padding 

        Returns: 
            Output tensor of shape (batch_size, out_channels, out_len)
        """
        input = _ensure_tensor(input)
        weight = _ensure_tensor(weight)
        if bias is not None: 
            bias = _ensure_tensor(bias)

        batch_size, in_channels, seq_len = input.shape
        out_channels, _, kernel_size = weight.shape

        out_len = (seq_len + 2 * padding - kernel_size) // stride + 1 

        if padding > 0: 
            input_padded = np.pad(
                    input.data, 
                    ((0,0), (0,0), (padding, padding)),
                    mode = 'constant'
            )
        else:
            input_padded = input.data 


        output_data = np.zeros((batch_size, out_channels, out_len))

        for b in range(batch_size):
            for c_out in range(out_channels):
                for c_in in range(in_channels):
                    for i in range(out_len):
                        i_start = i * stride 
                        region = input_padded[b, c_in, i_start:i_start + kernel_size]
                        output_data[b, c_out, i] += np.sum(region * weight.data[c_out, c_in, :])

        if bias is not None:
            reshaped_bias = bias.data.reshape(1, out_channels, 1)
            output_data += reshaped_bias

        ctx.save_for_backwards(input, weight, bias)
        ctx.save_data(stride=stride, padding=padding, input_shape=input.shape,
                      weight_shape=weight.shape, output_shape=output_data.shape,
                      padded_data=input_padded)
        
        return Tensor(output_data, requires_grad=input.requires_grad or weight.requires_grad)
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for Conv1D.
        
        Args:
            grad_output: Gradient of loss with respect to output
            
        Returns:
            Gradients for input, weight, bias, stride, and padding
        """
        input, weight, bias = ctx.saved_tensors
        stride = ctx.saved_data['stride']
        padding = ctx.saved_data['padding']
        input_shape = ctx.saved_data['input_shape']
        weight_shape = ctx.saved_data['weight_shape']
        output_shape = ctx.saved_data['output_shape']
        input_padded = ctx.saved_data['padded_data']
        
        batch_size, in_channels, seq_len = input_shape
        out_channels, _, kernel_size = weight_shape
        _, _, out_len = output_shape
        
        grad_input_data = np.zeros_like(input.data)
        grad_weight_data = np.zeros_like(weight.data)
        grad_bias_data = None if bias is None else np.zeros_like(bias.data)
        
        if bias is not None and bias.requires_grad:
            grad_bias_data = np.sum(grad_output.data, axis=(0, 2))
        
        if weight.requires_grad:
            for b in range(batch_size):
                for c_out in range(out_channels):
                    for c_in in range(in_channels):
                        for i in range(out_len):
                            i_start = i * stride
                            region = input_padded[b, c_in, i_start:i_start + kernel_size]
                            grad_weight_data[c_out, c_in, :] += region * grad_output.data[b, c_out, i]
        
        if input.requires_grad:
            grad_input_padded = np.zeros_like(input_padded)
            
            for b in range(batch_size):
                for c_out in range(out_channels):
                    for c_in in range(in_channels):
                        for i in range(out_len):
                            i_start = i * stride
                            for k in range(kernel_size):
                                grad_input_padded[b, c_in, i_start + k] += (
                                    weight.data[c_out, c_in, k] * grad_output.data[b, c_out, i]
                                )
                                
            if padding > 0:
                grad_input_data = grad_input_padded[:, :, padding:-padding]
            else:
                grad_input_data = grad_input_padded
        
        grad_input = Tensor(grad_input_data) if input.requires_grad else None
        grad_weight = Tensor(grad_weight_data) if weight.requires_grad else None
        grad_bias = Tensor(grad_bias_data) if bias is not None and bias.requires_grad else None
        
        return grad_input, grad_weight, grad_bias, None, None

class Conv2DFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding):
        """
        Forward pass for Conv2D.
        
        Args:
            input: Input tensor of shape (batch_size, in_channels, height, width)
            weight: Weight tensor of shape (out_channels, in_channels, kernel_h, kernel_w)
            bias: Bias tensor of shape (out_channels) or None
            stride: Convolution stride (tuple of 2 integers)
            padding: Convolution padding (tuple of 2 integers)
        
        Returns:
            Output tensor of shape (batch_size, out_channels, out_height, out_width)
        """
        input = _ensure_tensor(input)
        weight = _ensure_tensor(weight)
        if bias is not None:
            bias = _ensure_tensor(bias)
        
        batch_size, in_channels, height, width = input.shape
        out_channels, _, kernel_h, kernel_w = weight.shape
        
        stride_h, stride_w = stride
        padding_h, padding_w = padding
        
        out_height = (height + 2 * padding_h - kernel_h) // stride_h + 1
        out_width = (width + 2 * padding_w - kernel_w) // stride_w + 1
        
        if padding_h > 0 or padding_w > 0:
            input_padded = np.pad(
                input.data,
                ((0, 0), (0, 0), (padding_h, padding_h), (padding_w, padding_w)),
                mode='constant'
            )
        else:
            input_padded = input.data
        
        output_data = np.zeros((batch_size, out_channels, out_height, out_width))
        
        # Im2col approach for faster computation
        def im2col(x, h_out, w_out):
            n_samples, channels, h_in, w_in = x.shape
            
            # Initialize output matrix
            col = np.zeros((n_samples, channels, kernel_h, kernel_w, h_out, w_out))
            
            # Fill the column matrix
            for h in range(kernel_h):
                h_max = h + stride_h * h_out
                for w in range(kernel_w):
                    w_max = w + stride_w * w_out
                    col[:, :, h, w, :, :] = x[:, :, h:h_max:stride_h, w:w_max:stride_w]
                    
            # Reshape to (n_samples * h_out * w_out, channels * kernel_h * kernel_w)
            col = col.transpose(0, 4, 5, 1, 2, 3).reshape(n_samples * h_out * w_out, -1)
            return col
        
        # Use im2col for convolution
        x_col = im2col(input_padded, out_height, out_width)
        
        # Reshape weight for matrix multiplication
        weight_col = weight.data.reshape(out_channels, -1)
        
        # Compute convolution as matrix multiplication
        output_data_flat = np.matmul(x_col, weight_col.T)
        
        # Reshape output
        output_data = output_data_flat.reshape(batch_size, out_height, out_width, out_channels)
        output_data = output_data.transpose(0, 3, 1, 2)  # NHWC -> NCHW
        
        if bias is not None:
            reshaped_bias = bias.data.reshape(1, out_channels, 1, 1)
            output_data += reshaped_bias
        
        ctx.save_for_backwards(input, weight, bias)
        ctx.save_data(stride=stride, padding=padding, input_shape=input.shape,
                      weight_shape=weight.shape, output_shape=output_data.shape,
                      x_col=x_col, input_padded=input_padded)
        
        return Tensor(output_data, requires_grad=input.requires_grad or weight.requires_grad)
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for Conv2D.
        
        Args:
            grad_output: Gradient of loss with respect to output
            
        Returns:
            Gradients for input, weight, bias, stride, and padding
        """
        input, weight, bias = ctx.saved_tensors
        stride = ctx.saved_data['stride']
        padding = ctx.saved_data['padding']
        input_shape = ctx.saved_data['input_shape']
        weight_shape = ctx.saved_data['weight_shape']
        output_shape = ctx.saved_data['output_shape']
        x_col = ctx.saved_data['x_col']
        input_padded = ctx.saved_data['input_padded']
        
        batch_size, in_channels, height, width = input_shape
        out_channels, _, kernel_h, kernel_w = weight_shape
        _, _, out_height, out_width = output_shape
        
        stride_h, stride_w = stride
        padding_h, padding_w = padding
        
        grad_input_data = np.zeros_like(input.data)
        grad_weight_data = np.zeros_like(weight.data)
        grad_bias_data = None if bias is None else np.zeros_like(bias.data)
        
        if bias is not None and bias.requires_grad:
            grad_bias_data = np.sum(grad_output.data, axis=(0, 2, 3))
        
        if weight.requires_grad:
            grad_out_reshaped = grad_output.data.transpose(0, 2, 3, 1).reshape(-1, out_channels)
            
            grad_weight_data_flat = np.matmul(grad_out_reshaped.T, x_col)
            grad_weight_data = grad_weight_data_flat.reshape(weight_shape)
        
        if input.requires_grad:
            weight_reshaped = weight.data.reshape(out_channels, -1).T
            grad_out_reshaped = grad_output.data.transpose(0, 2, 3, 1).reshape(-1, out_channels)
            
            grad_col = np.matmul(grad_out_reshaped, weight_reshaped.T)
            
            grad_input_padded = np.zeros_like(input_padded)
            
            grad_col_reshaped = grad_col.reshape(batch_size, out_height, out_width, -1)
            
            # Perform col2im
            for h in range(kernel_h):
                for w in range(kernel_w):
                    for c in range(in_channels):
                        # Extract the appropriate slice of grad_col_reshaped
                        col_slice = grad_col_reshaped[:, :, :, c * kernel_h * kernel_w + h * kernel_w + w]
                        
                        # Apply to padded gradient
                        for y in range(out_height):
                            for x in range(out_width):
                                grad_input_padded[:, c, y*stride_h+h, x*stride_w+w] += col_slice[:, y, x]
            
            # Extract the gradient from padded version if padding was used
            if padding_h > 0 or padding_w > 0:
                grad_input_data = grad_input_padded[:, :, padding_h:padding_h+height, padding_w:padding_w+width]
            else:
                grad_input_data = grad_input_padded
        
        # Create Tensor objects for gradients
        grad_input = Tensor(grad_input_data) if input.requires_grad else None
        grad_weight = Tensor(grad_weight_data) if weight.requires_grad else None
        grad_bias = Tensor(grad_bias_data) if bias is not None and bias.requires_grad else None
        
        # Return gradients (None for stride and padding as they're not learnable)
        return grad_input, grad_weight, grad_bias, None, None

class Conv3DFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding):
        """
        Forward pass for Conv3D.
        
        Args:
            input: Input tensor of shape (batch_size, in_channels, depth, height, width)
            weight: Weight tensor of shape (out_channels, in_channels, kernel_d, kernel_h, kernel_w)
            bias: Bias tensor of shape (out_channels) or None
            stride: Convolution stride (tuple of 3 integers)
            padding: Convolution padding (tuple of 3 integers)
        
        Returns:
            Output tensor of shape (batch_size, out_channels, out_depth, out_height, out_width)
        """
        input = _ensure_tensor(input)
        weight = _ensure_tensor(weight)
        if bias is not None:
            bias = _ensure_tensor(bias)
        
        batch_size, in_channels, depth, height, width = input.shape
        out_channels, _, kernel_d, kernel_h, kernel_w = weight.shape
        
        stride_d, stride_h, stride_w = stride
        padding_d, padding_h, padding_w = padding
        
        out_depth = (depth + 2 * padding_d - kernel_d) // stride_d + 1
        out_height = (height + 2 * padding_h - kernel_h) // stride_h + 1
        out_width = (width + 2 * padding_w - kernel_w) // stride_w + 1
        
        if padding_d > 0 or padding_h > 0 or padding_w > 0:
            input_padded = np.pad(
                input.data,
                ((0, 0), (0, 0), (padding_d, padding_d), (padding_h, padding_h), (padding_w, padding_w)),
                mode='constant'
            )
        else:
            input_padded = input.data
        
        output_data = np.zeros((batch_size, out_channels, out_depth, out_height, out_width))
        
        for b in range(batch_size):
            for c_out in range(out_channels):
                for c_in in range(in_channels):
                    for d in range(out_depth):
                        d_start = d * stride_d
                        for h in range(out_height):
                            h_start = h * stride_h
                            for w in range(out_width):
                                w_start = w * stride_w
                                
                                region = input_padded[b, c_in, 
                                                    d_start:d_start+kernel_d, 
                                                    h_start:h_start+kernel_h, 
                                                    w_start:w_start+kernel_w]
                                output_data[b, c_out, d, h, w] += np.sum(region * weight.data[c_out, c_in])
        
        if bias is not None:
            reshaped_bias = bias.data.reshape(1, out_channels, 1, 1, 1)
            output_data += reshaped_bias
        
        ctx.save_for_backwards(input, weight, bias)
        ctx.save_data(stride=stride, padding=padding, input_shape=input.shape,
                      weight_shape=weight.shape, output_shape=output_data.shape,
                      input_padded=input_padded)
        
        return Tensor(output_data, requires_grad=input.requires_grad or weight.requires_grad)
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for Conv3D.
        
        Args:
            grad_output: Gradient of loss with respect to output
            
        Returns:
            Gradients for input, weight, bias, stride, and padding
        """
        input, weight, bias = ctx.saved_tensors
        stride = ctx.saved_data['stride']
        padding = ctx.saved_data['padding']
        input_shape = ctx.saved_data['input_shape']
        weight_shape = ctx.saved_data['weight_shape']
        output_shape = ctx.saved_data['output_shape']
        input_padded = ctx.saved_data['input_padded']
        
        batch_size, in_channels, depth, height, width = input_shape
        out_channels, _, kernel_d, kernel_h, kernel_w = weight_shape
        _, _, out_depth, out_height, out_width = output_shape
        
        stride_d, stride_h, stride_w = stride
        padding_d, padding_h, padding_w = padding
        
        grad_input_data = np.zeros_like(input.data)
        grad_weight_data = np.zeros_like(weight.data)
        grad_bias_data = None if bias is None else np.zeros_like(bias.data)
        
        if bias is not None and bias.requires_grad:
            grad_bias_data = np.sum(grad_output.data, axis=(0, 2, 3, 4))
        
        if weight.requires_grad:
            for b in range(batch_size):
                for c_out in range(out_channels):
                    for c_in in range(in_channels):
                        for d in range(out_depth):
                            d_start = d * stride_d
                            for h in range(out_height):
                                h_start = h * stride_h
                                for w in range(out_width):
                                    w_start = w * stride_w
                                    
                                    region = input_padded[b, c_in, 
                                                      d_start:d_start+kernel_d, 
                                                      h_start:h_start+kernel_h, 
                                                      w_start:w_start+kernel_w]
                                    
                                    grad_weight_data[c_out, c_in] += region * grad_output.data[b, c_out, d, h, w]
        
        if input.requires_grad:
            grad_input_padded = np.zeros_like(input_padded)
            
            for b in range(batch_size):
                for c_out in range(out_channels):
                    for c_in in range(in_channels):
                        for d in range(out_depth):
                            d_start = d * stride_d
                            for h in range(out_height):
                                h_start = h * stride_h
                                for w in range(out_width):
                                    w_start = w * stride_w
                                    
                                    grad_input_padded[b, c_in, 
                                                   d_start:d_start+kernel_d, 
                                                   h_start:h_start+kernel_h, 
                                                   w_start:w_start+kernel_w] += (
                                        weight.data[c_out, c_in] * grad_output.data[b, c_out, d, h, w]
                                    )
            
            if padding_d > 0 or padding_h > 0 or padding_w > 0:
                grad_input_data = grad_input_padded[:, :, 
                                                 padding_d:padding_d+depth, 
                                                 padding_h:padding_h+height, 
                                                 padding_w:padding_w+width]
            else:
                grad_input_data = grad_input_padded
        
        grad_input = Tensor(grad_input_data) if input.requires_grad else None
        grad_weight = Tensor(grad_weight_data) if weight.requires_grad else None
        grad_bias = Tensor(grad_bias_data) if bias is not None and bias.requires_grad else None
        
        return grad_input, grad_weight, grad_bias, None, None

class TransposedConv2DFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding, output_padding):
        """
        Forward pass for TransposedConv2D.
        
        Args:
            input: Input tensor of shape (batch_size, in_channels, height, width)
            weight: Weight tensor of shape (in_channels, out_channels, kernel_h, kernel_w)
            bias: Bias tensor of shape (out_channels) or None
            stride: Convolution stride (tuple of 2 integers)
            padding: Convolution padding (tuple of 2 integers)
            output_padding: Additional padding for output (tuple of 2 integers)
        
        Returns:
            Output tensor of shape (batch_size, out_channels, out_height, out_width)
        """
        input = _ensure_tensor(input)
        weight = _ensure_tensor(weight)
        if bias is not None:
            bias = _ensure_tensor(bias)
        
        batch_size, in_channels, height, width = input.shape
        in_channels, out_channels, kernel_h, kernel_w = weight.shape
        
        stride_h, stride_w = stride
        padding_h, padding_w = padding
        output_padding_h, output_padding_w = output_padding
        
        # Calculate output dimensions
        out_height = (height - 1) * stride_h - 2 * padding_h + kernel_h + output_padding_h
        out_width = (width - 1) * stride_w - 2 * padding_w + kernel_w + output_padding_w
        
        output_data = np.zeros((batch_size, out_channels, out_height, out_width))
        
        # Perform transposed convolution
        for b in range(batch_size):
            for c_out in range(out_channels):
                for c_in in range(in_channels):
                    for h in range(height):
                        for w in range(width):
                            # Calculate output position in transposed convolution
                            h_start = h * stride_h - padding_h
                            w_start = w * stride_w - padding_w
                            
                            # For each position in the kernel
                            for kh in range(kernel_h):
                                for kw in range(kernel_w):
                                    h_out_pos = h_start + kh
                                    w_out_pos = w_start + kw
                                    
                                    # Only update if within output bounds
                                    if (0 <= h_out_pos < out_height) and (0 <= w_out_pos < out_width):
                                        # Note weight indexing is flipped for transposed convolution
                                        output_data[b, c_out, h_out_pos, w_out_pos] += (
                                            input.data[b, c_in, h, w] * weight.data[c_in, c_out, kh, kw]
                                        )
        
        if bias is not None:
            reshaped_bias = bias.data.reshape(1, out_channels, 1, 1)
            output_data += reshaped_bias
        
        ctx.save_for_backwards(input, weight, bias)
        ctx.save_data(stride=stride, padding=padding, output_padding=output_padding,
                      input_shape=input.shape, weight_shape=weight.shape, 
                      output_shape=output_data.shape)
        
        return Tensor(output_data, requires_grad=input.requires_grad or weight.requires_grad)
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for TransposedConv2D.
        
        Args:
            grad_output: Gradient of loss with respect to output
            
        Returns:
            Gradients for input, weight, bias, stride, padding, and output_padding
        """
        input, weight, bias = ctx.saved_tensors
        stride = ctx.saved_data['stride']
        padding = ctx.saved_data['padding']
        output_padding = ctx.saved_data['output_padding']
        input_shape = ctx.saved_data['input_shape']
        weight_shape = ctx.saved_data['weight_shape']
        output_shape = ctx.saved_data['output_shape']
        
        batch_size, in_channels, height, width = input_shape
        in_channels, out_channels, kernel_h, kernel_w = weight_shape
        _, _, out_height, out_width = output_shape
        
        stride_h, stride_w = stride
        padding_h, padding_w = padding
        
        grad_input_data = np.zeros_like(input.data)
        grad_weight_data = np.zeros_like(weight.data)
        grad_bias_data = None if bias is None else np.zeros_like(bias.data)
        
        if bias is not None and bias.requires_grad:
            grad_bias_data = np.sum(grad_output.data, axis=(0, 2, 3))
        
        if weight.requires_grad:
            for b in range(batch_size):
                for c_out in range(out_channels):
                    for c_in in range(in_channels):
                        for h in range(height):
                            for w in range(width):
                                h_start = h * stride_h - padding_h
                                w_start = w * stride_w - padding_w
                                
                                for kh in range(kernel_h):
                                    for kw in range(kernel_w):
                                        h_out_pos = h_start + kh
                                        w_out_pos = w_start + kw
                                        
                                        if (0 <= h_out_pos < out_height) and (0 <= w_out_pos < out_width):
                                            grad_weight_data[c_in, c_out, kh, kw] += (
                                                input.data[b, c_in, h, w] * grad_output.data[b, c_out, h_out_pos, w_out_pos]
                                            )
        
        # Compute gradient for input - effectively a normal convolution
        if input.requires_grad:
            for b in range(batch_size):
                for c_in in range(in_channels):
                    for h in range(height):
                        for w in range(width):
                            h_start = h * stride_h - padding_h
                            w_start = w * stride_w - padding_w
                            
                            for c_out in range(out_channels):
                                for kh in range(kernel_h):
                                    for kw in range(kernel_w):
                                        h_out_pos = h_start + kh
                                        w_out_pos = w_start + kw
                                        
                                        if (0 <= h_out_pos < out_height) and (0 <= w_out_pos < out_width):
                                            grad_input_data[b, c_in, h, w] += (
                                                weight.data[c_in, c_out, kh, kw] * 
                                                grad_output.data[b, c_out, h_out_pos, w_out_pos]
                                            )
        
        grad_input = Tensor(grad_input_data) if input.requires_grad else None
        grad_weight = Tensor(grad_weight_data) if weight.requires_grad else None
        grad_bias = Tensor(grad_bias_data) if bias is not None and bias.requires_grad else None
        
        return grad_input, grad_weight, grad_bias, None, None, None

class DepthwiseSeparableConv2DFunction(Function):
    @staticmethod
    def forward(ctx, input, depthwise_weight, pointwise_weight, bias, stride, padding, depth_multiplier):
        """
        Forward pass for DepthwiseSeparableConv2D.
        
        Args:
            input: Input tensor of shape (batch_size, in_channels, height, width)
            depthwise_weight: Depthwise weight tensor of shape (depth_channels, 1, kernel_h, kernel_w)
            pointwise_weight: Pointwise weight tensor of shape (out_channels, depth_channels, 1, 1)
            bias: Bias tensor of shape (out_channels) or None
            stride: Convolution stride (tuple of 2 integers)
            padding: Convolution padding (tuple of 2 integers)
            depth_multiplier: Multiplier for the depth dimension
        
        Returns:
            Output tensor of shape (batch_size, out_channels, out_height, out_width)
        """
        input = _ensure_tensor(input)
        depthwise_weight = _ensure_tensor(depthwise_weight)
        pointwise_weight = _ensure_tensor(pointwise_weight)
        if bias is not None:
            bias = _ensure_tensor(bias)
        
        batch_size, in_channels, height, width = input.shape
        depth_channels, _, kernel_h, kernel_w = depthwise_weight.shape
        out_channels, _, _, _ = pointwise_weight.shape
        
        stride_h, stride_w = stride
        padding_h, padding_w = padding
        
        out_height = (height + 2 * padding_h - kernel_h) // stride_h + 1
        out_width = (width + 2 * padding_w - kernel_w) // stride_w + 1
        
        if padding_h > 0 or padding_w > 0:
            input_padded = np.pad(
                input.data,
                ((0, 0), (0, 0), (padding_h, padding_h), (padding_w, padding_w)),
                mode='constant'
            )
        else:
            input_padded = input.data
        
        # Initialize intermediate output for depthwise convolution
        depth_out = np.zeros((batch_size, depth_channels, out_height, out_width))
        
        for b in range(batch_size):
            for c_in in range(in_channels):
                for c_depth in range(depth_multiplier):
                    c_out = c_in * depth_multiplier + c_depth
                    for h in range(out_height):
                        h_start = h * stride_h
                        for w in range(out_width):
                            w_start = w * stride_w
                            
                            region = input_padded[b, c_in, h_start:h_start+kernel_h, w_start:w_start+kernel_w]
                            
                            # Apply depthwise filter
                            depth_out[b, c_out, h, w] = np.sum(region * depthwise_weight.data[c_out, 0])
        
        # Perform pointwise convolution (1x1 convolution to mix channels)
        output_data = np.zeros((batch_size, out_channels, out_height, out_width))
        
        for b in range(batch_size):
            for h in range(out_height):
                for w in range(out_width):
                    # Extract the depth channels at this position
                    depth_channels_data = depth_out[b, :, h, w]
                    
                    # For each output channel
                    for c_out in range(out_channels):
                        # Apply 1x1 convolution
                        output_data[b, c_out, h, w] = np.sum(
                            depth_channels_data * pointwise_weight.data[c_out, :, 0, 0]
                        )
        
        if bias is not None:
            reshaped_bias = bias.data.reshape(1, out_channels, 1, 1)
            output_data += reshaped_bias
        
        ctx.save_for_backwards(input, depthwise_weight, pointwise_weight, bias)
        ctx.save_data(stride=stride, padding=padding, depth_multiplier=depth_multiplier,
                      input_shape=input.shape, depthwise_weight_shape=depthwise_weight.shape,
                      pointwise_weight_shape=pointwise_weight.shape, output_shape=output_data.shape,
                      depth_out=depth_out, input_padded=input_padded)
        
        return Tensor(output_data, requires_grad=input.requires_grad or depthwise_weight.requires_grad or pointwise_weight.requires_grad)
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for DepthwiseSeparableConv2D.
        
        Args:
            grad_output: Gradient of loss with respect to output
            
        Returns:
            Gradients for input, depthwise_weight, pointwise_weight, bias, stride, padding, and depth_multiplier
        """
        input, depthwise_weight, pointwise_weight, bias = ctx.saved_tensors
        stride = ctx.saved_data['stride']
        padding = ctx.saved_data['padding']
        depth_multiplier = ctx.saved_data['depth_multiplier']
        input_shape = ctx.saved_data['input_shape']
        depthwise_weight_shape = ctx.saved_data['depthwise_weight_shape']
        pointwise_weight_shape = ctx.saved_data['pointwise_weight_shape']
        output_shape = ctx.saved_data['output_shape']
        depth_out = ctx.saved_data['depth_out']
        input_padded = ctx.saved_data['input_padded']
        
        batch_size, in_channels, height, width = input_shape
        depth_channels, _, kernel_h, kernel_w = depthwise_weight_shape
        out_channels, _, _, _ = pointwise_weight_shape
        _, _, out_height, out_width = output_shape
        
        stride_h, stride_w = stride
        padding_h, padding_w = padding
        
        grad_input_data = np.zeros_like(input.data)
        grad_depthwise_weight_data = np.zeros_like(depthwise_weight.data)
        grad_pointwise_weight_data = np.zeros_like(pointwise_weight.data)
        grad_bias_data = None if bias is None else np.zeros_like(bias.data)
        
        if bias is not None and bias.requires_grad:
            grad_bias_data = np.sum(grad_output.data, axis=(0, 2, 3))
        
        # First, compute gradient for pointwise convolution
        grad_depth_out = np.zeros_like(depth_out)
        
        if pointwise_weight.requires_grad:
            for b in range(batch_size):
                for h in range(out_height):
                    for w in range(out_width):
                        depth_channels_data = depth_out[b, :, h, w]
                        
                        for c_out in range(out_channels):
                            # Gradient for pointwise weights
                            grad_pointwise_weight_data[c_out, :, 0, 0] += (
                                depth_channels_data * grad_output.data[b, c_out, h, w]
                            )
                            
                            # Gradient for depth output
                            grad_depth_out[b, :, h, w] += (
                                pointwise_weight.data[c_out, :, 0, 0] * grad_output.data[b, c_out, h, w]
                            )
        
        # Compute gradient for depthwise convolution
        if depthwise_weight.requires_grad or input.requires_grad:
            grad_input_padded = np.zeros_like(input_padded)
            
            for b in range(batch_size):
                for c_in in range(in_channels):
                    for c_depth in range(depth_multiplier):
                        c_out = c_in * depth_multiplier + c_depth
                        
                        if depthwise_weight.requires_grad:
                            for h in range(out_height):
                                h_start = h * stride_h
                                for w in range(out_width):
                                    w_start = w * stride_w
                                    
                                    # Extract region for gradient computation
                                    region = input_padded[b, c_in, h_start:h_start+kernel_h, w_start:w_start+kernel_w]
                                    
                                    # Gradient for depthwise weights
                                    grad_depthwise_weight_data[c_out, 0] += (
                                        region * grad_depth_out[b, c_out, h, w]
                                    )
                        
                        if input.requires_grad:
                            for h in range(out_height):
                                h_start = h * stride_h
                                for w in range(out_width):
                                    w_start = w * stride_w
                                    
                                    # Gradient for input
                                    grad_input_padded[b, c_in, h_start:h_start+kernel_h, w_start:w_start+kernel_w] += (
                                        depthwise_weight.data[c_out, 0] * grad_depth_out[b, c_out, h, w]
                                    )
            
            if padding_h > 0 or padding_w > 0:
                grad_input_data = grad_input_padded[:, :, padding_h:padding_h+height, padding_w:padding_w+width]
            else:
                grad_input_data = grad_input_padded
        
        grad_input = Tensor(grad_input_data) if input.requires_grad else None
        grad_depthwise_weight = Tensor(grad_depthwise_weight_data) if depthwise_weight.requires_grad else None
        grad_pointwise_weight = Tensor(grad_pointwise_weight_data) if pointwise_weight.requires_grad else None
        grad_bias = Tensor(grad_bias_data) if bias is not None and bias.requires_grad else None
        
        return grad_input, grad_depthwise_weight, grad_pointwise_weight, grad_bias, None, None, None

class DilatedConv2DFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding, dilation):
        """
        Forward pass for DilatedConv2D.
        
        Args:
            input: Input tensor of shape (batch_size, in_channels, height, width)
            weight: Weight tensor of shape (out_channels, in_channels, kernel_h, kernel_w)
            bias: Bias tensor of shape (out_channels) or None
            stride: Convolution stride (tuple of 2 integers)
            padding: Convolution padding (tuple of 2 integers)
            dilation: Dilation factor (tuple of 2 integers)
        
        Returns:
            Output tensor of shape (batch_size, out_channels, out_height, out_width)
        """
        input = _ensure_tensor(input)
        weight = _ensure_tensor(weight)
        if bias is not None:
            bias = _ensure_tensor(bias)
        
        batch_size, in_channels, height, width = input.shape
        out_channels, _, kernel_h, kernel_w = weight.shape
        
        stride_h, stride_w = stride
        padding_h, padding_w = padding
        dilation_h, dilation_w = dilation
        
        out_height = (height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
        out_width = (width + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1
        
        if padding_h > 0 or padding_w > 0:
            input_padded = np.pad(
                input.data,
                ((0, 0), (0, 0), (padding_h, padding_h), (padding_w, padding_w)),
                mode='constant'
            )
        else:
            input_padded = input.data
        
        output_data = np.zeros((batch_size, out_channels, out_height, out_width))
        
        for b in range(batch_size):
            for c_out in range(out_channels):
                for h in range(out_height):
                    h_start = h * stride_h
                    for w in range(out_width):
                        w_start = w * stride_w
                        
                        val = 0.0
                        
                        for c_in in range(in_channels):
                            for kh in range(kernel_h):
                                h_offset = kh * dilation_h
                                
                                for kw in range(kernel_w):
                                    w_offset = kw * dilation_w
                                    
                                    h_pos = h_start + h_offset
                                    w_pos = w_start + w_offset
                                    
                                    if h_pos < input_padded.shape[2] and w_pos < input_padded.shape[3]:
                                        input_val = input_padded[b, c_in, h_pos, w_pos]
                                        kernel_val = weight.data[c_out, c_in, kh, kw]
                                        val += input_val * kernel_val
                        
                        output_data[b, c_out, h, w] = val
        
        if bias is not None:
            reshaped_bias = bias.data.reshape(1, out_channels, 1, 1)
            output_data += reshaped_bias
        
        ctx.save_for_backwards(input, weight, bias)
        ctx.save_data(stride=stride, padding=padding, dilation=dilation,
                      input_shape=input.shape, weight_shape=weight.shape,
                      output_shape=output_data.shape, input_padded=input_padded)
        
        return Tensor(output_data, requires_grad=input.requires_grad or weight.requires_grad)
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for DilatedConv2D.
        
        Args:
            grad_output: Gradient of loss with respect to output
            
        Returns:
            Gradients for input, weight, bias, stride, padding, and dilation
        """
        input, weight, bias = ctx.saved_tensors
        stride = ctx.saved_data['stride']
        padding = ctx.saved_data['padding']
        dilation = ctx.saved_data['dilation']
        input_shape = ctx.saved_data['input_shape']
        weight_shape = ctx.saved_data['weight_shape']
        output_shape = ctx.saved_data['output_shape']
        input_padded = ctx.saved_data['input_padded']
        
        batch_size, in_channels, height, width = input_shape
        out_channels, _, kernel_h, kernel_w = weight_shape
        _, _, out_height, out_width = output_shape
        
        stride_h, stride_w = stride
        padding_h, padding_w = padding
        dilation_h, dilation_w = dilation
        
        grad_input_data = np.zeros_like(input.data)
        grad_weight_data = np.zeros_like(weight.data)
        grad_bias_data = None if bias is None else np.zeros_like(bias.data)
        
        if bias is not None and bias.requires_grad:
            # Sum gradient over batch and spatial dimensions
            grad_bias_data = np.sum(grad_output.data, axis=(0, 2, 3))
        
        # Compute gradient for weights
        if weight.requires_grad:
            for b in range(batch_size):
                for c_out in range(out_channels):
                    for c_in in range(in_channels):
                        for h in range(out_height):
                            h_start = h * stride_h
                            for w in range(out_width):
                                w_start = w * stride_w
                                
                                # For each kernel position
                                for kh in range(kernel_h):
                                    h_offset = kh * dilation_h
                                    for kw in range(kernel_w):
                                        w_offset = kw * dilation_w
                                        
                                        # Get input value at dilated position
                                        h_pos = h_start + h_offset
                                        w_pos = w_start + w_offset
                                        
                                        if h_pos < input_padded.shape[2] and w_pos < input_padded.shape[3]:
                                            grad_weight_data[c_out, c_in, kh, kw] += (
                                                input_padded[b, c_in, h_pos, w_pos] * 
                                                grad_output.data[b, c_out, h, w]
                                            )
        
        # Compute gradient for input
        if input.requires_grad:
            grad_input_padded = np.zeros_like(input_padded)
            
            for b in range(batch_size):
                for c_out in range(out_channels):
                    for c_in in range(in_channels):
                        for h in range(out_height):
                            h_start = h * stride_h
                            for w in range(out_width):
                                w_start = w * stride_w
                                
                                for kh in range(kernel_h):
                                    h_offset = kh * dilation_h
                                    for kw in range(kernel_w):
                                        w_offset = kw * dilation_w
                                        
                                        h_pos = h_start + h_offset
                                        w_pos = w_start + w_offset
                                        
                                        if h_pos < input_padded.shape[2] and w_pos < input_padded.shape[3]:
                                            grad_input_padded[b, c_in, h_pos, w_pos] += (
                                                weight.data[c_out, c_in, kh, kw] * 
                                                grad_output.data[b, c_out, h, w]
                                            )
            
            if padding_h > 0 or padding_w > 0:
                grad_input_data = grad_input_padded[:, :, padding_h:padding_h+height, padding_w:padding_w+width]
            else:
                grad_input_data = grad_input_padded
        
        grad_input = Tensor(grad_input_data) if input.requires_grad else None
        grad_weight = Tensor(grad_weight_data) if weight.requires_grad else None
        grad_bias = Tensor(grad_bias_data) if bias is not None and bias.requires_grad else None
        
        return grad_input, grad_weight, grad_bias, None, None, None
