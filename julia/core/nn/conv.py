import numpy as np 
from julia.core.tensor import Tensor, Function, _ensure_tensor 
from julia.core.nn.layers import Layer
from typing import Tuple, Union, Dict, List

class Conv1D(Layer):
    """
    1D Convolution: SIGNALS 
    args: 
        in_channels: # input channels 
        out_channels: # output channels 
        kernel_size: Size of conv kernel 
        stride: Stride of conv
        padding: Padding added to input 
        bias: To include a bias term 
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride 
        self.padding = padding 

        # Initialize using Kaiming/He initialization 
        # https://paperswithcode.com/method/he-initialization

        scale = np.sqrt(2.0 / (in_channels * kernel_size))
        weight_shape = (out_channels, in_channels, kernel_size)
        self.weight = Tensor(np.random.randn(*weight_shape) * scale, requires_grad=True)
        self.parameters.append(self.weight)

        if bias: 
            self.bias = Tensor(np.zeros(out_channels), requires_grad=True)
            self.parameters.append(self.bias)

        else:
            self.bias = None 

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass -> apply the 1D conv """
        
        x = _ensure_tensor(x)
        batch_size, in_channels, seq_len = x.shape

        out_len = (seq_len + 2 * self.padding - self.kernel_size) // self.stride + 1 

        if self.padding > 0: 
            x_padded = np.pad(
                            x.data,
                            ((0,0), (0,0), (self.padding, self.padding)), 
                            mode='constant'
                        )
        else:
            x_padded = x.data

        # Init output 
        out_data = np.zeros((batch_size, self.out_channels, out_len))
        
        # Conv 
        for b in range(batch_size):
            for c_out in range(self.out_channels):
                for c_in in range(self.in_channels):
                    for i in range(out_len):
                        """ 
                        Calculate starting positon 
                        Extract region to convolve with the kernel 
                        Multiply w/ kernel and add to output 
                        """
                        i_start = i * self.stride
                        region = x_padded[b, c_in, i_start:i_start + self.kernel_size]
                        out_data[b, c_out, i] += np.sum(region * self.weight.data[c_out, c_in, :])

        if self.bias is not None:
            # Reshape the bias 
            reshaped_bias = self.bias.data.reshape(1, self.out_channels, 1)
            out_data += reshaped_bias

        return Tensor(out_data, requires_grad=x.requires_grad)


class Conv2D(Layer):
    """
    2D convolution layer
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of convolutional kernel (int or tuple)
        stride: Stride of convolution (int or tuple)
        padding: Padding added to input (int or tuple)
        bias: Whether to include a bias term
    """
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: Union[int, Tuple[int, int]], 
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0, 
                 bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Handle scalar or tuple kernel size
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
            
        # Handle scalar or tuple stride
        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride
            
        # Handle scalar or tuple padding
        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding
        
        # Initialize weights using Kaiming/He initialization
        scale = np.sqrt(2.0 / (in_channels * self.kernel_size[0] * self.kernel_size[1]))
        weight_shape = (out_channels, in_channels, self.kernel_size[0], self.kernel_size[1])
        self.weight = Tensor(np.random.randn(*weight_shape) * scale, requires_grad=True)
        self.parameters.append(self.weight)
        
        if bias:
            self.bias = Tensor(np.zeros(out_channels), requires_grad=True)
            self.parameters.append(self.bias)
        else:
            self.bias = None
    
    def _im2col(self, x_data, h_out, w_out):
        """Convert image data to column format for convolution"""
        n_samples, in_channels, h_in, w_in = x_data.shape
        k_h, k_w = self.kernel_size
        s_h, s_w = self.stride
        p_h, p_w = self.padding
        
        # Pad input if needed
        if p_h > 0 or p_w > 0:
            x_padded = np.pad(
                x_data, 
                ((0, 0), (0, 0), (p_h, p_h), (p_w, p_w)), 
                mode='constant'
            )
        else:
            x_padded = x_data
            
        # Initialize output matrix
        col = np.zeros((n_samples, in_channels, k_h, k_w, h_out, w_out))
        
        # Fill the column matrix
        for h in range(k_h):
            h_max = h + s_h * h_out
            for w in range(k_w):
                w_max = w + s_w * w_out
                col[:, :, h, w, :, :] = x_padded[:, :, h:h_max:s_h, w:w_max:s_w]
                
        # Reshape to (n_samples * h_out * w_out, in_channels * k_h * k_w)
        col = col.transpose(0, 4, 5, 1, 2, 3).reshape(n_samples * h_out * w_out, -1)
        return col
            
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass: apply 2D convolution"""
        x = _ensure_tensor(x)
        
        # Get input dimensions (n_samples, in_channels, height, width)
        n_samples, in_channels, h_in, w_in = x.shape
        
        # Calculate output dimensions
        h_out = (h_in + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        w_out = (w_in + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1
        
        # Reshape weight for matrix multiplication
        # (out_channels, in_channels, k_h, k_w) -> (out_channels, in_channels * k_h * k_w)
        weight_col = self.weight.data.reshape(self.out_channels, -1)
        
        # Convert input to column format using im2col
        x_col = self._im2col(x.data, h_out, w_out)
        
        # Compute convolution as matrix multiplication
        out_data = np.matmul(x_col, weight_col.T)
        
        # Reshape output
        out_data = out_data.reshape(n_samples, h_out, w_out, self.out_channels)
        out_data = out_data.transpose(0, 3, 1, 2)  # NHWC -> NCHW
        
        # Create output tensor
        out = Tensor(out_data, requires_grad=x.requires_grad)
        
        if self.bias is not None:
            # Add bias to each feature map
            # Reshape bias for broadcasting: (out_channels) -> (1, out_channels, 1, 1)
            bias_reshaped = self.bias.data.reshape(1, self.out_channels, 1, 1)
            out.data += bias_reshaped
            
        return out

class Conv3D(Layer):
    """3D Conv"""
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: Union[int, Tuple[int, int, int]], 
                 stride: Union[int, Tuple[int, int, int]] = 1,
                 padding: Union[int, Tuple[int, int, int]] = 0, 
                 bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Handle scalar or tuple kernel size
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
            
        # Handle scalar or tuple stride
        if isinstance(stride, int):
            self.stride = (stride, stride, stride)
        else:
            self.stride = stride
            
        # Handle scalar or tuple padding
        if isinstance(padding, int):
            self.padding = (padding, padding, padding)
        else:
            self.padding = padding
        
        # Initialize weights using Kaiming/He initialization
        scale = np.sqrt(2.0 / (in_channels * np.prod(self.kernel_size)))
        weight_shape = (out_channels, in_channels) + self.kernel_size
        self.weight = Tensor(np.random.randn(*weight_shape) * scale, requires_grad=True)
        self.parameters.append(self.weight)
        
        if bias:
            self.bias = Tensor(np.zeros(out_channels), requires_grad=True)
            self.parameters.append(self.bias)
        else:
            self.bias = None
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass: apply 3D convolution"""
        x = _ensure_tensor(x)
        
        # Get input dimensions (batch_size, in_channels, depth, height, width)
        batch_size, in_channels, d_in, h_in, w_in = x.shape
        
        # Calculate output dimensions
        d_out = (d_in + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        h_out = (h_in + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1
        w_out = (w_in + 2 * self.padding[2] - self.kernel_size[2]) // self.stride[2] + 1
        
        # Pad input if needed
        if any(p > 0 for p in self.padding):
            x_padded = np.pad(
                x.data, 
                ((0, 0), (0, 0), 
                 (self.padding[0], self.padding[0]), 
                 (self.padding[1], self.padding[1]), 
                 (self.padding[2], self.padding[2])), 
                mode='constant'
            )
        else:
            x_padded = x.data
            
        # Initialize output
        out_data = np.zeros((batch_size, self.out_channels, d_out, h_out, w_out))
        
        # Get kernel dimensions
        k_d, k_h, k_w = self.kernel_size
        s_d, s_h, s_w = self.stride
        
        # Perform convolution
        for b in range(batch_size):
            for c_out in range(self.out_channels):
                for d in range(d_out):
                    d_start = d * s_d
                    for h in range(h_out):
                        h_start = h * s_h
                        for w in range(w_out):
                            w_start = w * s_w
                            
                            # Initialize accumulator for this output position
                            val = 0.0
                            
                            # Convolve with kernel
                            for c_in in range(in_channels):
                                # Extract region to convolve
                                region = x_padded[b, c_in, 
                                                 d_start:d_start + k_d, 
                                                 h_start:h_start + k_h, 
                                                 w_start:w_start + k_w]
                                
                                # Multiply with kernel and add to accumulator
                                val += np.sum(region * self.weight.data[c_out, c_in])
                            
                            # Store result in output
                            out_data[b, c_out, d, h, w] = val
        
        # Add bias if needed
        if self.bias is not None:
            # Reshape bias for broadcasting: (out_channels) -> (1, out_channels, 1, 1, 1)
            bias_reshaped = self.bias.data.reshape(1, self.out_channels, 1, 1, 1)
            out_data += bias_reshaped
            
        return Tensor(out_data, requires_grad=x.requires_grad)

class TransposedConv2D(Layer):
    """2D transposed conv layer | deconvolution"""

    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: Union[int, Tuple[int, int]], 
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0,
                 output_padding: Union[int, Tuple[int, int]] = 0,
                 bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Handle scalar or tuple parameters
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
            
        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride
            
        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding
            
        if isinstance(output_padding, int):
            self.output_padding = (output_padding, output_padding)

        else:
            self.output_padding = output_padding

        scale = np.sqrt(2.0 / (in_channels * np.prod(self.kernel_size)))
        weight_shape = (in_channels, out_channels) + self.kernel_size
        self.weight = Tensor(np.random.randn(*weight_shape) * scale, requires_grad=True)
        self.parameters.append(self.weight)
        
        if bias:
            self.bias = Tensor(np.zeros(out_channels), requires_grad=True)
            self.parameters.append(self.bias)
        else:
            self.bias = None
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass: apply 2D transposed convolution"""
        x = _ensure_tensor(x)
        
        # Get input dimensions (batch_size, in_channels, height, width)
        batch_size, in_channels, h_in, w_in = x.shape
        
        # Calculate output dimensions
        h_out = (h_in - 1) * self.stride[0] - 2 * self.padding[0] + self.kernel_size[0] + self.output_padding[0]
        w_out = (w_in - 1) * self.stride[1] - 2 * self.padding[1] + self.kernel_size[1] + self.output_padding[1]
        
        # Initialize output
        out_data = np.zeros((batch_size, self.out_channels, h_out, w_out))
        
        # Get kernel dimensions
        k_h, k_w = self.kernel_size
        s_h, s_w = self.stride
        p_h, p_w = self.padding
        
        # Perform transposed convolution
        for b in range(batch_size):
            for c_out in range(self.out_channels):
                for c_in in range(in_channels):
                    for h in range(h_in):
                        for w in range(w_in):
                            # Calculate output position in transposed convolution
                            h_start = h * s_h - p_h
                            w_start = w * s_w - p_w
                            
                            # For each position in the kernel
                            for kh in range(k_h):
                                for kw in range(k_w):
                                    h_out_pos = h_start + kh
                                    w_out_pos = w_start + kw
                                    
                                    # Only update if within output bounds
                                    if (0 <= h_out_pos < h_out) and (0 <= w_out_pos < w_out):
                                        # Note weight indexing is flipped for transposed convolution
                                        out_data[b, c_out, h_out_pos, w_out_pos] += \
                                            x.data[b, c_in, h, w] * self.weight.data[c_in, c_out, kh, kw]
        
        # Add bias if needed
        if self.bias is not None:
            # Reshape bias for broadcasting
            bias_reshaped = self.bias.data.reshape(1, self.out_channels, 1, 1)
            out_data += bias_reshaped
            
        return Tensor(out_data, requires_grad=x.requires_grad)

class DepthwiseSeparableConv2D(Layer):
    """
    Depthwise Separable 2D Convolution
    
    Consists of a depthwise convolution (one filter per input channel)
    followed by a pointwise convolution (1x1 convolution to mix channels)
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of depthwise convolutional kernel
        stride: Stride of depthwise convolution
        padding: Padding for depthwise convolution
        depth_multiplier: Multiplier for the depth dimension (default: 1)
        bias: Whether to include bias terms
    """
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0,
                 depth_multiplier: int = 1,
                 bias: bool = True):
        super().__init__()
        
        # Handle scalar or tuple kernel size
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
            
        # Handle scalar or tuple stride
        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride
            
        # Handle scalar or tuple padding
        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding
            
        self.depth_channels = in_channels * depth_multiplier
        
        # Depthwise convolution weights - separate filter for each input channel
        depthwise_shape = (self.depth_channels, 1) + self.kernel_size
        scale_depthwise = np.sqrt(2.0 / (np.prod(self.kernel_size)))
        self.depthwise_weight = Tensor(np.random.randn(*depthwise_shape) * scale_depthwise, requires_grad=True)
        self.parameters.append(self.depthwise_weight)
        
        # Pointwise convolution weights - 1x1 convolution to mix channels
        pointwise_shape = (out_channels, self.depth_channels, 1, 1)
        scale_pointwise = np.sqrt(2.0 / self.depth_channels)
        self.pointwise_weight = Tensor(np.random.randn(*pointwise_shape) * scale_pointwise, requires_grad=True)
        self.parameters.append(self.pointwise_weight)
        
        if bias:
            self.bias = Tensor(np.zeros(out_channels), requires_grad=True)
            self.parameters.append(self.bias)
        else:
            self.bias = None
            
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass: apply depthwise separable convolution"""
        x = _ensure_tensor(x)
        
        # Get input dimensions
        batch_size, in_channels, h_in, w_in = x.shape
        
        # Calculate output dimensions for depthwise convolution
        h_out = (h_in + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        w_out = (w_in + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1
        
        # Pad input if needed
        if self.padding[0] > 0 or self.padding[1] > 0:
            x_padded = np.pad(
                x.data,
                ((0, 0), (0, 0), (self.padding[0], self.padding[0]), (self.padding[1], self.padding[1])),
                mode='constant'
            )
        else:
            x_padded = x.data
            
        # Initialize intermediate output for depthwise convolution
        depth_out = np.zeros((batch_size, self.depth_channels, h_out, w_out))
        
        # Get kernel dimensions
        k_h, k_w = self.kernel_size
        s_h, s_w = self.stride
        
        # Perform depthwise convolution
        for b in range(batch_size):
            for c_in in range(in_channels):
                for c_depth in range(self.depth_channels // in_channels):
                    c_out = c_in * (self.depth_channels // in_channels) + c_depth
                    for h in range(h_out):
                        h_start = h * s_h
                        for w in range(w_out):
                            w_start = w * s_w
                            
                            # Extract region for convolution
                            region = x_padded[b, c_in, h_start:h_start + k_h, w_start:w_start + k_w]
                            
                            # Apply depthwise filter
                            depth_out[b, c_out, h, w] = np.sum(region * self.depthwise_weight.data[c_out, 0])
        
        # Perform pointwise convolution (1x1 convolution to mix channels)
        # Initialize output
        out_data = np.zeros((batch_size, self.pointwise_weight.shape[0], h_out, w_out))
        
        # For each output position
        for b in range(batch_size):
            for h in range(h_out):
                for w in range(w_out):
                    # Extract the depth channels at this position
                    depth_channels = depth_out[b, :, h, w]
                    
                    # For each output channel
                    for c_out in range(self.pointwise_weight.shape[0]):
                        # Apply 1x1 convolution
                        out_data[b, c_out, h, w] = np.sum(
                            depth_channels * self.pointwise_weight.data[c_out, :, 0, 0]
                        )
        
        # Add bias if needed
        if self.bias is not None:
            # Reshape bias for broadcasting
            bias_reshaped = self.bias.data.reshape(1, -1, 1, 1)
            out_data += bias_reshaped
            
        return Tensor(out_data, requires_grad=x.requires_grad)


class DilatedConv2D(Layer):
    """
    2D Dilated Convolution Layer
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of convolutional kernel
        stride: Stride of convolution
        padding: Padding added to input
        dilation: Dilation factor
        bias: Whether to include a bias term
    """
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Handle scalar or tuple parameters
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
            
        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride
            
        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding
            
        if isinstance(dilation, int):
            self.dilation = (dilation, dilation)
        else:
            self.dilation = dilation
            
        # Initialize weights
        scale = np.sqrt(2.0 / (in_channels * np.prod(self.kernel_size)))
        weight_shape = (out_channels, in_channels) + self.kernel_size
        self.weight = Tensor(np.random.randn(*weight_shape) * scale, requires_grad=True)
        self.parameters.append(self.weight)
        
        if bias:
            self.bias = Tensor(np.zeros(out_channels), requires_grad=True)
            self.parameters.append(self.bias)
        else:
            self.bias = None
            
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass: apply dilated convolution"""
        x = _ensure_tensor(x)
        
        # Get input dimensions
        batch_size, in_channels, h_in, w_in = x.shape
        
        # Calculate output dimensions
        h_out = (h_in + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1
        w_out = (w_in + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[1] + 1
        
        # Pad input if needed
        if self.padding[0] > 0 or self.padding[1] > 0:
            x_padded = np.pad(
                x.data,
                ((0, 0), (0, 0), (self.padding[0], self.padding[0]), (self.padding[1], self.padding[1])),
                mode='constant'
            )
        else:
            x_padded = x.data
            
        # Initialize output
        out_data = np.zeros((batch_size, self.out_channels, h_out, w_out))
        
        # Get parameters
        k_h, k_w = self.kernel_size
        s_h, s_w = self.stride
        d_h, d_w = self.dilation
        
        # Perform dilated convolution
        for b in range(batch_size):
            for c_out in range(self.out_channels):
                for h in range(h_out):
                    h_start = h * s_h
                    for w in range(w_out):
                        w_start = w * s_w
                        
                        # Initialize accumulator for this output position
                        val = 0.0
                        
                        # For each input channel
                        for c_in in range(in_channels):
                            # For each kernel position
                            for kh in range(k_h):
                                # Apply dilation to kernel position
                                h_offset = kh * d_h
                                
                                for kw in range(k_w):
                                    # Apply dilation to kernel position
                                    w_offset = kw * d_w
                                    
                                    # Get input value at dilated position
                                    h_pos = h_start + h_offset
                                    w_pos = w_start + w_offset
                                    
                                    if h_pos < x_padded.shape[2] and w_pos < x_padded.shape[3]:
                                        input_val = x_padded[b, c_in, h_pos, w_pos]
                                        kernel_val = self.weight.data[c_out, c_in, kh, kw]
                                        val += input_val * kernel_val
                        
                        # Store result
                        out_data[b, c_out, h, w] = val
        
        # Add bias if needed
        if self.bias is not None:
            # Reshape bias for broadcasting
            bias_reshaped = self.bias.data.reshape(1, -1, 1, 1)
            out_data += bias_reshaped
            
        return Tensor(out_data, requires_grad=x.requires_grad)
