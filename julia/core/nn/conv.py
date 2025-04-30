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

        if bias: 
            self.bias = Tensor(np.zeros(out_channels), requires_grad=True)

        else:
            self.bias = None 

    def forward(self, x: Tensor) -> Tensor:

        from julia.core.ops_nn import Conv1DFunction
        """Forward pass -> apply the 1D conv """
        
        x = _ensure_tensor(x)
        return Conv1DFunction.apply(x, self.weight, self.bias, self.stride, self.padding)

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
        
        if bias:
            self.bias = Tensor(np.zeros(out_channels), requires_grad=True)
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
        from julia.core.ops_nn import Conv2DFunction 
        """Forward pass: apply 2D convolution"""
        x = _ensure_tensor(x)
        return Conv2DFunction.apply(x, self.weight, self.bias, self.stride, self.padding)

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
        
        if bias:
            self.bias = Tensor(np.zeros(out_channels), requires_grad=True)
        else:
            self.bias = None
    
    def forward(self, x: Tensor) -> Tensor:
        from julia.core.ops_nn import Conv3DFunction
        """Forward pass: apply 3D convolution"""
        x = _ensure_tensor(x)
        return Conv3DFunction.apply(x, self.weight, self.bias, self.stride, self.padding)


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
        
        if bias:
            self.bias = Tensor(np.zeros(out_channels), requires_grad=True)
        else:
            self.bias = None
    
    def forward(self, x: Tensor) -> Tensor:
        from julia.core.ops_nn import TransposedConv2DFunction
        """Forward pass: apply 2D transposed convolution"""
        x = _ensure_tensor(x)
        
        return TransposedConv2DFunction.apply(x, self.weight, self.bias, self.stride, self.padding, self. output_padding)
        
        

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
        self.in_channels = in_channels
        
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
        
        # Pointwise convolution weights - 1x1 convolution to mix channels
        pointwise_shape = (out_channels, self.depth_channels, 1, 1)
        scale_pointwise = np.sqrt(2.0 / self.depth_channels)
        self.pointwise_weight = Tensor(np.random.randn(*pointwise_shape) * scale_pointwise, requires_grad=True)
        
        if bias:
            self.bias = Tensor(np.zeros(out_channels), requires_grad=True)
        else:
            self.bias = None
            
    def forward(self, x: Tensor) -> Tensor:
        from julia.core.ops_nn import DepthwiseSeparableConv2DFunction
        """Forward pass: apply depthwise separable convolution"""
        x = _ensure_tensor(x)
        depth_multipler = self.depth_channels // self.in_channels

        return DepthwiseSeparableConv2DFunction.apply(x, self.depthwise_weight, self.pointwise_weight, self.bias, self.stride, self.padding,depth_multipler)
        
        


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
        
        if bias:
            self.bias = Tensor(np.zeros(out_channels), requires_grad=True)
        else:
            self.bias = None
            
    def forward(self, x: Tensor) -> Tensor:
        from julia.core.ops_nn import DilatedConv2DFunction
        """Forward pass: apply dilated convolution"""
        x = _ensure_tensor(x)

        return DilatedConv2DFunction.apply(x, self.weight, self.bias, self.stride, self.padding, self.dilation)

        
