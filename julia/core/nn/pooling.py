import numpy as np 
from julia.core.tensor import Tensor, _ensure_tensor 
from julia.core.nn.layers import Layer
from typing import Tuple, Union, Optional

class MaxPool2D(Layer):
    """
    2D max pooling layer
    
    Args:
        kernel_size: Size of pooling window (int or tuple)
        stride: Stride of pooling operation (int or tuple)
        padding: Padding added to input (int or tuple)
    """
    def __init__(self, kernel_size: Union[int, Tuple[int, int]], 
                 stride: Optional[Union[int, Tuple[int, int]]] = None,
                 padding: Union[int, Tuple[int, int]] = 0):
        super().__init__()
        
        # Handle scalar or tuple kernel size
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
            
        # If stride is None, use kernel_size as stride
        if stride is None:
            self.stride = self.kernel_size
        elif isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride
            
        # Handle scalar or tuple padding
        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass: apply 2D max pooling"""
        x = _ensure_tensor(x)
        
        # Get input dimensions (n_samples, channels, height, width)
        n_samples, channels, h_in, w_in = x.shape
        
        # Calculate output dimensions
        h_out = (h_in + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        w_out = (w_in + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1
        
        # Pad input if needed
        if self.padding[0] > 0 or self.padding[1] > 0:
            x_padded = np.pad(
                x.data,
                ((0, 0), (0, 0), (self.padding[0], self.padding[0]), (self.padding[1], self.padding[1])),
                mode='constant',
                constant_values=float('-inf')  # Use -inf for max pooling padding
            )
        else:
            x_padded = x.data
            
        # Initialize output
        out_data = np.zeros((n_samples, channels, h_out, w_out))
        
        # Perform max pooling
        for h in range(h_out):
            for w in range(w_out):
                h_start = h * self.stride[0]
                h_end = h_start + self.kernel_size[0]
                w_start = w * self.stride[1]
                w_end = w_start + self.kernel_size[1]
                
                # Extract region and compute max
                region = x_padded[:, :, h_start:h_end, w_start:w_end]
                out_data[:, :, h, w] = np.max(region, axis=(2, 3))
                
        return Tensor(out_data, requires_grad=x.requires_grad)


class AvgPool2D(Layer):
    """
    2D average pooling layer
    
    Args:
        kernel_size: Size of pooling window (int or tuple)
        stride: Stride of pooling operation (int or tuple)
        padding: Padding added to input (int or tuple)
    """
    def __init__(self, kernel_size: Union[int, Tuple[int, int]], 
                 stride: Optional[Union[int, Tuple[int, int]]] = None,
                 padding: Union[int, Tuple[int, int]] = 0):
        super().__init__()
        
        # Handle scalar or tuple kernel size
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
            
        # If stride is None, use kernel_size as stride
        if stride is None:
            self.stride = self.kernel_size
        elif isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride
            
        # Handle scalar or tuple padding
        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass: apply 2D average pooling"""
        x = _ensure_tensor(x)
        
        # Get input dimensions (n_samples, channels, height, width)
        n_samples, channels, h_in, w_in = x.shape
        
        # Calculate output dimensions
        h_out = (h_in + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        w_out = (w_in + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1
        
        # Pad input if needed
        if self.padding[0] > 0 or self.padding[1] > 0:
            x_padded = np.pad(
                x.data,
                ((0, 0), (0, 0), (self.padding[0], self.padding[0]), (self.padding[1], self.padding[1])),
                mode='constant',
                constant_values=0  # Use 0 for avg pooling padding
            )
        else:
            x_padded = x.data
            
        # Initialize output
        out_data = np.zeros((n_samples, channels, h_out, w_out))
        
        # Perform average pooling
        for h in range(h_out):
            for w in range(w_out):
                h_start = h * self.stride[0]
                h_end = h_start + self.kernel_size[0]
                w_start = w * self.stride[1]
                w_end = w_start + self.kernel_size[1]
                
                # Extract region and compute average
                region = x_padded[:, :, h_start:h_end, w_start:w_end]
                out_data[:, :, h, w] = np.mean(region, axis=(2, 3))
                
        return Tensor(out_data, requires_grad=x.requires_grad)

class AdaptiveAvgPool2D(Layer):
    """
    Adaptive Average Pooling for 2D inputs
    
    Pools the input to the target output size
    
    Args:
        output_size: Target output size (H, W)
    """
    def __init__(self, output_size: Tuple[int, int]):
        super().__init__()
        self.output_size = output_size
        
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass: apply adaptive average pooling"""
        x = _ensure_tensor(x)
        
        # Get input dimensions
        batch_size, channels, h_in, w_in = x.shape
        h_out, w_out = self.output_size
        
        # Compute pooling regions
        h_stride = h_in / h_out
        w_stride = w_in / w_out
        
        # Initialize output
        out_data = np.zeros((batch_size, channels, h_out, w_out))
        
        # For each output position
        for b in range(batch_size):
            for c in range(channels):
                for h in range(h_out):
                    # Calculate pooling region for height
                    h_start = int(np.floor(h * h_stride))
                    h_end = int(np.ceil((h + 1) * h_stride))
                    h_end = min(h_end, h_in)  # Ensure we don't go out of bounds
                    
                    for w in range(w_out):
                        # Calculate pooling region for width
                        w_start = int(np.floor(w * w_stride))
                        w_end = int(np.ceil((w + 1) * w_stride))
                        w_end = min(w_end, w_in)  # Ensure we don't go out of bounds
                        
                        # Perform average pooling over the region
                        region = x.data[b, c, h_start:h_end, w_start:w_end]
                        out_data[b, c, h, w] = np.mean(region)
        
        return Tensor(out_data, requires_grad=x.requires_grad)


class GlobalAvgPool2D(Layer):
    """
    Global Average Pooling for 2D inputs
    
    Pools the entire spatial dimensions to a single value per channel
    """
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass: apply global average pooling"""
        x = _ensure_tensor(x)
        
        # Get input dimensions
        batch_size, channels, h_in, w_in = x.shape
        
        # Perform global average pooling
        out_data = np.mean(x.data, axis=(2, 3))
        
        # Reshape to (batch_size, channels, 1, 1) for consistency
        out_data = out_data.reshape(batch_size, channels, 1, 1)
        
        return Tensor(out_data, requires_grad=x.requires_grad)


class GlobalMaxPool2D(Layer):
    """
    Global Max Pooling for 2D inputs
    
    Pools the entire spatial dimensions to the maximum value per channel
    """
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass: apply global max pooling"""
        x = _ensure_tensor(x)
        
        # Get input dimensions
        batch_size, channels, h_in, w_in = x.shape
        
        # Perform global max pooling
        out_data = np.max(x.data, axis=(2, 3))
        
        # Reshape to (batch_size, channels, 1, 1) for consistency
        out_data = out_data.reshape(batch_size, channels, 1, 1)
        
        return Tensor(out_data, requires_grad=x.requires_grad)
