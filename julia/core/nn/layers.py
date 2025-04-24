from julia.core import tensor
import numpy as np 
from julia.core.tensor import Tensor, Function, _ensure_tensor 
from typing import Tuple, List, Optional, Union, Callable

class Layer:
    """ Base class """

    def __init__(self):
        self.training = True
        self.parameters = [] 

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        """forward pass"""
        raise NotImplementedError

    def parameters(self):
        return self.parameters 

    def train(self, mode=True):
        """Set training"""
        self.training = mode 
        return self 

    def eval(self):
        """set eval mode"""
        self.training = False
        return self 

class Linear(Layer):
    """
    y = xW^T + b
    args: 
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weights using He initialization
        scale = np.sqrt(2.0 / in_features)
        self.weight = Tensor(np.random.randn(out_features, in_features) * scale, requires_grad=True)
        self.parameters.append(self.weight)
        
        if bias:
            self.bias = Tensor(np.zeros(out_features), requires_grad=True)
            self.parameters.append(self.bias)
        else:
            self.bias = None

    def set_weight(self, new_weight_data):
        """
        Properly update the weight data without replacing the tensor object.
        This maintains the connection to the parameters list.
        """
        # Ensure the shapes match
        if new_weight_data.shape != self.weight.shape:
            raise ValueError(f"New weight shape {new_weight_data.shape} doesn't match expected {self.weight.shape}")
        
        # Update the data in-place
        self.weight.data = new_weight_data
        return self.weight
    
    def set_bias(self, new_bias_data):
        """
        Properly update the bias data without replacing the tensor object
        """
        if self.bias is None:
            raise ValueError("This layer was initialized with bias=False")
            
        # Ensure the shapes match
        if new_bias_data.shape != self.bias.shape:
            raise ValueError(f"New bias shape {new_bias_data.shape} doesn't match expected {self.bias.shape}")
        
        # Update the data in-place
        self.bias.data = new_bias_data
        return self.bias

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass: y = xW^T + b"""
        x = _ensure_tensor(x)
        
        # Matrix multiply: (batch_size, in_features) @ (out_features, in_features).T
        # We need to transpose the weight matrix to match dimensions
        weight_transposed = self.weight.transpose()
        out = x.matmul(weight_transposed)
        
        if self.bias is not None:
            # Add bias with broadcasting
            out = out + self.bias
            
        return out
    
    #TODO Sequential + dropout + flatten

class Sequential(Layer):
    """
    Sequential layer container
    """

    def __init__(self, *layers):
        super().__init__()
        self.layers = list(layers)

        self.parameters = []
        for layer in self.layers:
            if hasattr(layer, 'parameters'):
                self.parameters.extend(layer.parameters)

    def add(self, layer: Layer):
        """Add layer to sequence"""
        self.layers.append(layer)

    def forward(self, x: Tensor) -> Tensor:
        """Pass through the layers"""
        for layer in self.layers:
            x = layer(x)
        return x

    def train(self, mode=True):
        """Set the training mode for the layers"""
        self.training = mode 
        for layer in self.layers:
            if hasattr(layer, 'train'):
                layer.train(mode)
        return self

    def eval(self):
        """Set eval mode"""
        self.train(false)
        return self


class Dropout(Layer):
    """
    Dropout layer 
    Args: 
        p: probability of element being zeroed
        inplace: True = directly modify the tensor
    """

    def __init__(self, p: float = 0.5, inplace: bool = False):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError(f"Dropout probability must be between 0 and 1, given probablity {p}")
        self.p = p 
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        """ Applying dropout """
        x = _ensure_tensor(x)
        if not self.training or self.p == 0:
            return x 

        mask = np.random.binomial(1, 1 -self.p, size=x.shape) / (1 - self.p) # Creates a dropout mask 
        
        if self.inplace:
            x.data *= mask
            return x 
        else:
            return Tensor(x.data * mask, requires_grad=x.requires_grad)

class Flatten(Layer):
    """
    Flatten the input tensor along dimensions 
    Args:
        start_dim: first dimension to flatten (inclusive)
        end_dim: last dimension to flatten (inclusive)
    """

    def __init__(self, start_dim: int = 1, end_dim: int = -1):
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x: Tensor) -> Tensor:
        """Flatten"""
        x = _ensure_tensor(x)
        
        # Case: end_dim is negative 
        if self.end_dim < 0:
            self.end_dim = len(x.shape) + self.end_dim

        # Case: start_dim == end_dim 
        if self.start_dim == self.end_dim:
            return x 

        # Calculate the shape of the flatten tensor 
        new_shape = list(x.shape)
        flat_dim_size = 1
        for i in range(self.start_dim, self.end_dim + 1):
            flat_dim_size *= new_shape[i]

        new_shape = new_shape[:self.start_dim] + [flat_dim_size] + new_shape[self.end_dim + 1:]
        return Tensor(x.data.reshape(new_shape), requires_grad=x.requires_grad)


class BatchNorm2D(Layer):
    """
    Batch normalization for 2D data (NCHW format)
    
    Args:
        num_features: Number of features/channels
        eps: Value added to denominator for numerical stability
        momentum: Value used for running_mean and running_var computation
        affine: Whether to use learnable affine parameters
        track_running_stats: Whether to track running statistics
    """
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1,
                 affine: bool = True, track_running_stats: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        
        if self.affine:
            self.weight = Tensor(np.ones(num_features), requires_grad=True)
            self.bias = Tensor(np.zeros(num_features), requires_grad=True)
            self.parameters.append(self.weight)
            self.parameters.append(self.bias)
        else:
            self.weight = None
            self.bias = None
            
        if self.track_running_stats:
            self.running_mean = np.zeros(num_features)
            self.running_var = np.ones(num_features)
        else:
            self.running_mean = None
            self.running_var = None
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass: apply batch normalization"""
        x = _ensure_tensor(x)
        
        # Get input dimensions (n_samples, channels, height, width)
        n_samples, channels, height, width = x.shape
        
        if self.training or not self.track_running_stats:
            # Calculate batch statistics
            # Reshape for proper broadcasting
            # (N, C, H, W) -> (N, C, H*W) -> mean across N and H*W dimensions
            x_reshaped = x.data.reshape(n_samples, channels, -1)
            batch_mean = np.mean(x_reshaped, axis=(0, 2))
            batch_var = np.var(x_reshaped, axis=(0, 2))
            
            if self.track_running_stats:
                # Update running statistics
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
                
            mean = batch_mean
            var = batch_var
        else:
            # Use running statistics
            mean = self.running_mean
            var = self.running_var
        
        # Normalize
        # Reshape mean and var for broadcasting
        mean = mean.reshape(1, channels, 1, 1)
        var = var.reshape(1, channels, 1, 1)
        
        x_normalized = (x.data - mean) / np.sqrt(var + self.eps)
        
        if self.affine:
            # Apply affine transformation
            # Reshape weight and bias for broadcasting
            weight = self.weight.data.reshape(1, channels, 1, 1)
            bias = self.bias.data.reshape(1, channels, 1, 1)
            
            out_data = x_normalized * weight + bias
        else:
            out_data = x_normalized
            
        return Tensor(out_data, requires_grad=x.requires_grad)
