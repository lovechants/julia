from julia.core import tensor
import numpy as np 
from julia.core.tensor import Tensor, Function, _ensure_tensor 
from typing import Tuple, List, Optional, Union, Callable, Any

class Layer:
    def __init__(self):
        super().__setattr__('_parameters', {})
        super().__setattr__('_modules', {})
        super().__setattr__('_buffers', {})
        super().__setattr__('training', True)

    def train(self, mode: bool = True):
        self.training = mode
        for module in self._modules.values():
            module.train(mode)
        return self

    def eval(self):
        return self.train(False)
    
    def parameters(self, recurse: bool = True) -> List[Tensor]:
        params_set = set()
        modules_visited = set()

        def collect_params(module, current_params_set, current_modules_visited):
            module_id = id(module)
            if module_id in current_modules_visited:
                return
            current_modules_visited.add(module_id)

            # Check direct parameters
            if hasattr(module, '_parameters'):
                direct_params = getattr(module, '_parameters', {}).values()
                for param in direct_params:
                    if param is not None:
                        current_params_set.add(param)

            # Check submodules
            if recurse and hasattr(module, '_modules'):
                submodules = getattr(module, '_modules', {})
                for submodule_name, submodule in submodules.items():
                    if submodule is not None:
                        collect_params(submodule, current_params_set, current_modules_visited)

        collect_params(self, params_set, modules_visited)
        return list(params_set)

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()
    
    def __setattr__(self, name: str, value: Any):
        if '_parameters' not in self.__dict__ or \
           '_modules' not in self.__dict__ or \
           '_buffers' not in self.__dict__:
             super().__setattr__(name, value)
             return

        params = self.__dict__.get('_parameters')
        modules = self.__dict__.get('_modules')
        buffers = self.__dict__.get('_buffers')

        if params is None or modules is None or buffers is None:
             super().__setattr__(name, value)
             return

        # Remove existing attribute from internal dicts first if replacing
        if name in params: 
            del params[name]
        if name in modules: 
            del modules[name]
        if name in buffers: 
            del buffers[name]

        if isinstance(value, Tensor) and value.requires_grad:
            params[name] = value
        elif isinstance(value, Layer):
            modules[name] = value

        super().__setattr__(name, value)

    def __getattr__(self, name: str) -> Union[Tensor, 'Layer', Any]:
         # Check internal dicts first
         if '_parameters' in self.__dict__:
             if name in self._parameters: return self._parameters[name]
         if '_modules' in self.__dict__:
             if name in self._modules: return self._modules[name]
         if '_buffers' in self.__dict__:
             if name in self._buffers: return self._buffers[name]

         # If not found, raise AttributeError (standard behavior)
         raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __delattr__(self, name: str):
         # Remove from internal dicts if present
         if name in self._parameters: del self._parameters[name]
         elif name in self._modules: del self._modules[name]
         elif name in self._buffers: del self._buffers[name]
         # Call superclass's __delattr__
         super().__delattr__(name)

    def register_buffer(self, name: str, tensor: Optional[Any]):
         # Ensure internal dicts exist
         if '_buffers' not in self.__dict__:
             raise AttributeError("Layer not initialized properly (_buffers missing)")
         if name in self._buffers: del self._buffers[name]
         if tensor is not None:
             self._buffers[name] = tensor
         # Also assign as regular attribute using super to avoid loops if __setattr__ is complex
         super().__setattr__(name, tensor)

    def _register_module(self, name: str, module: Optional['Layer']):
         # Helper mainly for internal use if needed, __setattr__ handles most cases
         if module is None:
             self.__delattr__(name)
             return
         if not isinstance(module, Layer):
             raise TypeError(f"Cannot register '{name}' - not a Layer")
         setattr(self, name, module)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

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
        
        if bias:
            self.bias = Tensor(np.zeros(out_features), requires_grad=True)
        else:
            self.register_buffer('bias', None)

    def set_weight(self, new_weight_data):
        """
        Properly update the weight data without replacing the tensor object.
        This maintains the connection to the parameters list.
        """
        if new_weight_data.shape != self.weight.shape:
            raise ValueError(f"New weight shape {new_weight_data.shape} doesn't match expected {self.weight.shape}")
        
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
            out = out + self.bias
            
        return out
    
    #TODO Sequential + dropout + flatten

class Sequential(Layer):
    """
    Sequential layer container
    """
    def __init__(self, *layers):
        super().__init__() # Calls Layer.__init__ which sets up _modules

        if not hasattr(self, '_modules') or self._modules is None:
             # This should not happen if Layer.__init__ is correct
             raise RuntimeError("Sequential parent Layer not initialized correctly (_modules missing)")

        self._layers_list = [] # Store ordered layers/functions for forward pass
        for i, layer in enumerate(layers):
            layer_name = f'layer_{i}'
            if isinstance(layer, Layer):
                self._modules[layer_name] = layer
                super().__setattr__(layer_name, layer)
                self._layers_list.append(layer) # Add to ordered list
            elif callable(layer):
                self._layers_list.append(layer)
            else:
                raise TypeError(f"Sequential layer expected Layer instance or callable, got {type(layer)}")


    def forward(self, x: Tensor) -> Tensor:
        for layer_or_fn in self._layers_list:
            x = layer_or_fn(x)
        return x



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
        from julia.core.ops import Dropout,Add
        """ Applying dropout """
        x = _ensure_tensor(x)
        return x.dropout(self.p, self.training)

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
