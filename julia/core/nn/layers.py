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

    def __init__(self, in_features: int, out_features: int, bias: bool=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        scale = np.sqrt(2.0 / in_features)
        self.weight = Tensor(np.random.randn(out_features, in_features) * scale, requires_grad=True)
        self.parameters.append(self.weight)

        if bias: 
            self.bias = Tensor(np.zeros(out_features), requires_grad=True)
            self.parameters.append(self.bias)
        else:
            self.bias = None 

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass: y = xW^T + b"""
        x = _ensure_tensor(x)

        """ Matrix Multiply: """
        weight_T = Tensor(self.weight.data.T, requires_grad=self.weight.requires_grad)
        out = x.matmul(weight_T)
        if self.bias is not None: 
            out = out + self.bias 
        return out
    
    #TODO Sequential + dropout + flatten

