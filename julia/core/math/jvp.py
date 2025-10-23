import numpy as np 
from typing import Callable, List, Tuple, Union, Optional

"""
Why JVP or VJP 
- memory-efficient training 
- gradient checkpoints 
- second order optimization for the hessian / newton 
- linear attention 
- efficient attention gradient computation 
- meta learning + hyperparam optimization 
- diff eq for nn 
- RNN 
"""

class Dual: 
    """
    Dual number 
    """

    def __init__(self, primal, tangent):
        self.primal = primal 
        self.tangent = tangent 

    def __add__(self, other):
        if isinstance(other, Dual):
            return Dual(self.primal + other.primal, self.tangent + other.tangent)
        else:
            return Dual(self.primal + other, self.tangent + other)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if isinstance(self, Dual):
            return Dual(self.primal * other.primal, self.primal * other.tangent + self.tangent * other.primal)
        else:
            return Dual(self.primal * other, self.tangent * other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __sub__(self, other):
        if isinstance(other, Dual):
            return Dual(self.primal - other.primal, self.tangent - other.tangent)
        else:
            return Dual(self.primal - other, self.tangent)

    def __rsub__(self, other):
        return Dual(other - self.primal, -self.tangent)
    
    def __truediv__(self, other):
        if isinstance(other, Dual):
            return Dual(
                self.primal / other.primal,
                (self.tangent * other.primal - self.primal * other.tangent) / (other.primal ** 2)
            )
        else:
            return Dual(self.primal / other, self.tangent / other)
    
    def __rtruediv__(self, other):
        return Dual(
            other / self.primal,
            -other * self.tangent / (self.primal ** 2)
        )
    
    def __pow__(self, other):
        if isinstance(other, (int, float)):
            return Dual(
                self.primal ** other,
                other * (self.primal ** (other - 1)) * self.tangent
            )
        else:
            raise NotImplementedError("Dual power only supports scalar exponents")
    
    def __neg__(self):
        return Dual(-self.primal, -self.tangent)
    
    def sin(self):
        return Dual(np.sin(self.primal), np.cos(self.primal) * self.tangent)
    
    def cos(self):
        return Dual(np.cos(self.primal), -np.sin(self.primal) * self.tangent)
    
    def exp(self):
        exp_primal = np.exp(self.primal)
        return Dual(exp_primal, exp_primal * self.tangent)
    
    def log(self):
        return Dual(np.log(self.primal), self.tangent / self.primal)
    
    def tanh(self):
        tanh_primal = np.tanh(self.primal)
        return Dual(tanh_primal, (1 - tanh_primal ** 2) * self.tangent)
    
    def sigmoid(self):
        sig = 1 / (1 + np.exp(-self.primal))
        return Dual(sig, sig * (1 - sig) * self.tangent)

class JVP: 
    """
    Forward mode ad via JVP
    """

    @staticmethod
    def compute(func: Callable, primals: List, tangents: List) -> Tuple:
        """
        Compute : (Put the equation here when im not rushing and tired)
        Return 
        """

        if len(primals) != len(tangents):
            raise ValueError(f"Primals and tangents length mistmatch: {len(primals)} & {len(tangents)}")

        for p, t in zip(primals, tangents):
            if hasattr(p, 'shape') and hasattr(t, 'shape'):
                if p.shape != t.shape:
                    raise ValueError(f"Primal and tangent shape mismatch: primal - {p.shape} & tangent - {t.shape}")

        dual_inputs = []
        for primal, tangent in zip(primals, tangents):
            dual_inputs.append(Dual(primal, tangent))

        dual_output = func(*dual_inputs)

        if isinstance(dual_output, Dual):
            return dual_output.primal, dual_output.tangent 

        elif isinstance(dual_output, Tuple):
            primals_out = tuple(d.primal if isinstance(d, Dual) else d for d in dual_output)
            tangents_out = tuple(d.tangent if isinstance(d, Dual) else None for d in dual_output)

            return primals_out, tangents_out
        
        else:
            return dual_output, None 

def jvp(func: Callable, primals: List, tangents: List) -> Tuple:
    return JVP.compute(func, primals, tangents)

def jacobian(func: Callable, x):
    """Compute full Jacobian matrix using JVP"""
    if hasattr(x, 'shape'):
        input_shape = x.shape
        input_size = np.prod(input_shape)
    else:
        input_size = 1
        input_shape = (1,)
    
    primal_out, _ = jvp(func, [x], [np.zeros_like(x)])
    
    if hasattr(primal_out, 'shape'):
        output_shape = primal_out.shape
        output_size = np.prod(output_shape)
    else:
        output_size = 1
        output_shape = (1,)
    
    jac = np.zeros((output_size, input_size))
    
    for i in range(input_size):
        tangent = np.zeros(input_shape)
        tangent.flat[i] = 1.0
        
        _, tangent_out = jvp(func, [x], [tangent])
        
        if hasattr(tangent_out, 'flatten'):
            jac[:, i] = tangent_out.flatten()
        else:
            jac[:, i] = tangent_out
    
    return jac.reshape(output_shape + input_shape)
