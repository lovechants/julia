import numpy as np 
from typing import Callable, List, Tuple, Union 
from julia.core.math import jvp, Dual, jacobian

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

class VJP:
    """
    Reverse Mode AD via VJP 
    """

    @staticmethod
    def compute(func: Callable, primals: List) -> Tuple[any, Callable]:
        """
        Compute VJP: PLEASE PUT EQUATION 
        return: primal_out, vjp_fn
        """


        tape = []

        def traced_func(*args):
            result = func(*args)
            tape.append((args, result))

            return result


        primal_out = traced_func(*primals)

        def vjp_fn(cotangent):
            if not tape: raise RuntimeError("VJP called before forward pass")

        args, result = tape[-1]

        grads = []
        for arg in args: 
            if hasattr(arg, 'grad'):
                arg.grad = None

        if hasattr(result, 'backward'):
            if hasattr(cotangent, 'data'):



