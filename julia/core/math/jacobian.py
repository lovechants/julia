import numpy as np 
from typing import List, Dict, Optional, Any, Callable, Union, Tuple
from julia.core.tensor import Function, Tensor

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

class JacobianVectorProduct:
    """
    Compiute Jacobian-vector products 
    """
    @staticmethod
    def compute_jvp(func: Callable, inputs: List[Tensor], v: List[Tensor]) -> Tensor:
        """
        Compute JVP for a function with respect to its inputs and vectors 
        Args: 
            Func: Function that takes inputs and returns a Tensor 
            Inputs: List of input Tensors to the function 
            v: List of vectors to compute JVP with the same shape of the inputs

        Returns: 
            JVP result 
        """

        input_copies = [] 
        for inp in inputs:
            if isinstance(inp, Tensor):
                copy = Tensor(inp.data.copy(), requires_grad=True)
                input_copies.append(copy)
            else:
                input_copies.append(inp)

        output = func(*input_copies)

        if len(v) != len(input_copies):
            raise ValueError(f"Expected vector: {len(input_copies)} got, {len(v)}")

        for i, (inp,vec) in enumerate(zip(input_copies, v)):
            if isinstance(inp, Tensor) and isinstance(vec, Tensor):
                if inp.shape != vec.shape:
                    raise ValueError(f"Vector shape {vec.shape} doesn't match input shape {inp.shape} at index {i}")

        epsilon = 1e-6

        perturbed_inputs = []
        for inp, vec in zip(inputs, v):
            if isinstance(inp, Tensor) and isinstance(vec, Tensor):
                perturbed = Tensor(inp.data + epsilon * vec.data, requires_grad=True)
                perturbed_inputs.append(perturbed)
            else:
                perturbed_inputs.append(inp)

        perturbed_ouputs = func(*perturbed_inputs)
        # Compute JVP as (f(x + ε*v) - f(x)) / ε
        if isinstance(output, Tensor) and isinstance(perturbed_ouputs, Tensor):
            jvp_result = Tensor((perturbed_ouputs.data - output.data) / epsilon)
            return jvp_result
        else:
            raise TypeError("Function output must be a Tensor")

class VectorJacobianProduct:
    """
    Compute Vector-Jacobian products (VJPs) efficiently
    """
    @staticmethod
    def compute_vjp(func: Callable, inputs: List[Tensor], v: Tensor) -> List[Tensor]:
        """
        Compute VJP for a function with respect to its inputs and a vector
        
        Args:
            func: Function that takes inputs and returns a Tensor
            inputs: List of input Tensors to the function
            v: Vector to compute VJP with (same shape as function output)
            
        Returns:
            List of VJP results (one per input)
        """
        # Ensure inputs require grad
        input_copies = []
        for inp in inputs:
            if isinstance(inp, Tensor):
                copy = Tensor(inp.data.copy(), requires_grad=True)
                input_copies.append(copy)
            else:
                input_copies.append(inp)
        
        # Forward pass
        output = func(*input_copies)
        
        # Ensure vector matches output shape
        if isinstance(output, Tensor) and isinstance(v, Tensor):
            if output.shape != v.shape:
                raise ValueError(f"Vector shape {v.shape} doesn't match output shape {output.shape}")
        else:
            raise TypeError("Function output and vector must be Tensors")
        
        # Run backward pass with vector as initial gradient
        output.backward(gradient=v)
        
        # Extract gradients for each input
        vjp_results = []
        for inp_copy in input_copies:
            if isinstance(inp_copy, Tensor) and inp_copy.grad is not None:
                vjp_results.append(inp_copy.grad)
            else:
                vjp_results.append(None)
        
        return vjp_results


def compute_jacobian(func: Callable, inputs: List[Tensor], create_graph: bool = False) -> List[np.ndarray]:
    """
    Compute the full Jacobian matrix for a function with respect to its inputs
    
    Args:
        func: Function that takes inputs and returns a Tensor
        inputs: List of input Tensors to the function
        create_graph: Whether to create a computational graph for the Jacobian
        
    Returns:
        List of Jacobian matrices (one per input)
    """
    # Ensure inputs require grad
    input_copies = []
    for inp in inputs:
        if isinstance(inp, Tensor):
            copy = Tensor(inp.data.copy(), requires_grad=True)
            input_copies.append(copy)
        else:
            input_copies.append(inp)
    
    # Forward pass
    output = func(*input_copies)
    
    if not isinstance(output, Tensor):
        raise TypeError("Function output must be a Tensor")
    
    # Initialize list to store Jacobian matrices
    jacobians = []
    
    # For each input that requires grad
    for i, inp in enumerate(input_copies):
        if isinstance(inp, Tensor):
            # Initialize Jacobian matrix for this input
            if output.data.ndim == 0:  # Scalar output
                jacobian = np.zeros(inp.shape)
            else:
                jacobian = np.zeros(output.shape + inp.shape)
            
            # For each element in the output
            if output.data.ndim == 0:  # Scalar output
                # Create a unit vector
                v = Tensor(1.0)
                
                # Compute vector-Jacobian product
                output.backward(gradient=v, retain_graph=True, create_graph=create_graph)
                
                # Get gradients
                if inp.grad is not None:
                    jacobian = inp.grad.data.copy()
            else:
                # For each element in the output
                it = np.nditer(output.data, flags=['multi_index'])
                while not it.finished:
                    # Clear gradients
                    for p in input_copies:
                        if isinstance(p, Tensor):
                            p.grad = None
                    
                    # Create a unit vector
                    unit_vec = np.zeros_like(output.data)
                    unit_vec[it.multi_index] = 1.0
                    
                    # Compute vector-Jacobian product
                    output.backward(gradient=Tensor(unit_vec), retain_graph=True, create_graph=create_graph)
                    
                    # Get gradients
                    if inp.grad is not None:
                        jacobian[it.multi_index] = inp.grad.data.copy()
                    
                    it.iternext()
            
            jacobians.append(jacobian)
        else:
            # Non-Tensor inputs have no Jacobian
            jacobians.append(None)
    
    return jacobians

