import numpy as np
from typing import List, Callable
from julia.core.tensor import Tensor

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
    Compute Jacobian-vector products efficiently
    """

    @staticmethod
    def compute_jvp(func: Callable, inputs: List[Tensor], v: List[Tensor]) -> Tensor:
        """
        Compute JVP for a function with respect to its inputs and vectors
        Args:
            func: Function that takes inputs and returns a Tensor
            inputs: List of input Tensors to the function
            v: List of vectors to compute JVP with the same shape of the inputs

        Returns:
            JVP result
        """
        # Validate inputs
        if len(v) != len(inputs):
            raise ValueError(f"Expected {len(inputs)} vectors, got {len(v)}")

        for i, (inp, vec) in enumerate(zip(inputs, v)):
            if isinstance(inp, Tensor) and isinstance(vec, Tensor):
                if inp.shape != vec.shape:
                    raise ValueError(
                        f"Vector shape {vec.shape} doesn't match input shape {inp.shape} at index {i}"
                    )

        epsilon = 1e-7  # Smaller epsilon for better numerical precision

        # Create perturbed inputs
        perturbed_inputs = []
        for inp, vec in zip(inputs, v):
            if isinstance(inp, Tensor) and isinstance(vec, Tensor):
                perturbed = Tensor(inp.data + epsilon * vec.data, requires_grad=False)
                perturbed_inputs.append(perturbed)
            else:
                perturbed_inputs.append(inp)

        # Compute function at original and perturbed points
        original_output = func(*inputs)
        perturbed_output = func(*perturbed_inputs)

        # Compute JVP as (f(x + ε*v) - f(x)) / ε
        if isinstance(original_output, Tensor) and isinstance(perturbed_output, Tensor):
            jvp_result = Tensor(
                (perturbed_output.data - original_output.data) / epsilon
            )
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
        input_copies = []
        for inp in inputs:
            if isinstance(inp, Tensor):
                copy = Tensor(inp.data.copy(), requires_grad=True)
                input_copies.append(copy)
            else:
                input_copies.append(inp)

        output = func(*input_copies)

        if isinstance(output, Tensor) and isinstance(v, Tensor):
            if output.shape != v.shape:
                raise ValueError(
                    f"Vector shape {v.shape} doesn't match output shape {output.shape}"
                )
        else:
            raise TypeError("Function output and vector must be Tensors")

        output.backward(gradient=v)

        vjp_results = []
        for inp_copy in input_copies:
            if isinstance(inp_copy, Tensor) and inp_copy.grad is not None:
                vjp_results.append(inp_copy.grad)
            else:
                vjp_results.append(None)

        return vjp_results


def compute_jacobian(
    func: Callable, inputs: List[Tensor], create_graph: bool = False
) -> List[np.ndarray]:
    """
    Compute the full Jacobian matrix for a function with respect to its inputs

    Args:
        func: Function that takes inputs and returns a Tensor
        inputs: List of input Tensors to the function
        create_graph: Whether to create a computational graph for the Jacobian

    Returns:
        List of Jacobian matrices (one per input)
    """
    input_copies = []
    for inp in inputs:
        if isinstance(inp, Tensor):
            copy = Tensor(inp.data.copy(), requires_grad=True)
            input_copies.append(copy)
        else:
            input_copies.append(inp)

    output = func(*input_copies)

    if not isinstance(output, Tensor):
        raise TypeError("Function output must be a Tensor")

    jacobians = []

    for i, inp in enumerate(input_copies):
        if isinstance(inp, Tensor):
            # Initialize Jacobian matrix for this input
            if output.data.ndim == 0:  # Scalar output
                jacobian = np.zeros(inp.shape)
            else:
                jacobian = np.zeros(output.shape + inp.shape)

            if output.data.ndim == 0:
                v = Tensor(1.0)

                for p in input_copies:
                    if isinstance(p, Tensor):
                        p.grad = None

                # Compute vector-Jacobian product
                output.backward(
                    gradient=v, retain_graph=True, create_graph=create_graph
                )

                # Get gradients
                if inp.grad is not None:
                    jacobian = inp.grad.data.copy()
            else:
                it = np.nditer(output.data, flags=["multi_index"])
                while not it.finished:
                    for p in input_copies:
                        if isinstance(p, Tensor):
                            p.grad = None

                    unit_vec = np.zeros_like(output.data)
                    unit_vec[it.multi_index] = 1.0

                    # Compute vector-Jacobian product
                    output.backward(
                        gradient=Tensor(unit_vec),
                        retain_graph=True,
                        create_graph=create_graph,
                    )

                    if inp.grad is not None:
                        jacobian[it.multi_index] = inp.grad.data.copy()

                    it.iternext()

            jacobians.append(jacobian)
        else:
            # Non-Tensor inputs have no Jacobian
            jacobians.append(None)

    return jacobians
