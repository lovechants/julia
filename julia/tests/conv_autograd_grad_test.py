import numpy as np
import unittest  # Or use plain asserts if not using unittest structure

from julia.core.tensor import Tensor
from julia.core.nn.conv import (
    Conv1D,
    Conv2D,
    Conv3D,
    TransposedConv2D,
    DepthwiseSeparableConv2D,
    DilatedConv2D,
)


def compute_numerical_gradient(
    tensor_to_check: Tensor,
    loss_function,  # Function that takes no args and computes loss
    eps=1e-5,
):
    """
    Computes numerical gradient for a given tensor using central differences.

    Args:
        tensor_to_check: The Tensor object (parameter or input) to compute gradient for.
        loss_function: A zero-argument function that performs the forward pass
                       and returns the scalar loss value (as a float or 0-dim ndarray).
                       It must use the current state of tensor_to_check.data.
        eps: The small perturbation value.

    Returns:
        A numpy array containing the numerical gradient.
    """
    numerical_grad = np.zeros_like(
        tensor_to_check.data, dtype=np.float64
    )  # Use float64 for precision
    original_data = tensor_to_check.data.copy()

    # Iterate through each element of the tensor data
    it = np.nditer(original_data, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        original_value = original_data[idx]

        # Compute loss for (value + eps)
        tensor_to_check.data[idx] = original_value + eps
        loss_plus = loss_function()

        # Compute loss for (value - eps)
        tensor_to_check.data[idx] = original_value - eps
        loss_minus = loss_function()

        # Compute numerical gradient for this element
        numerical_grad[idx] = (loss_plus - loss_minus) / (2 * eps)

        # Restore original value
        tensor_to_check.data[idx] = original_value

        it.iternext()

    return numerical_grad


# --- Gradient Tests ---


# Use unittest.TestCase structure or plain pytest functions with asserts
class TestConvGradientCheck(unittest.TestCase):
    def _check_gradients(self, layer, x, layer_name="Layer"):
        """Helper method to perform gradient check for a layer"""
        print(f"\n--- Gradient Check for {layer_name} ---")

        # Ensure parameters require grad
        params_to_check = []
        if hasattr(layer, "weight") and layer.weight is not None:
            layer.weight.requires_grad = True
            params_to_check.append(layer.weight)
        if hasattr(layer, "bias") and layer.bias is not None:
            layer.bias.requires_grad = True
            params_to_check.append(layer.bias)
        # Add specific weights for DepthwiseSeparableConv2D
        if hasattr(layer, "depthwise_weight") and layer.depthwise_weight is not None:
            layer.depthwise_weight.requires_grad = True
            params_to_check.append(layer.depthwise_weight)
        if hasattr(layer, "pointwise_weight") and layer.pointwise_weight is not None:
            layer.pointwise_weight.requires_grad = True
            params_to_check.append(layer.pointwise_weight)

        # Define loss function (closure to capture layer and x)
        def compute_loss():
            y = layer(x)
            # Use .sum() which should call Sum.apply for autograd
            loss_tensor = y.sum()
            # Return scalar loss value
            return (
                loss_tensor.data.item()
                if loss_tensor.data.ndim == 0
                else loss_tensor.data
            )

        # 1. Compute Analytical Gradients
        # Zero grads first
        x.zero_grad()
        for p in params_to_check:
            p.zero_grad()
        # Forward and Backward
        y = layer(x)
        loss = y.sum()
        loss.backward()

        # Store analytical gradients
        analytic_grads = {}
        analytic_grads["x"] = x.grad.data.copy() if x.grad else None
        for i, p in enumerate(params_to_check):
            analytic_grads[f"param_{i}"] = p.grad.data.copy() if p.grad else None

        # 2. Compute Numerical Gradients
        numerical_grads = {}
        # Check input gradient
        print("  Computing numerical gradient for input x...")
        numerical_grads["x"] = compute_numerical_gradient(x, compute_loss)
        # Check parameter gradients
        for i, p in enumerate(params_to_check):
            print(
                f"  Computing numerical gradient for parameter {i} (shape {p.shape})..."
            )
            numerical_grads[f"param_{i}"] = compute_numerical_gradient(p, compute_loss)

        # 3. Compare Gradients
        print("  Comparing gradients...")
        # Compare input gradient
        self.assertIsNotNone(analytic_grads["x"], "Analytical grad for x is None")
        self.assertTrue(
            np.allclose(
                analytic_grads["x"], numerical_grads["x"], rtol=1e-4, atol=1e-5
            ),
            f"{layer_name} Input 'x' gradient mismatch:\nAnalytic:\n{analytic_grads['x']}\nNumerical:\n{numerical_grads['x']}",
        )
        # Compare parameter gradients
        for i, p in enumerate(params_to_check):
            param_key = f"param_{i}"
            self.assertIsNotNone(
                analytic_grads[param_key], f"Analytical grad for param {i} is None"
            )
            self.assertTrue(
                np.allclose(
                    analytic_grads[param_key],
                    numerical_grads[param_key],
                    rtol=1e-4,
                    atol=1e-5,
                ),
                f"{layer_name} Param {i} (shape {p.shape}) gradient mismatch:\nAnalytic:\n{analytic_grads[param_key]}\nNumerical:\n{numerical_grads[param_key]}",
            )
        print("  Gradients match!")

    def test_conv1d_gradient_check(self):
        batch, in_c, seq = 2, 2, 5  # Small dimensions
        out_c, k, s, p = 3, 3, 1, 1
        x = Tensor(np.random.randn(batch, in_c, seq), requires_grad=True)
        conv = Conv1D(in_c, out_c, k, s, p, bias=True)
        self._check_gradients(conv, x, "Conv1D")

    def test_conv2d_gradient_check(self):
        batch, in_c, h, w = 2, 2, 4, 5  # Small dimensions
        out_c, k, s, p = 3, (2, 3), 1, 1
        x = Tensor(np.random.randn(batch, in_c, h, w), requires_grad=True)
        conv = Conv2D(in_c, out_c, k, s, p, bias=True)
        self._check_gradients(conv, x, "Conv2D")

    def test_conv3d_gradient_check(self):
        batch, in_c, d, h, w = 2, 1, 3, 4, 4  # Small dimensions
        out_c, k, s, p = 2, 2, 1, 1
        x = Tensor(np.random.randn(batch, in_c, d, h, w), requires_grad=True)
        conv = Conv3D(in_c, out_c, k, s, p, bias=True)
        self._check_gradients(conv, x, "Conv3D")

    def test_transposed_conv2d_gradient_check(self):
        batch, in_c, h, w = 2, 2, 4, 5
        out_c, k, s, p, op = 3, 3, 2, 1, 1
        x = Tensor(np.random.randn(batch, in_c, h, w), requires_grad=True)
        conv = TransposedConv2D(in_c, out_c, k, s, p, op, bias=True)
        self._check_gradients(conv, x, "TransposedConv2D")

    def test_depthwise_separable_conv2d_gradient_check(self):
        batch, in_c, h, w = 2, 2, 5, 5
        out_c, k, s, p, dm = 3, 3, 1, 1, 1  # depth_multiplier=1
        x = Tensor(np.random.randn(batch, in_c, h, w), requires_grad=True)
        conv = DepthwiseSeparableConv2D(in_c, out_c, k, s, p, dm, bias=True)
        self._check_gradients(conv, x, "DepthwiseSeparableConv2D")

    def test_dilated_conv2d_gradient_check(self):
        batch, in_c, h, w = 2, 2, 6, 6
        out_c, k, s, p, d = 3, 2, 1, 1, 2  # dilation=2
        x = Tensor(np.random.randn(batch, in_c, h, w), requires_grad=True)
        conv = DilatedConv2D(in_c, out_c, k, s, p, d, bias=True)
        self._check_gradients(conv, x, "DilatedConv2D")


if __name__ == "__main__":
    # Example of running with unittest runner
    # You can also run this file with pytest
    unittest.main()
