import pytest
import numpy as np
import torch
import torch.nn as nn

# Adjust paths as necessary
try:
    from julia.core.tensor import Tensor
    from julia.core.nn.conv import Conv2D

    JULIA_IMPORTED = True
except ImportError:
    JULIA_IMPORTED = False

    # Define dummy classes if imports fail, so pytest collection doesn't break
    class Tensor:
        pass

    class Conv2D:
        pass


# Skip tests if julia components couldn't be imported
pytestmark = pytest.mark.skipif(not JULIA_IMPORTED, reason="Julia components not found")

# --- Test Parameters ---
# Define various configurations to test
test_configs = [
    # in_c, out_c, kernel, stride, padding, bias, batch, height, width
    (3, 8, 3, 1, 0, True, 2, 10, 10),  # Basic case
    (1, 4, 5, 1, 2, True, 1, 16, 16),  # Single input channel, padding
    (4, 16, 3, 2, 1, False, 4, 28, 28),  # Stride=2, no bias
    (8, 8, (1, 1), 1, 0, True, 2, 8, 8),  # 1x1 kernel
    (2, 6, (3, 5), (1, 2), (1, 1), True, 3, 12, 15),  # Rectangular kernel/stride/pad
]


@pytest.mark.parametrize(
    "in_c, out_c, kernel, stride, padding, bias, batch, h, w", test_configs
)
def test_conv2d_pytorch_comparison(
    in_c, out_c, kernel, stride, padding, bias, batch, h, w
):
    """
    Compares the forward pass output of julia.Conv2D (NumPy based)
    with torch.nn.Conv2d using the same inputs and weights.
    """
    print(
        f"\nTesting Config: in={in_c}, out={out_c}, k={kernel}, s={stride}, p={padding}, bias={bias}, shape=({batch},{in_c},{h},{w})"
    )

    julia_conv = Conv2D(
        in_channels=in_c,
        out_channels=out_c,
        kernel_size=kernel,
        stride=stride,
        padding=padding,
        bias=bias,
    )

    torch_conv = nn.Conv2d(
        in_channels=in_c,
        out_channels=out_c,
        kernel_size=kernel,
        stride=stride,
        padding=padding,
        bias=bias,
    )

    # 3. Synchronize Parameters (Copy Julia -> PyTorch)
    # Ensure weights and biases are identical
    julia_weight_np = julia_conv.weight.data.copy()
    julia_bias_np = julia_conv.bias.data.copy() if bias else None

    with torch.no_grad():  # Disable gradient tracking for parameter assignment
        # Convert NumPy arrays to PyTorch tensors and assign
        torch_conv.weight.data = torch.from_numpy(julia_weight_np).float()
        if bias and julia_bias_np is not None:
            torch_conv.bias.data = torch.from_numpy(julia_bias_np).float()

    # 4. Create Identical Inputs
    input_np = np.random.randn(batch, in_c, h, w).astype(np.float32)

    # Julia Tensor input
    julia_input = Tensor(input_np)

    # PyTorch Tensor input
    torch_input = torch.from_numpy(input_np)

    # 5. Perform Forward Passes
    # Julia forward pass
    # Set to eval mode if it affects forward pass (e.g., future BatchNorm)
    # julia_conv.eval() # Assuming your Layer has train/eval
    julia_output = julia_conv(julia_input)

    # PyTorch forward pass
    torch_conv.eval()  # Set PyTorch layer to eval mode
    torch_output = torch_conv(torch_input)

    # 6. Compare Outputs
    julia_output_np = julia_output.data
    torch_output_np = (
        torch_output.detach().numpy()
    )  # Get NumPy array from PyTorch tensor

    # Check shapes first
    assert julia_output_np.shape == torch_output_np.shape, (
        f"Shape mismatch: Julia={julia_output_np.shape}, PyTorch={torch_output_np.shape}"
    )

    # Check values using np.allclose for float comparison
    # Adjust tolerances (rtol, atol) if needed based on float precision
    assert np.allclose(julia_output_np, torch_output_np, rtol=1e-5, atol=1e-6), (
        "Value mismatch between Julia and PyTorch outputs."
    )

    print("Outputs match")
