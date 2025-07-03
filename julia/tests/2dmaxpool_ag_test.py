import numpy as np
import unittest
from julia.core.tensor import Tensor
from julia.core.ops_nn import MaxPool2DFunction


class TestPoolingFunctions(unittest.TestCase):
    def test_maxpool2d_forward(self):
        """Test the forward pass of MaxPool2D"""
        # Create a simple input
        x = np.array(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
            dtype=np.float32,
        ).reshape(1, 1, 4, 4)
        x_tensor = Tensor(x, requires_grad=True)

        # Apply MaxPool2D
        kernel_size = 2
        stride = 2
        padding = 0

        output = MaxPool2DFunction.apply(x_tensor, kernel_size, stride, padding)

        # Expected output: max values in each 2x2 window
        expected = np.array([[6, 8], [14, 16]], dtype=np.float32).reshape(1, 1, 2, 2)

        self.assertEqual(output.shape, (1, 1, 2, 2))
        np.testing.assert_allclose(output.data, expected)

    def test_maxpool2d_backward(self):
        """Test the backward pass of MaxPool2D"""
        # Create a simple input
        x = np.array(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
            dtype=np.float32,
        ).reshape(1, 1, 4, 4)
        x_tensor = Tensor(x, requires_grad=True)

        # Apply MaxPool2D
        kernel_size = 2
        stride = 2
        padding = 0

        output = MaxPool2DFunction.apply(x_tensor, kernel_size, stride, padding)

        # Create a gradient for backprop
        grad_output = np.ones_like(output.data)

        # Backpropagate
        output.backward(Tensor(grad_output))

        # Verify gradient shape
        self.assertEqual(x_tensor.grad.shape, x_tensor.shape)

        # Expected gradient: 1 at positions of max values, 0 elsewhere
        expected_grad = np.zeros_like(x)
        expected_grad[0, 0, 1, 1] = 1  # Position of 6
        expected_grad[0, 0, 1, 3] = 1  # Position of 8
        expected_grad[0, 0, 3, 1] = 1  # Position of 14
        expected_grad[0, 0, 3, 3] = 1  # Position of 16

        np.testing.assert_allclose(x_tensor.grad.data, expected_grad)


if __name__ == "__main__":
    unittest.main()
