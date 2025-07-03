import numpy as np
import unittest
from julia.core.tensor import Tensor
from julia.core.ops_nn import (
    MaxPool2DFunction,
    AdaptiveMaxPool2DFunction,
    AvgPool2DFunction,
    GlobalAvgPool2DFunction,
    GlobalMaxPool2DFunction,
    AdaptiveAvgPool2DFunction,
)


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

    def test_avgpool2d_forward(self):
        """Test the forward pass of AvgPool2D"""
        # Create a simple input
        x = np.array(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
            dtype=np.float32,
        ).reshape(1, 1, 4, 4)
        x_tensor = Tensor(x, requires_grad=True)

        # Apply AvgPool2D
        kernel_size = 2
        stride = 2
        padding = 0

        output = AvgPool2DFunction.apply(x_tensor, kernel_size, stride, padding)

        # Expected output: average value in each 2x2 window
        expected = np.array(
            [
                [(1 + 2 + 5 + 6) / 4, (3 + 4 + 7 + 8) / 4],
                [(9 + 10 + 13 + 14) / 4, (11 + 12 + 15 + 16) / 4],
            ],
            dtype=np.float32,
        ).reshape(1, 1, 2, 2)

        self.assertEqual(output.shape, (1, 1, 2, 2))
        np.testing.assert_allclose(output.data, expected)

    def test_avgpool2d_backward(self):
        """Test the backward pass of AvgPool2D"""
        # Create a simple input
        x = np.array(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
            dtype=np.float32,
        ).reshape(1, 1, 4, 4)
        x_tensor = Tensor(x, requires_grad=True)

        # Apply AvgPool2D
        kernel_size = 2
        stride = 2
        padding = 0

        output = AvgPool2DFunction.apply(x_tensor, kernel_size, stride, padding)

        # Create a gradient for backprop
        grad_output = np.ones_like(output.data)

        # Backpropagate
        output.backward(Tensor(grad_output))

        # Verify gradient shape
        self.assertEqual(x_tensor.grad.shape, x_tensor.shape)

        # Expected gradient: 0.25 everywhere (1/4 for each position in the 2x2 window)
        expected_grad = np.ones_like(x) * 0.25

        np.testing.assert_allclose(x_tensor.grad.data, expected_grad)

    def test_adaptiveavgpool2d_forward(self):
        """Test the forward pass of AdaptiveAvgPool2D"""
        # Create a simple input
        x = np.array(
            [
                [1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10],
                [11, 12, 13, 14, 15],
                [16, 17, 18, 19, 20],
                [21, 22, 23, 24, 25],
            ],
            dtype=np.float32,
        ).reshape(1, 1, 5, 5)
        x_tensor = Tensor(x, requires_grad=True)

        # Apply AdaptiveAvgPool2D to get 2x2 output
        output_size = (2, 2)

        output = AdaptiveAvgPool2DFunction.apply(x_tensor, output_size)

        # Expected output: average value in each adaptive window
        # For 5x5 -> 2x2, each window covers either 2x2 or 3x3 elements
        # First window: [1,2,3, 6,7,8, 11,12,13] -> average = 7
        # Second window: [4,5, 9,10, 14,15] -> average = 9.5
        # Third window: [16,17,18, 21,22,23] -> average = 19.5
        # Fourth window: [19,20, 24,25] -> average = 22
        expected = np.array([[7, 9.5], [19.5, 22]], dtype=np.float32).reshape(
            1, 1, 2, 2
        )

        self.assertEqual(output.shape, (1, 1, 2, 2))
        np.testing.assert_allclose(output.data, expected)

    def test_adaptiveavgpool2d_backward(self):
        """Test the backward pass of AdaptiveAvgPool2D"""
        # Create a simple input
        x = np.array(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
            dtype=np.float32,
        ).reshape(1, 1, 4, 4)
        x_tensor = Tensor(x, requires_grad=True)

        # Apply AdaptiveAvgPool2D to get 2x2 output
        output_size = (2, 2)

        output = AdaptiveAvgPool2DFunction.apply(x_tensor, output_size)

        # Create a gradient for backprop
        grad_output = np.ones_like(output.data)

        # Backpropagate
        output.backward(Tensor(grad_output))

        # Verify gradient shape
        self.assertEqual(x_tensor.grad.shape, x_tensor.shape)

        # Expected gradient: 0.25 everywhere (1/4 for each position in the adaptive window)
        expected_grad = np.ones_like(x) * 0.25

        np.testing.assert_allclose(x_tensor.grad.data, expected_grad)

    def test_adaptivemaxpool2d_forward(self):
        """Test the forward pass of AdaptiveMaxPool2D"""
        # Create a simple input
        x = np.array(
            [
                [1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10],
                [11, 12, 13, 14, 15],
                [16, 17, 18, 19, 20],
                [21, 22, 23, 24, 25],
            ],
            dtype=np.float32,
        ).reshape(1, 1, 5, 5)
        x_tensor = Tensor(x, requires_grad=True)

        # Apply AdaptiveMaxPool2D to get 2x2 output
        output_size = (2, 2)

        output = AdaptiveMaxPool2DFunction.apply(x_tensor, output_size)

        # Expected output: max value in each adaptive window
        # For 5x5 -> 2x2, each window covers either 2x2 or 3x3 elements
        # First window: [1,2,3, 6,7,8, 11,12,13] -> max = 13
        # Second window: [4,5, 9,10, 14,15] -> max = 15
        # Third window: [16,17,18, 21,22,23] -> max = 23
        # Fourth window: [19,20, 24,25] -> max = 25
        expected = np.array([[13, 15], [23, 25]], dtype=np.float32).reshape(1, 1, 2, 2)

        self.assertEqual(output.shape, (1, 1, 2, 2))
        np.testing.assert_allclose(output.data, expected)

    def test_adaptivemaxpool2d_backward(self):
        """Test the backward pass of AdaptiveMaxPool2D"""
        # Create a simple input
        x = np.array(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
            dtype=np.float32,
        ).reshape(1, 1, 4, 4)
        x_tensor = Tensor(x, requires_grad=True)

        # Apply AdaptiveMaxPool2D to get 2x2 output
        output_size = (2, 2)

        output = AdaptiveMaxPool2DFunction.apply(x_tensor, output_size)

        # Create a gradient for backprop
        grad_output = np.ones_like(output.data)

        # Backpropagate
        output.backward(Tensor(grad_output))

        # Verify gradient shape
        self.assertEqual(x_tensor.grad.shape, x_tensor.shape)

        # Expected gradient: 1 at positions of max values, 0 elsewhere
        expected_grad = np.zeros_like(x)
        expected_grad[0, 0, 1, 1] = 1  # Position of 6 (max in top-left window)
        expected_grad[0, 0, 1, 3] = 1  # Position of 8 (max in top-right window)
        expected_grad[0, 0, 3, 1] = 1  # Position of 14 (max in bottom-left window)
        expected_grad[0, 0, 3, 3] = 1  # Position of 16 (max in bottom-right window)

        np.testing.assert_allclose(x_tensor.grad.data, expected_grad)

    def test_maxpool2d_with_padding(self):
        """Test MaxPool2D with padding"""
        # Create a simple input
        x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32).reshape(
            1, 1, 3, 3
        )
        x_tensor = Tensor(x, requires_grad=True)

        # Apply MaxPool2D with padding
        kernel_size = 2
        stride = 1
        padding = 1

        output = MaxPool2DFunction.apply(x_tensor, kernel_size, stride, padding)

        # Expected output shape with padding
        self.assertEqual(output.shape, (1, 1, 4, 4))

        # Verify a few values
        # Top-left corner with padding will have one valid value (1) and three -inf values
        self.assertEqual(output.data[0, 0, 0, 0], 1)

        # Center of the output should have the max of a 2x2 window
        self.assertEqual(output.data[0, 0, 1, 1], 5)
        self.assertEqual(output.data[0, 0, 1, 2], 6)
        self.assertEqual(output.data[0, 0, 2, 1], 8)
        self.assertEqual(output.data[0, 0, 2, 2], 9)

        # Test backward pass
        grad_output = np.ones_like(output.data)
        output.backward(Tensor(grad_output))

        # Verify gradient shape
        self.assertEqual(x_tensor.grad.shape, x_tensor.shape)

        # Each value in the input will be the maximum in at least one window
        # So each position should have a non-zero gradient
        self.assertTrue(np.all(x_tensor.grad.data > 0))

    def test_avgpool2d_with_padding(self):
        """Test AvgPool2D with padding"""
        # Create a simple input
        x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32).reshape(
            1, 1, 3, 3
        )
        x_tensor = Tensor(x, requires_grad=True)

        # Apply AvgPool2D with padding
        kernel_size = 2
        stride = 1
        padding = 1

        output = AvgPool2DFunction.apply(
            x_tensor, kernel_size, stride, padding, count_include_pad=True
        )

        # Expected output shape with padding
        self.assertEqual(output.shape, (1, 1, 4, 4))

        # Test backward pass
        grad_output = np.ones_like(output.data)
        output.backward(Tensor(grad_output))

        # Verify gradient shape
        self.assertEqual(x_tensor.grad.shape, x_tensor.shape)

        # With count_include_pad=True, gradient should be evenly distributed
        # Elements in the interior participate in more windows than elements at the border
        self.assertTrue(np.all(x_tensor.grad.data > 0))

        # Test with count_include_pad=False
        x_tensor.grad = None
        output = AvgPool2DFunction.apply(
            x_tensor, kernel_size, stride, padding, count_include_pad=False
        )
        output.backward(Tensor(grad_output))

        # Verify gradient with count_include_pad=False
        self.assertEqual(x_tensor.grad.shape, x_tensor.shape)
        self.assertTrue(np.all(x_tensor.grad.data > 0))

    def test_globalavgpool2d_forward(self):
        """Test the forward pass of GlobalAvgPool2D"""
        # Create a simple input
        x = np.array(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
            dtype=np.float32,
        ).reshape(1, 1, 4, 4)
        x_tensor = Tensor(x, requires_grad=True)

        # Apply GlobalAvgPool2D
        output = GlobalAvgPool2DFunction.apply(x_tensor)

        # Expected output: average of all values
        expected = np.array([[[[8.5]]]])  # (1+2+...+16)/16 = 8.5

        self.assertEqual(output.shape, (1, 1, 1, 1))
        np.testing.assert_allclose(output.data, expected)

    def test_globalavgpool2d_backward(self):
        """Test the backward pass of GlobalAvgPool2D"""
        # Create a simple input
        x = np.array(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
            dtype=np.float32,
        ).reshape(1, 1, 4, 4)
        x_tensor = Tensor(x, requires_grad=True)

        # Apply GlobalAvgPool2D
        output = GlobalAvgPool2DFunction.apply(x_tensor)

        # Create a gradient for backprop
        grad_output = np.ones_like(output.data)

        # Backpropagate
        output.backward(Tensor(grad_output))

        # Verify gradient shape
        self.assertEqual(x_tensor.grad.shape, x_tensor.shape)

        # Expected gradient: 1/16 everywhere (1/16 for each of the 16 positions)
        expected_grad = np.ones_like(x) * (1.0 / 16)

        np.testing.assert_allclose(x_tensor.grad.data, expected_grad)

    def test_globalmaxpool2d_forward(self):
        """Test the forward pass of GlobalMaxPool2D"""
        # Create a simple input
        x = np.array(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
            dtype=np.float32,
        ).reshape(1, 1, 4, 4)
        x_tensor = Tensor(x, requires_grad=True)

        # Apply GlobalMaxPool2D
        output = GlobalMaxPool2DFunction.apply(x_tensor)

        # Expected output: max value
        expected = np.array([[[[16.0]]]])

        self.assertEqual(output.shape, (1, 1, 1, 1))
        np.testing.assert_allclose(output.data, expected)

    def test_globalmaxpool2d_backward(self):
        """Test the backward pass of GlobalMaxPool2D"""
        # Create a simple input
        x = np.array(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
            dtype=np.float32,
        ).reshape(1, 1, 4, 4)
        x_tensor = Tensor(x, requires_grad=True)

        # Apply GlobalMaxPool2D
        output = GlobalMaxPool2DFunction.apply(x_tensor)

        # Create a gradient for backprop
        grad_output = np.ones_like(output.data)

        # Backpropagate
        output.backward(Tensor(grad_output))

        # Verify gradient shape
        self.assertEqual(x_tensor.grad.shape, x_tensor.shape)

        # Expected gradient: 1 at the position of the max value (16), 0 elsewhere
        expected_grad = np.zeros_like(x)
        expected_grad[0, 0, 3, 3] = 1.0  # Position of 16

        np.testing.assert_allclose(x_tensor.grad.data, expected_grad)

    def test_multiple_channels(self):
        """Test pooling with multiple channels"""
        # Create a multi-channel input
        x = np.random.randn(2, 3, 8, 8).astype(
            np.float32
        )  # 2 batches, 3 channels, 8x8 spatial
        x_tensor = Tensor(x, requires_grad=True)

        # Test each pooling operation
        pooling_ops = [
            (MaxPool2DFunction.apply, (2, 2, 0)),
            (AvgPool2DFunction.apply, (2, 2, 0, True)),
            (AdaptiveAvgPool2DFunction.apply, ((4, 4),)),
            (AdaptiveMaxPool2DFunction.apply, ((4, 4),)),
            (GlobalAvgPool2DFunction.apply, ()),
            (GlobalMaxPool2DFunction.apply, ()),
        ]

        for op, args in pooling_ops:
            # Apply pooling
            output = op(x_tensor, *args)

            # Verify shape
            if "Global" in op.__name__:
                self.assertEqual(output.shape[:2], (2, 3))
                self.assertEqual(output.shape[2:], (1, 1))
            elif "Adaptive" in op.__name__:
                self.assertEqual(output.shape, (2, 3, 4, 4))
            else:
                self.assertEqual(output.shape, (2, 3, 4, 4))

            # Test backward pass
            grad_output = np.ones_like(output.data)
            x_tensor.grad = None
            output.backward(Tensor(grad_output))

            # Verify gradient shape
            self.assertEqual(x_tensor.grad.shape, x_tensor.shape)


if __name__ == "__main__":
    unittest.main()
