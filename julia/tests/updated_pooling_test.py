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
        x = np.array(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
            dtype=np.float32,
        ).reshape(1, 1, 4, 4)
        x_tensor = Tensor(x, requires_grad=True)

        kernel_size = 2
        stride = 2
        padding = 0

        output = MaxPool2DFunction.apply(x_tensor, kernel_size, stride, padding)

        expected = np.array([[6, 8], [14, 16]], dtype=np.float32).reshape(1, 1, 2, 2)

        self.assertEqual(output.shape, (1, 1, 2, 2))
        np.testing.assert_allclose(output.data, expected)

    def test_maxpool2d_backward(self):
        """Test the backward pass of MaxPool2D"""
        x = np.array(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
            dtype=np.float32,
        ).reshape(1, 1, 4, 4)
        x_tensor = Tensor(x, requires_grad=True)

        kernel_size = 2
        stride = 2
        padding = 0

        output = MaxPool2DFunction.apply(x_tensor, kernel_size, stride, padding)
        grad_output = np.ones_like(output.data)
        output.backward(Tensor(grad_output))

        self.assertEqual(x_tensor.grad.shape, x_tensor.shape)

        expected_grad = np.zeros_like(x)
        expected_grad[0, 0, 1, 1] = 1
        expected_grad[0, 0, 1, 3] = 1
        expected_grad[0, 0, 3, 1] = 1
        expected_grad[0, 0, 3, 3] = 1

        np.testing.assert_allclose(x_tensor.grad.data, expected_grad)

    def test_avgpool2d_forward(self):
        """Test the forward pass of AvgPool2D"""
        x = np.array(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
            dtype=np.float32,
        ).reshape(1, 1, 4, 4)
        x_tensor = Tensor(x, requires_grad=True)

        kernel_size = 2
        stride = 2
        padding = 0

        output = AvgPool2DFunction.apply(x_tensor, kernel_size, stride, padding)

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
        x = np.array(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
            dtype=np.float32,
        ).reshape(1, 1, 4, 4)
        x_tensor = Tensor(x, requires_grad=True)

        kernel_size = 2
        stride = 2
        padding = 0

        output = AvgPool2DFunction.apply(x_tensor, kernel_size, stride, padding)
        grad_output = np.ones_like(output.data)
        output.backward(Tensor(grad_output))

        self.assertEqual(x_tensor.grad.shape, x_tensor.shape)

        expected_grad = np.ones_like(x) * 0.25

        np.testing.assert_allclose(x_tensor.grad.data, expected_grad)

    def test_adaptiveavgpool2d_forward(self):
        """Test the forward pass of AdaptiveAvgPool2D"""
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

        output_size = (2, 2)

        output = AdaptiveAvgPool2DFunction.apply(x_tensor, output_size)

        # Expected output matches PyTorch AdaptiveAvgPool2d behavior
        # Window [0,0]: rows 0-2, cols 0-2 -> average = 7
        # Window [0,1]: rows 0-2, cols 3-4 -> average = 9  
        # Window [1,0]: rows 3-4, cols 0-2 -> average = 17
        # Window [1,1]: rows 3-4, cols 3-4 -> average = 19
        expected = np.array([[7, 9], [17, 19]], dtype=np.float32).reshape(1, 1, 2, 2)

        self.assertEqual(output.shape, (1, 1, 2, 2))
        np.testing.assert_allclose(output.data, expected)

    def test_adaptiveavgpool2d_backward(self):
        """Test the backward pass of AdaptiveAvgPool2D"""
        x = np.array(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
            dtype=np.float32,
        ).reshape(1, 1, 4, 4)
        x_tensor = Tensor(x, requires_grad=True)

        output_size = (2, 2)

        output = AdaptiveAvgPool2DFunction.apply(x_tensor, output_size)
        grad_output = np.ones_like(output.data)
        output.backward(Tensor(grad_output))

        self.assertEqual(x_tensor.grad.shape, x_tensor.shape)

        # For 4x4 -> 2x2, each window is exactly 2x2, so gradient is 0.25 everywhere
        expected_grad = np.ones_like(x) * 0.25

        np.testing.assert_allclose(x_tensor.grad.data, expected_grad)

    def test_adaptivemaxpool2d_forward(self):
        """Test the forward pass of AdaptiveMaxPool2D"""
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

        output_size = (2, 2)

        output = AdaptiveMaxPool2DFunction.apply(x_tensor, output_size)

        # Expected output matches PyTorch
        expected = np.array([[13, 15], [23, 25]], dtype=np.float32).reshape(1, 1, 2, 2)

        self.assertEqual(output.shape, (1, 1, 2, 2))
        np.testing.assert_allclose(output.data, expected)

    def test_adaptivemaxpool2d_backward(self):
        """Test the backward pass of AdaptiveMaxPool2D"""
        x = np.array(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
            dtype=np.float32,
        ).reshape(1, 1, 4, 4)
        x_tensor = Tensor(x, requires_grad=True)

        output_size = (2, 2)

        output = AdaptiveMaxPool2DFunction.apply(x_tensor, output_size)
        grad_output = np.ones_like(output.data)
        output.backward(Tensor(grad_output))

        self.assertEqual(x_tensor.grad.shape, x_tensor.shape)

        expected_grad = np.zeros_like(x)
        expected_grad[0, 0, 1, 1] = 1
        expected_grad[0, 0, 1, 3] = 1
        expected_grad[0, 0, 3, 1] = 1
        expected_grad[0, 0, 3, 3] = 1

        np.testing.assert_allclose(x_tensor.grad.data, expected_grad)

    def test_maxpool2d_with_padding(self):
        """Test MaxPool2D with padding"""
        x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32).reshape(
            1, 1, 3, 3
        )
        x_tensor = Tensor(x, requires_grad=True)

        kernel_size = 2
        stride = 1
        padding = 1

        output = MaxPool2DFunction.apply(x_tensor, kernel_size, stride, padding)

        self.assertEqual(output.shape, (1, 1, 4, 4))

        self.assertEqual(output.data[0, 0, 0, 0], 1)
        self.assertEqual(output.data[0, 0, 1, 1], 5)
        self.assertEqual(output.data[0, 0, 1, 2], 6)
        self.assertEqual(output.data[0, 0, 2, 1], 8)
        self.assertEqual(output.data[0, 0, 2, 2], 9)

        grad_output = np.ones_like(output.data)
        output.backward(Tensor(grad_output))

        self.assertEqual(x_tensor.grad.shape, x_tensor.shape)
        self.assertTrue(np.all(x_tensor.grad.data > 0))

    def test_avgpool2d_with_padding(self):
        """Test AvgPool2D with padding"""
        x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32).reshape(
            1, 1, 3, 3
        )
        x_tensor = Tensor(x, requires_grad=True)

        kernel_size = 2
        stride = 1
        padding = 1

        output = AvgPool2DFunction.apply(
            x_tensor, kernel_size, stride, padding, count_include_pad=True
        )

        self.assertEqual(output.shape, (1, 1, 4, 4))

        grad_output = np.ones_like(output.data)
        output.backward(Tensor(grad_output))

        self.assertEqual(x_tensor.grad.shape, x_tensor.shape)
        self.assertTrue(np.all(x_tensor.grad.data > 0))

        x_tensor.grad = None
        output = AvgPool2DFunction.apply(
            x_tensor, kernel_size, stride, padding, count_include_pad=False
        )
        output.backward(Tensor(grad_output))

        self.assertEqual(x_tensor.grad.shape, x_tensor.shape)
        self.assertTrue(np.all(x_tensor.grad.data > 0))

    def test_globalavgpool2d_forward(self):
        """Test the forward pass of GlobalAvgPool2D"""
        x = np.array(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
            dtype=np.float32,
        ).reshape(1, 1, 4, 4)
        x_tensor = Tensor(x, requires_grad=True)

        output = GlobalAvgPool2DFunction.apply(x_tensor)

        expected = np.array([[[[8.5]]]])

        self.assertEqual(output.shape, (1, 1, 1, 1))
        np.testing.assert_allclose(output.data, expected)

    def test_globalavgpool2d_backward(self):
        """Test the backward pass of GlobalAvgPool2D"""
        x = np.array(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
            dtype=np.float32,
        ).reshape(1, 1, 4, 4)
        x_tensor = Tensor(x, requires_grad=True)

        output = GlobalAvgPool2DFunction.apply(x_tensor)
        grad_output = np.ones_like(output.data)
        output.backward(Tensor(grad_output))

        self.assertEqual(x_tensor.grad.shape, x_tensor.shape)

        expected_grad = np.ones_like(x) * (1.0 / 16)

        np.testing.assert_allclose(x_tensor.grad.data, expected_grad)

    def test_globalmaxpool2d_forward(self):
        """Test the forward pass of GlobalMaxPool2D"""
        x = np.array(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
            dtype=np.float32,
        ).reshape(1, 1, 4, 4)
        x_tensor = Tensor(x, requires_grad=True)

        output = GlobalMaxPool2DFunction.apply(x_tensor)

        expected = np.array([[[[16.0]]]])

        self.assertEqual(output.shape, (1, 1, 1, 1))
        np.testing.assert_allclose(output.data, expected)

    def test_globalmaxpool2d_backward(self):
        """Test the backward pass of GlobalMaxPool2D"""
        x = np.array(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
            dtype=np.float32,
        ).reshape(1, 1, 4, 4)
        x_tensor = Tensor(x, requires_grad=True)

        output = GlobalMaxPool2DFunction.apply(x_tensor)
        grad_output = np.ones_like(output.data)
        output.backward(Tensor(grad_output))

        self.assertEqual(x_tensor.grad.shape, x_tensor.shape)

        expected_grad = np.zeros_like(x)
        expected_grad[0, 0, 3, 3] = 1.0

        np.testing.assert_allclose(x_tensor.grad.data, expected_grad)

    def test_multiple_channels(self):
        """Test pooling with multiple channels"""
        x = np.random.randn(2, 3, 8, 8).astype(np.float32)
        x_tensor = Tensor(x, requires_grad=True)

        pooling_ops = [
            (MaxPool2DFunction.apply, (2, 2, 0)),
            (AvgPool2DFunction.apply, (2, 2, 0, True)),
            (AdaptiveAvgPool2DFunction.apply, ((4, 4),)),  # Keep nested tuple
            (AdaptiveMaxPool2DFunction.apply, ((4, 4),)),  # Keep nested tuple
            (GlobalAvgPool2DFunction.apply, ()),
            (GlobalMaxPool2DFunction.apply, ()),
        ]

        for op, args in pooling_ops:
            print(f"\nTesting: {op.__name__}")
            output = op(x_tensor, *args)
            print(f"Output shape: {output.shape}")
            
            op_class_name = op.__self__.__name__ if hasattr(op, '__self__') else op.__qualname__.split('.')[0]
            print(f"Op class: {op_class_name}")
            
            if "Global" in op_class_name:
                self.assertEqual(output.shape[:2], (2, 3))
                self.assertEqual(output.shape[2:], (1, 1))
            elif "Adaptive" in op_class_name:
                self.assertEqual(output.shape, (2, 3, 4, 4))
            else:
                self.assertEqual(output.shape, (2, 3, 4, 4))

            grad_output = np.ones_like(output.data)
            x_tensor.grad = None
            output.backward(Tensor(grad_output))

            self.assertEqual(x_tensor.grad.shape, x_tensor.shape)

if __name__ == "__main__":
    unittest.main()
