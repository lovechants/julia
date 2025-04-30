import numpy as np
from julia.core.tensor import Tensor
from julia.core.nn.conv import Conv1D, Conv2D, Conv3D, TransposedConv2D, DepthwiseSeparableConv2D, DilatedConv2D
import unittest

class TestConvAutograd(unittest.TestCase):
    def test_conv1d_forward_backward(self):
        """Test forward and backward pass for Conv1D layer"""
        # Input: (batch_size, in_channels, sequence_length)
        batch_size, in_channels, seq_len = 2, 3, 10
        out_channels, kernel_size, stride, padding = 4, 3, 1, 1
        
        # Create input tensor
        x = Tensor(np.random.randn(batch_size, in_channels, seq_len), requires_grad=True)
        
        # Create Conv1D layer
        conv = Conv1D(in_channels, out_channels, kernel_size, stride, padding)
        
        # Compute output
        y = conv(x)
        
        # Ensure output has the correct shape
        expected_out_len = (seq_len + 2 * padding - kernel_size) // stride + 1
        self.assertEqual(y.shape, (batch_size, out_channels, expected_out_len))
        
        # Compute gradients with respect to a dummy loss
        # Use sum of output to simulate a scalar loss
        loss = y.sum()
        loss.backward()
        
        # Check that gradients are not None
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(conv.weight.grad)
        self.assertIsNotNone(conv.bias.grad)
        
        # Check gradient shapes
        self.assertEqual(x.grad.shape, x.shape)
        self.assertEqual(conv.weight.grad.shape, conv.weight.shape)
        self.assertEqual(conv.bias.grad.shape, conv.bias.shape)
        
        # Check that gradients are non-zero
        self.assertFalse(np.all(x.grad.data == 0))
        self.assertFalse(np.all(conv.weight.grad.data == 0))
        self.assertFalse(np.all(conv.bias.grad.data == 0))
    
    def test_conv2d_forward_backward(self):
        """Test forward and backward pass for Conv2D layer"""
        # Input: (batch_size, in_channels, height, width)
        batch_size, in_channels, height, width = 2, 3, 16, 16
        out_channels, kernel_size, stride, padding = 4, 3, 1, 1
        
        # Create input tensor
        x = Tensor(np.random.randn(batch_size, in_channels, height, width), requires_grad=True)
        
        # Create Conv2D layer
        conv = Conv2D(in_channels, out_channels, kernel_size, stride, padding)
        
        # Compute output
        y = conv(x)
        
        # Ensure output has the correct shape
        expected_h_out = (height + 2 * padding - kernel_size) // stride + 1
        expected_w_out = (width + 2 * padding - kernel_size) // stride + 1
        self.assertEqual(y.shape, (batch_size, out_channels, expected_h_out, expected_w_out))
        
        # Compute gradients with respect to a dummy loss
        loss = y.sum()
        loss.backward()
        
        # Check that gradients are not None
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(conv.weight.grad)
        self.assertIsNotNone(conv.bias.grad)
        
        # Check gradient shapes
        self.assertEqual(x.grad.shape, x.shape)
        self.assertEqual(conv.weight.grad.shape, conv.weight.shape)
        self.assertEqual(conv.bias.grad.shape, conv.bias.shape)
        
        # Check that gradients are non-zero
        self.assertFalse(np.all(x.grad.data == 0))
        self.assertFalse(np.all(conv.weight.grad.data == 0))
        self.assertFalse(np.all(conv.bias.grad.data == 0))
    
    def test_conv3d_forward_backward(self):
        """Test forward and backward pass for Conv3D layer"""
        # Input: (batch_size, in_channels, depth, height, width)
        batch_size, in_channels, depth, height, width = 2, 2, 8, 8, 8
        out_channels, kernel_size, stride, padding = 3, 3, 1, 1
        
        # Create input tensor
        x = Tensor(np.random.randn(batch_size, in_channels, depth, height, width), requires_grad=True)
        
        # Create Conv3D layer
        conv = Conv3D(in_channels, out_channels, kernel_size, stride, padding)
        
        # Compute output
        y = conv(x)
        
        # Ensure output has the correct shape
        expected_d_out = (depth + 2 * padding - kernel_size) // stride + 1
        expected_h_out = (height + 2 * padding - kernel_size) // stride + 1
        expected_w_out = (width + 2 * padding - kernel_size) // stride + 1
        self.assertEqual(y.shape, (batch_size, out_channels, expected_d_out, expected_h_out, expected_w_out))
        
        # Compute gradients with respect to a dummy loss
        loss = y.sum()
        loss.backward()
        
        # Check that gradients are not None
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(conv.weight.grad)
        self.assertIsNotNone(conv.bias.grad)
        
        # Check gradient shapes
        self.assertEqual(x.grad.shape, x.shape)
        self.assertEqual(conv.weight.grad.shape, conv.weight.shape)
        self.assertEqual(conv.bias.grad.shape, conv.bias.shape)
        
        # Check that gradients are non-zero
        self.assertFalse(np.all(x.grad.data == 0))
        self.assertFalse(np.all(conv.weight.grad.data == 0))
        self.assertFalse(np.all(conv.bias.grad.data == 0))
    
    def test_transposed_conv2d_forward_backward(self):
        """Test forward and backward pass for TransposedConv2D layer"""
        # Input: (batch_size, in_channels, height, width)
        batch_size, in_channels, height, width = 2, 3, 8, 8
        out_channels, kernel_size, stride, padding = 4, 3, 2, 1
        output_padding = 1
        
        # Create input tensor
        x = Tensor(np.random.randn(batch_size, in_channels, height, width), requires_grad=True)
        
        # Create TransposedConv2D layer
        conv = TransposedConv2D(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        
        # Compute output
        y = conv(x)
        
        # Ensure output has the correct shape
        expected_h_out = (height - 1) * stride - 2 * padding + kernel_size + output_padding
        expected_w_out = (width - 1) * stride - 2 * padding + kernel_size + output_padding
        self.assertEqual(y.shape, (batch_size, out_channels, expected_h_out, expected_w_out))
        
        # Compute gradients with respect to a dummy loss
        loss = y.sum()
        loss.backward()
        
        # Check that gradients are not None
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(conv.weight.grad)
        self.assertIsNotNone(conv.bias.grad)
        
        # Check gradient shapes
        self.assertEqual(x.grad.shape, x.shape)
        self.assertEqual(conv.weight.grad.shape, conv.weight.shape)
        self.assertEqual(conv.bias.grad.shape, conv.bias.shape)
        
        # Check that gradients are non-zero
        self.assertFalse(np.all(x.grad.data == 0))
        self.assertFalse(np.all(conv.weight.grad.data == 0))
        self.assertFalse(np.all(conv.bias.grad.data == 0))
    
    def test_depthwise_separable_conv2d_forward_backward(self):
        """Test forward and backward pass for DepthwiseSeparableConv2D layer"""
        # Input: (batch_size, in_channels, height, width)
        batch_size, in_channels, height, width = 2, 3, 16, 16
        out_channels, kernel_size, stride, padding = 4, 3, 1, 1
        depth_multiplier = 2
        
        # Create input tensor
        x = Tensor(np.random.randn(batch_size, in_channels, height, width), requires_grad=True)
        
        # Create DepthwiseSeparableConv2D layer
        conv = DepthwiseSeparableConv2D(in_channels, out_channels, kernel_size, stride, padding, depth_multiplier)
        
        # Compute output
        y = conv(x)
        
        # Ensure output has the correct shape
        expected_h_out = (height + 2 * padding - kernel_size) // stride + 1
        expected_w_out = (width + 2 * padding - kernel_size) // stride + 1
        self.assertEqual(y.shape, (batch_size, out_channels, expected_h_out, expected_w_out))
        
        # Compute gradients with respect to a dummy loss
        loss = y.sum()
        loss.backward()
        
        # Check that gradients are not None
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(conv.depthwise_weight.grad)
        self.assertIsNotNone(conv.pointwise_weight.grad)
        self.assertIsNotNone(conv.bias.grad)
        
        # Check gradient shapes
        self.assertEqual(x.grad.shape, x.shape)
        self.assertEqual(conv.depthwise_weight.grad.shape, conv.depthwise_weight.shape)
        self.assertEqual(conv.pointwise_weight.grad.shape, conv.pointwise_weight.shape)
        self.assertEqual(conv.bias.grad.shape, conv.bias.shape)
        
        # Check that gradients are non-zero
        self.assertFalse(np.all(x.grad.data == 0))
        self.assertFalse(np.all(conv.depthwise_weight.grad.data == 0))
        self.assertFalse(np.all(conv.pointwise_weight.grad.data == 0))
        self.assertFalse(np.all(conv.bias.grad.data == 0))
    
    def test_dilated_conv2d_forward_backward(self):
        """Test forward and backward pass for DilatedConv2D layer"""
        # Input: (batch_size, in_channels, height, width)
        batch_size, in_channels, height, width = 2, 3, 16, 16
        out_channels, kernel_size, stride, padding = 4, 3, 1, 2
        dilation = 2
        
        # Create input tensor
        x = Tensor(np.random.randn(batch_size, in_channels, height, width), requires_grad=True)
        
        # Create DilatedConv2D layer
        conv = DilatedConv2D(in_channels, out_channels, kernel_size, stride, padding, dilation)
        
        # Compute output
        y = conv(x)
        
        # Ensure output has the correct shape
        expected_h_out = (height + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
        expected_w_out = (width + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
        self.assertEqual(y.shape, (batch_size, out_channels, expected_h_out, expected_w_out))
        
        # Compute gradients with respect to a dummy loss
        loss = y.sum()
        loss.backward()
        
        # Check that gradients are not None
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(conv.weight.grad)
        self.assertIsNotNone(conv.bias.grad)
        
        # Check gradient shapes
        self.assertEqual(x.grad.shape, x.shape)
        self.assertEqual(conv.weight.grad.shape, conv.weight.shape)
        self.assertEqual(conv.bias.grad.shape, conv.bias.shape)
        
        # Check that gradients are non-zero
        self.assertFalse(np.all(x.grad.data == 0))
        self.assertFalse(np.all(conv.weight.grad.data == 0))
        self.assertFalse(np.all(conv.bias.grad.data == 0))

if __name__ == "__main__":
    unittest.main()
