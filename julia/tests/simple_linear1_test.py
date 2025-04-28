import numpy as np
import unittest
from julia.core.tensor import Tensor
from julia.core.nn.layers import Linear

class TestLinearLayer(unittest.TestCase):
    def test_linear_forward_backward(self):
        """Test the forward and backward pass of a Linear layer"""
        # Create a Linear layer
        in_features = 3
        out_features = 2
        linear = Linear(in_features, out_features)
        
        # Initialize weight and bias to small known values
        linear.weight = Tensor(np.ones((out_features, in_features)) * 0.1, requires_grad=True)
        linear.bias = Tensor(np.ones(out_features) * 0.1, requires_grad=True)
        
        # Create input
        batch_size = 2
        x = Tensor(np.ones((batch_size, in_features)), requires_grad=True)
        
        # Forward pass
        y = linear(x)
        
        # Verify output shape and values
        self.assertEqual(y.shape, (batch_size, out_features))
        
        # Expected output: 0.1 * (1+1+1) + 0.1 = 0.4 for each element
        expected_output = np.ones((batch_size, out_features)) * 0.4
        self.assertTrue(np.allclose(y.data, expected_output))
        
        # Create a very simple loss (sum of all outputs)
        loss = y.data.sum()
        
        # Initialize gradient as ones - this is simpler than using a loss function
        # Gradient shape should match y's shape
        y_grad = np.ones_like(y.data)
        
        # Debug info
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {y.shape}")
        print(f"Weight shape: {linear.weight.shape}")
        print(f"Bias shape: {linear.bias.shape}")
        print(f"Gradient shape: {y_grad.shape}")
        
        # Set gradient on y and call backward
        y.backward(Tensor(y_grad))
        
        # Check that gradients were computed
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(linear.weight.grad)
        self.assertIsNotNone(linear.bias.grad)
        
        # Print gradient shapes for debugging
        print(f"Input gradient shape: {x.grad.shape}")
        print(f"Weight gradient shape: {linear.weight.grad.shape}")
        print(f"Bias gradient shape: {linear.bias.grad.shape}")
        
        # Verify gradient shapes
        self.assertEqual(x.grad.shape, x.shape)
        self.assertEqual(linear.weight.grad.shape, linear.weight.shape)
        self.assertEqual(linear.bias.grad.shape, linear.bias.shape)
        
        # For this simple case, we can even verify the gradient values
        # x_grad = weight.T * y_grad
        expected_x_grad = np.ones((batch_size, in_features)) * 0.2  # Each output contributes 0.1 gradient
        self.assertTrue(np.allclose(x.grad.data, expected_x_grad))
        
        # weight_grad = x.T * y_grad
        expected_weight_grad = np.ones((out_features, in_features)) * batch_size  # Each batch contributes 1
        self.assertTrue(np.allclose(linear.weight.grad.data, expected_weight_grad))
        
        # bias_grad = sum(y_grad, dim=0)
        expected_bias_grad = np.ones(out_features) * batch_size  # Each batch contributes 1
        self.assertTrue(np.allclose(linear.bias.grad.data, expected_bias_grad))

if __name__ == '__main__':
    unittest.main()
