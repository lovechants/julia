import numpy as np
import unittest
from julia.core.tensor import Tensor
from julia.core.nn.layers import Linear


class TestLinearParameters(unittest.TestCase):
    '''
    def test_parameter_identity(self):
        """Test that parameters are properly maintained as identity objects"""
        linear = Linear(in_features=3, out_features=2)

        # Test initial identity
        self.assertTrue(linear.weight is linear.parameters()[0],
                       "Weight should be the same object as the first parameter")

        if linear.bias is not None:
            self.assertTrue(linear.bias is linear.parameters()[1],
                          "Bias should be the same object as the second parameter")

        # Set new weight data using proper method
        new_weight_data = np.ones((2, 3)) * 0.1
        linear.set_weight(new_weight_data)

        # Verify object identity is maintained
        self.assertTrue(linear.weight is linear.parameters()[0],
                      "Weight should still be the same object after update")

        # Also verify data was updated
        self.assertTrue(np.allclose(linear.weight.data, new_weight_data),
                      "Weight data should be updated")
        self.assertTrue(np.allclose(linear.parameters()[0].data, new_weight_data),
                      "Parameter data should be updated")
    '''

    def test_parameter_identity(self):
        linear = Linear(in_features=3, out_features=2)
        params_list = linear.parameters()  # Call the method

        # Check if weight object is in the returned list
        self.assertTrue(
            any(linear.weight is p for p in params_list),
            "Weight object not found in parameters list",
        )

        # Check bias if it exists
        if linear.bias is not None:
            self.assertTrue(
                any(linear.bias is p for p in params_list),
                "Bias object not found in parameters list",
            )

    def test_gradient_flow(self):
        """Test that gradients flow properly to parameters"""
        linear = Linear(in_features=2, out_features=1)

        # Set deterministic values for testing
        linear.set_weight(np.array([[1.0, 2.0]]))
        linear.set_bias(np.array([0.5]))

        # Input tensor
        x = Tensor(np.array([[3.0, 4.0]]), requires_grad=True)

        # Forward pass
        y = linear(x)

        # Expected output: [3*1 + 4*2 + 0.5] = [11.5]
        expected_output = np.array([[11.5]])
        self.assertTrue(
            np.allclose(y.data, expected_output),
            f"Expected output {expected_output}, got {y.data}",
        )

        # Set gradient directly on output
        y_grad = np.ones_like(y.data)
        y.backward(Tensor(y_grad))

        # Check weight gradient
        self.assertIsNotNone(linear.weight.grad, "Weight should have gradient")

        # Expected weight gradient for y = x @ W.T + b, with grad_y = 1:
        # grad_W = transpose(x.T @ grad_y)
        # Since grad_y is [[1.0]], this simplifies to transpose(x.T)
        # which is just x itself if x is a row vector.
        # Let's calculate it explicitly:
        # x.T = [[3.], [4.]] (shape 2, 1)
        # grad_y = [[1.]] (shape 1, 1)
        # x.T @ grad_y = [[3.], [4.]] (shape 2, 1)
        # transpose(x.T @ grad_y) = [[3., 4.]] (shape 1, 2)
        expected_weight_grad = np.array([[3.0, 4.0]])  # Shape (1, 2) matches W

        self.assertTrue(
            np.allclose(linear.weight.grad.data, expected_weight_grad),
            f"Expected weight grad {expected_weight_grad}, got {linear.weight.grad.data}",
        )

        # Check bias gradient (Optional but good)
        self.assertIsNotNone(linear.bias.grad, "Bias should have gradient")
        # Expected bias gradient: grad_y summed appropriately
        # Since bias is (1,) and grad_y is (1, 1), sum over batch dim (axis 0)
        expected_bias_grad = np.sum(y_grad, axis=0)  # Should be [1.0]
        self.assertTrue(
            np.allclose(linear.bias.grad.data, expected_bias_grad),
            f"Expected bias grad {expected_bias_grad}, got {linear.bias.grad.data}",
        )


if __name__ == "__main__":
    unittest.main()
