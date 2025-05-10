import unittest
import numpy as np
from julia.core.tensor import Tensor

class TestRetainGraph(unittest.TestCase):
    """Test cases for detach and retain"""

    def test_detach(self):
        """Test detach creates a tensro disconnected from graph"""
        x = Tensor([2.0], requires_grad=True)
        y = x * 3 
        z = y.detach() * 2 

        # Backward should effect x -> y and not through z 
        sum_output = y + z 
        sum_output.backward()
        self.assertAlmostEqual(x.grad.data[0], 3.0)

        # Verify Z is detached 
        self.assertIsNone(z.grad)
        self.assertIsNone(z._backward_node)

    def test_retain_graph(self):
        """Test the retain method keeps the graph for multiple backward passes"""
        x = Tensor([2.0], requires_grad=True)
        y = x * x 

        # First backwards 
        y.backward(retain_graph=True)
        first_grad = x.grad.data.copy()
        print(f"First grad {first_grad}")

        # Should be able to clear the graident and run backwards again 
        x.zero_grad()
        y.backward()
        second_grad = x.grad.data.copy()
        print(f"Second grad {second_grad}\n")

        # Both gradients should be ~ dy/dx = 2x-4
        self.assertAlmostEqual(first_grad[0], 4.0)
        self.assertAlmostEqual(second_grad[0], 4.0)

