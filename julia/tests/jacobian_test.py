import numpy as np
import unittest
from julia.core.tensor import Tensor
from julia.core.math.jacobian import (
    JacobianVectorProduct,
    VectorJacobianProduct,
    compute_jacobian,
)


class TestJacobianMatrix(unittest.TestCase):
    def test_jvp_linear_function(self):
        """Test JVP for a linear function"""

        # Define a simple linear function
        def linear_func(x):
            return x * 2

        # Create input and vector
        x = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
        v = Tensor(np.array([1.0, 1.0, 1.0]))

        # Compute JVP
        jvp_result = JacobianVectorProduct.compute_jvp(linear_func, [x], [v])

        # Expected result: J * v = [2, 2, 2]
        expected = np.array([2.0, 2.0, 2.0])

        np.testing.assert_allclose(jvp_result.data, expected)

    def test_vjp_linear_function(self):
        """Test VJP for a linear function"""

        # Define a simple linear function
        def linear_func(x):
            return x * 2

        # Create input and vector
        x = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
        v = Tensor(np.array([1.0, 1.0, 1.0]))

        # Compute VJP
        vjp_results = VectorJacobianProduct.compute_vjp(linear_func, [x], v)

        # Expected result: v^T * J = [2, 2, 2]
        expected = np.array([2.0, 2.0, 2.0])

        np.testing.assert_allclose(vjp_results[0].data, expected)

    def test_jacobian_linear_function(self):
        """Test full Jacobian computation for a linear function"""

        # Define a simple linear function
        def linear_func(x):
            return x * 2

        # Create input
        x = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)

        # Compute Jacobian
        jacobians = compute_jacobian(linear_func, [x])

        # Expected Jacobian: diagonal matrix with 2 on the diagonal
        expected = np.diag([2.0, 2.0, 2.0])

        np.testing.assert_allclose(jacobians[0], expected)

    def test_jvp_nonlinear_function(self):
        """Test JVP for a nonlinear function"""

        # Define a nonlinear function: f(x) = x^2
        def nonlinear_func(x):
            return x * x

        # Create input and vector
        x = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
        v = Tensor(np.array([1.0, 1.0, 1.0]))

        # Compute JVP
        jvp_result = JacobianVectorProduct.compute_jvp(nonlinear_func, [x], [v])

        # Expected result: J * v = [2*x1, 2*x2, 2*x3] * v = [2, 4, 6]
        expected = np.array([2.0, 4.0, 6.0])

        np.testing.assert_allclose(jvp_result.data, expected)

    def test_vjp_nonlinear_function(self):
        """Test VJP for a nonlinear function"""

        # Define a nonlinear function: f(x) = x^2
        def nonlinear_func(x):
            return x * x

        # Create input and vector
        x = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
        v = Tensor(np.array([1.0, 1.0, 1.0]))

        # Compute VJP
        vjp_results = VectorJacobianProduct.compute_vjp(nonlinear_func, [x], v)

        # Expected result: v^T * J = [2*x1, 2*x2, 2*x3] = [2, 4, 6]
        expected = np.array([2.0, 4.0, 6.0])

        np.testing.assert_allclose(vjp_results[0].data, expected)

    def test_jacobian_nonlinear_function(self):
        """Test full Jacobian computation for a nonlinear function"""

        # Define a nonlinear function: f(x) = x^2
        def nonlinear_func(x):
            return x * x

        # Create input
        x = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)

        # Compute Jacobian
        jacobians = compute_jacobian(nonlinear_func, [x])

        # Expected Jacobian: diagonal matrix with 2*x on the diagonal
        expected = np.diag([2.0, 4.0, 6.0])

        np.testing.assert_allclose(jacobians[0], expected)

    def test_jvp_vector_to_scalar_function(self):
        """Test JVP for a function that maps a vector to a scalar"""

        # Define a function: f(x) = sum(x)
        def sum_func(x):
            return x.sum()

        # Create input and vector
        x = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
        v = Tensor(np.array([1.0, 1.0, 1.0]))

        # Compute JVP
        jvp_result = JacobianVectorProduct.compute_jvp(sum_func, [x], [v])

        # Expected result: J * v = [1, 1, 1] * [1, 1, 1] = 3
        expected = np.array(3.0)

        np.testing.assert_allclose(jvp_result.data, expected)

    def test_vjp_vector_to_scalar_function(self):
        """Test VJP for a function that maps a vector to a scalar"""

        # Define a function: f(x) = sum(x)
        def sum_func(x):
            return x.sum()

        # Create input and vector
        x = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
        v = Tensor(np.array(1.0))

        # Compute VJP
        vjp_results = VectorJacobianProduct.compute_vjp(sum_func, [x], v)

        # Expected result: v^T * J = 1 * [1, 1, 1] = [1, 1, 1]
        expected = np.array([1.0, 1.0, 1.0])

        np.testing.assert_allclose(vjp_results[0].data, expected)

    def test_jacobian_vector_to_scalar_function(self):
        """Test full Jacobian computation for a function that maps a vector to a scalar"""

        # Define a function: f(x) = sum(x)
        def sum_func(x):
            return x.sum()

        # Create input
        x = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)

        # Compute Jacobian
        jacobians = compute_jacobian(sum_func, [x])

        # Expected Jacobian: row vector of ones
        expected = np.array([1.0, 1.0, 1.0])

        np.testing.assert_allclose(jacobians[0], expected)

    def test_jvp_multiple_inputs(self):
        """Test JVP for a function with multiple inputs"""

        # Define a function: f(x, y) = x + y
        def add_func(x, y):
            return x + y

        # Create inputs and vectors
        x = Tensor(np.array([1.0, 2.0]), requires_grad=True)
        y = Tensor(np.array([3.0, 4.0]), requires_grad=True)
        vx = Tensor(np.array([1.0, 0.0]))
        vy = Tensor(np.array([0.0, 1.0]))

        # Compute JVP
        jvp_result = JacobianVectorProduct.compute_jvp(add_func, [x, y], [vx, vy])

        # Expected result: J_x * vx + J_y * vy = [1, 0] + [0, 1] = [1, 1]
        expected = np.array([1.0, 1.0])

        np.testing.assert_allclose(jvp_result.data, expected)

    def test_vjp_multiple_inputs(self):
        """Test VJP for a function with multiple inputs"""

        # Define a function: f(x, y) = x + y
        def add_func(x, y):
            return x + y

        # Create inputs and vector
        x = Tensor(np.array([1.0, 2.0]), requires_grad=True)
        y = Tensor(np.array([3.0, 4.0]), requires_grad=True)
        v = Tensor(np.array([1.0, 1.0]))

        # Compute VJP
        vjp_results = VectorJacobianProduct.compute_vjp(add_func, [x, y], v)

        # Expected results: v^T * J_x = [1, 1], v^T * J_y = [1, 1]
        expected_x = np.array([1.0, 1.0])
        expected_y = np.array([1.0, 1.0])

        np.testing.assert_allclose(vjp_results[0].data, expected_x)
        np.testing.assert_allclose(vjp_results[1].data, expected_y)

    def test_jacobian_multiple_inputs(self):
        """Test full Jacobian computation for a function with multiple inputs"""

        # Define a function: f(x, y) = x + y
        def add_func(x, y):
            return x + y

        # Create inputs
        x = Tensor(np.array([1.0, 2.0]), requires_grad=True)
        y = Tensor(np.array([3.0, 4.0]), requires_grad=True)

        # Compute Jacobians
        jacobians = compute_jacobian(add_func, [x, y])

        # Expected Jacobians: identity matrices
        expected_x = np.eye(2)
        expected_y = np.eye(2)

        np.testing.assert_allclose(jacobians[0], expected_x)
        np.testing.assert_allclose(jacobians[1], expected_y)

    def test_jvp_vector_valued_function(self):
        """Test JVP for a vector-valued function"""

        # Define a function: f(x) = [2*x, x^2]
        def vector_func(x):
            return Tensor(np.stack([2 * x.data, x.data**2], axis=0))

        # Create input and vector
        x = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
        v = Tensor(np.array([1.0, 1.0, 1.0]))

        # Compute JVP
        jvp_result = JacobianVectorProduct.compute_jvp(vector_func, [x], [v])

        # Expected result for first component: J_1 * v = [2, 2, 2] * [1, 1, 1] = 6
        # Expected result for second component: J_2 * v = [2, 4, 6] * [1, 1, 1] = 12
        expected = np.array([6.0, 12.0])

        np.testing.assert_allclose(jvp_result.data, expected)

    def test_vjp_vector_valued_function(self):
        """Test VJP for a vector-valued function"""

        # Define a function: f(x) = [2*x, x^2]
        def vector_func(x):
            return Tensor(np.stack([2 * x.data, x.data**2], axis=0))

        # Create input and vector
        x = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
        v = Tensor(np.array([1.0, 2.0]))  # Vector matching output dimensions

        # Compute VJP
        vjp_results = VectorJacobianProduct.compute_vjp(vector_func, [x], v)

        # Expected result: v^T * J = [1*2 + 2*2*x1, 1*2 + 2*2*x2, 1*2 + 2*2*x3] = [6, 10, 14]
        expected = np.array([6.0, 10.0, 14.0])

        np.testing.assert_allclose(vjp_results[0].data, expected)


if __name__ == "__main__":
    unittest.main()

    # Define a simple linear function
