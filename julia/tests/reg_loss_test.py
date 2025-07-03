import numpy as np
import unittest
from julia.core.tensor import Tensor
from julia.core.nn.layers import Linear, Sequential
from julia.core.optim import Adam
from julia.core.loss import (
    mse_loss,
    mae_loss,
    huber_loss,
    log_cosh_loss,
    MSELoss,
)


class TestRegressionLosses(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.batch_size = 32
        self.input_dim = 10
        self.output_dim = 1

        # Create synthetic regression data
        X = np.random.randn(self.batch_size, self.input_dim)
        true_weights = np.random.randn(self.input_dim, self.output_dim)
        noise = np.random.randn(self.batch_size, self.output_dim) * 0.1
        y = X @ true_weights + noise

        self.X = Tensor(X, requires_grad=False)
        self.y_true = Tensor(y, requires_grad=False)

        # Create a simple model
        self.model = Sequential(
            Linear(self.input_dim, 32),
            lambda x: x.relu(),
            Linear(32, 16),
            lambda x: x.relu(),
            Linear(16, self.output_dim),
        )

        # Generate predictions
        self.y_pred = self.model(self.X)

    def test_mse_loss_forward(self):
        """Test MSE loss forward pass"""
        loss = mse_loss(self.y_pred, self.y_true)

        # Check that loss is a scalar
        self.assertEqual(loss.data.shape, ())

        # Check that loss is positive
        self.assertGreater(loss.data, 0)

        # Manual calculation
        diff = self.y_pred.data - self.y_true.data
        expected_loss = np.mean(diff**2)
        self.assertAlmostEqual(loss.data, expected_loss, places=6)

    def test_mse_loss_backward(self):
        """Test MSE loss backward pass"""
        loss = mse_loss(self.y_pred, self.y_true)
        loss.backward()

        # Check gradients exist
        for param in self.model.parameters():
            self.assertIsNotNone(param.grad)
            self.assertFalse(np.all(param.grad.data == 0))

    def test_mse_loss_reductions(self):
        """Test MSE loss with different reductions"""
        loss_none = MSELoss.apply(self.y_pred, self.y_true, "none")
        loss_mean = MSELoss.apply(self.y_pred, self.y_true, "mean")
        loss_sum = MSELoss.apply(self.y_pred, self.y_true, "sum")

        # Check shapes
        self.assertEqual(loss_none.shape, self.y_pred.shape)
        self.assertEqual(loss_mean.data.shape, ())
        self.assertEqual(loss_sum.data.shape, ())

        # Check relationships
        self.assertAlmostEqual(loss_mean.data, np.mean(loss_none.data), places=6)
        self.assertAlmostEqual(loss_sum.data, np.sum(loss_none.data), places=6)

    def test_mae_loss_forward(self):
        """Test MAE loss forward pass"""
        loss = mae_loss(self.y_pred, self.y_true)

        self.assertEqual(loss.data.shape, ())
        self.assertGreater(loss.data, 0)

        # Manual calculation
        diff = self.y_pred.data - self.y_true.data
        expected_loss = np.mean(np.abs(diff))
        self.assertAlmostEqual(loss.data, expected_loss, places=6)

    def test_mae_loss_backward(self):
        """Test MAE loss backward pass"""
        loss = mae_loss(self.y_pred, self.y_true)
        loss.backward()

        for param in self.model.parameters():
            self.assertIsNotNone(param.grad)

    def test_huber_loss_forward(self):
        """Test Huber loss forward pass"""
        delta = 1.0
        loss = huber_loss(self.y_pred, self.y_true, delta)

        self.assertEqual(loss.data.shape, ())
        self.assertGreater(loss.data, 0)

        # Manual calculation
        diff = self.y_pred.data - self.y_true.data
        abs_diff = np.abs(diff)
        quadratic = 0.5 * diff**2
        linear = delta * (abs_diff - 0.5 * delta)
        expected_loss = np.mean(np.where(abs_diff <= delta, quadratic, linear))
        self.assertAlmostEqual(loss.data, expected_loss, places=6)

    def test_huber_loss_different_deltas(self):
        """Test Huber loss with different delta values"""
        deltas = [0.1, 1.0, 5.0]
        losses = []

        for delta in deltas:
            loss = huber_loss(self.y_pred, self.y_true, delta)
            losses.append(loss.data)

        # Huber loss should vary with delta
        self.assertNotEqual(losses[0], losses[1])
        self.assertNotEqual(losses[1], losses[2])

    def test_log_cosh_loss_forward(self):
        """Test Log-Cosh loss forward pass"""
        loss = log_cosh_loss(self.y_pred, self.y_true)

        self.assertEqual(loss.data.shape, ())
        self.assertGreater(loss.data, 0)

        # Manual calculation
        diff = self.y_pred.data - self.y_true.data
        expected_loss = np.mean(np.log(np.cosh(diff)))
        self.assertAlmostEqual(loss.data, expected_loss, places=6)

    def test_log_cosh_loss_backward(self):
        """Test Log-Cosh loss backward pass"""
        loss = log_cosh_loss(self.y_pred, self.y_true)
        loss.backward()

        for param in self.model.parameters():
            self.assertIsNotNone(param.grad)

    def test_loss_comparison_with_outliers(self):
        """Test how different losses handle outliers"""
        # Create data with outliers
        y_clean = self.y_true.data.copy()
        y_outliers = y_clean.copy()
        y_outliers[0] += 10.0  # Add a large outlier

        y_true_outliers = Tensor(y_outliers)

        mse_clean = mse_loss(self.y_pred, self.y_true)
        mse_outliers = mse_loss(self.y_pred, y_true_outliers)

        mae_clean = mae_loss(self.y_pred, self.y_true)
        mae_outliers = mae_loss(self.y_pred, y_true_outliers)

        huber_clean = huber_loss(self.y_pred, self.y_true, delta=1.0)
        huber_outliers = huber_loss(self.y_pred, y_true_outliers, delta=1.0)

        # MSE should be most affected by outliers
        mse_ratio = mse_outliers.data / mse_clean.data
        mae_ratio = mae_outliers.data / mae_clean.data
        huber_ratio = huber_outliers.data / huber_clean.data

        # MSE should have the highest ratio (most sensitive to outliers)
        self.assertGreater(mse_ratio, mae_ratio)
        self.assertGreater(mse_ratio, huber_ratio)

    def test_gradient_magnitudes(self):
        """Test that gradients have reasonable magnitudes"""
        losses = [
            mse_loss(self.y_pred, self.y_true),
            mae_loss(self.y_pred, self.y_true),
            huber_loss(self.y_pred, self.y_true, delta=1.0),
            log_cosh_loss(self.y_pred, self.y_true),
        ]

        for loss in losses:
            # Clear gradients
            for param in self.model.parameters():
                param.zero_grad()

            loss.backward()

            # Check gradient magnitudes are reasonable
            for param in self.model.parameters():
                grad_norm = np.linalg.norm(param.grad.data)
                self.assertGreater(grad_norm, 1e-8)  # Not too small
                self.assertLess(grad_norm, 1e3)  # Not too large

    def test_training_convergence(self):
        """Test that models can train with different loss functions"""
        losses_to_test = [
            ("MSE", lambda pred, true: mse_loss(pred, true)),
            ("MAE", lambda pred, true: mae_loss(pred, true)),
            ("Huber", lambda pred, true: huber_loss(pred, true, delta=1.0)),
            ("LogCosh", lambda pred, true: log_cosh_loss(pred, true)),
        ]

        for loss_name, loss_fn in losses_to_test:
            # Create fresh model for each test
            model = Sequential(
                Linear(self.input_dim, 16),
                lambda x: x.relu(),
                Linear(16, self.output_dim),
            )

            optimizer = Adam(model.parameters(), lr=0.01)

            initial_loss = None
            final_loss = None

            # Train for a few epochs
            for epoch in range(20):
                optimizer.zero_grad()
                pred = model(self.X)
                loss = loss_fn(pred, self.y_true)

                if epoch == 0:
                    initial_loss = loss.data
                if epoch == 19:
                    final_loss = loss.data

                loss.backward()
                optimizer.step()

            # Loss should decrease
            self.assertLess(
                final_loss,
                initial_loss,
                f"{loss_name} loss did not decrease during training",
            )

    def test_loss_numerical_stability(self):
        """Test loss functions with extreme values"""
        # Test with very small values
        y_pred_small = Tensor(np.random.randn(10, 1) * 1e-6, requires_grad=True)
        y_true_small = Tensor(np.random.randn(10, 1) * 1e-6)

        # Test with very large values
        y_pred_large = Tensor(np.random.randn(10, 1) * 1e3, requires_grad=True)
        y_true_large = Tensor(np.random.randn(10, 1) * 1e3)

        test_cases = [
            (y_pred_small, y_true_small, "small values"),
            (y_pred_large, y_true_large, "large values"),
        ]

        for y_pred, y_true, case_name in test_cases:
            for loss_fn in [mse_loss, mae_loss, huber_loss, log_cosh_loss]:
                try:
                    loss = loss_fn(y_pred, y_true)

                    # Check for NaN or Inf
                    self.assertFalse(
                        np.isnan(loss.data),
                        f"{loss_fn.__name__} produced NaN with {case_name}",
                    )
                    self.assertFalse(
                        np.isinf(loss.data),
                        f"{loss_fn.__name__} produced Inf with {case_name}",
                    )

                    # Test backward pass
                    y_pred.zero_grad()
                    loss.backward()

                    self.assertFalse(
                        np.any(np.isnan(y_pred.grad.data)),
                        f"{loss_fn.__name__} produced NaN gradients with {case_name}",
                    )

                except Exception as e:
                    self.fail(f"{loss_fn.__name__} failed with {case_name}: {e}")

    def test_batch_independence(self):
        """Test that loss computation is independent across batch samples"""
        # Compute loss for full batch
        full_loss = mse_loss(self.y_pred, self.y_true, reduction="none")

        # Compute loss for individual samples
        individual_losses = []
        for i in range(self.batch_size):
            pred_sample = Tensor(self.y_pred.data[i : i + 1])
            true_sample = Tensor(self.y_true.data[i : i + 1])
            loss_sample = mse_loss(pred_sample, true_sample, reduction="none")
            individual_losses.append(loss_sample.data[0])

        # Compare
        np.testing.assert_allclose(
            full_loss.data.flatten(), individual_losses, rtol=1e-6
        )


if __name__ == "__main__":
    unittest.main()
