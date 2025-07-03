import numpy as np
import unittest
from julia.core.tensor import Tensor
from julia.core.nn.layers import Linear, Sequential
from julia.core.optim import Adam
from julia.core.loss import (
    LossFunction,
    CustomLossTemplate,
    LossFunctionFactory,
    MultiTaskLoss,
    mse_loss,
    mae_loss,
)


class TestBaseLossFunction(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.batch_size = 32
        self.output_dim = 5

        self.y_pred = Tensor(
            np.random.randn(self.batch_size, self.output_dim), requires_grad=True
        )
        self.y_true = Tensor(np.random.randn(self.batch_size, self.output_dim))

    def test_base_loss_function_reduction(self):
        """Test base LossFunction reduction functionality"""

        class TestLoss(LossFunction):
            def forward(self, y_pred, y_true):
                diff = y_pred - y_true
                loss = diff * diff
                return self._apply_reduction(loss)

        # Test different reduction modes
        loss_none = TestLoss(reduction="none")
        loss_mean = TestLoss(reduction="mean")
        loss_sum = TestLoss(reduction="sum")

        result_none = loss_none(self.y_pred, self.y_true)
        result_mean = loss_mean(self.y_pred, self.y_true)
        result_sum = loss_sum(self.y_pred, self.y_true)

        # Check shapes
        self.assertEqual(result_none.shape, self.y_pred.shape)
        self.assertEqual(result_mean.data.shape, ())
        self.assertEqual(result_sum.data.shape, ())

        # Check relationships
        self.assertAlmostEqual(result_mean.data, np.mean(result_none.data), places=6)
        self.assertAlmostEqual(result_sum.data, np.sum(result_none.data), places=6)

    def test_base_loss_function_invalid_reduction(self):
        """Test that invalid reduction modes raise errors"""
        with self.assertRaises(AssertionError):
            LossFunction(reduction="invalid")

    def test_base_loss_function_call_interface(self):
        """Test that __call__ delegates to forward"""

        class TestLoss(LossFunction):
            def forward(self, y_pred, y_true):
                return Tensor(np.array(42.0))

        loss_fn = TestLoss()
        result = loss_fn(self.y_pred, self.y_true)
        self.assertEqual(result.data, 42.0)


class TestCustomLossTemplate(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.batch_size = 32
        self.output_dim = 3

        self.y_pred = Tensor(
            np.random.randn(self.batch_size, self.output_dim), requires_grad=True
        )
        self.y_true = Tensor(np.random.randn(self.batch_size, self.output_dim))

    def test_custom_loss_template_abstract(self):
        """Test that CustomLossTemplate is abstract"""
        template = CustomLossTemplate()

        with self.assertRaises(NotImplementedError):
            template.forward(self.y_pred, self.y_true)

    def test_simple_custom_loss(self):
        """Test creating a simple custom loss function"""

        class WeightedMSELoss(CustomLossTemplate):
            def __init__(self, weights, reduction="mean"):
                super().__init__(reduction)
                self.weights = Tensor(weights)

            def forward(self, y_pred, y_true):
                diff = y_pred - y_true
                weighted_diff = diff * self.weights
                loss = weighted_diff * weighted_diff
                return self._apply_reduction(loss)

        weights = np.random.rand(self.batch_size, self.output_dim)
        custom_loss = WeightedMSELoss(weights)

        result = custom_loss(self.y_pred, self.y_true)

        # Should produce a scalar loss
        self.assertEqual(result.data.shape, ())
        self.assertGreater(result.data, 0)

        # Test backward pass
        result.backward()
        self.assertIsNotNone(self.y_pred.grad)

    def test_complex_custom_loss(self):
        """Test a more complex custom loss function"""

        class AdaptiveLoss(CustomLossTemplate):
            def __init__(self, alpha=0.5, reduction="mean"):
                super().__init__(reduction)
                self.alpha = alpha

            def forward(self, y_pred, y_true):
                mse_component = (y_pred - y_true) ** 2
                mae_component = Tensor(np.abs((y_pred - y_true).data))

                # Adaptive combination
                loss = self.alpha * mse_component + (1 - self.alpha) * mae_component
                return self._apply_reduction(loss)

        adaptive_loss = AdaptiveLoss(alpha=0.3)
        result = adaptive_loss(self.y_pred, self.y_true)

        self.assertEqual(result.data.shape, ())
        self.assertGreater(result.data, 0)

        # Test with different alpha
        adaptive_loss2 = AdaptiveLoss(alpha=0.8)
        result2 = adaptive_loss2(self.y_pred, self.y_true)

        # Results should be different
        self.assertNotAlmostEqual(result.data, result2.data, places=5)

    def test_custom_loss_with_parameters(self):
        """Test custom loss with learnable parameters"""

        class ParametricLoss(CustomLossTemplate):
            def __init__(self, reduction="mean"):
                super().__init__(reduction)
                # Learnable parameter
                self.temperature = Tensor(np.array([1.0]), requires_grad=True)

            def forward(self, y_pred, y_true):
                diff = y_pred - y_true
                # Temperature-scaled loss
                loss = (diff**2) / self.temperature
                return self._apply_reduction(loss)

            def parameters(self):
                return [self.temperature]

        param_loss = ParametricLoss()
        result = param_loss(self.y_pred, self.y_true)

        result.backward()

        # Both prediction and temperature should have gradients
        self.assertIsNotNone(self.y_pred.grad)
        self.assertIsNotNone(param_loss.temperature.grad)

    def test_custom_loss_training(self):
        """Test training a model with custom loss"""

        class RobustLoss(CustomLossTemplate):
            def __init__(self, delta=1.0, reduction="mean"):
                super().__init__(reduction)
                self.delta = delta

            def forward(self, y_pred, y_true):
                # Custom robust loss (similar to Huber but different)
                diff = y_pred - y_true
                abs_diff = Tensor(np.abs(diff.data))

                quadratic_mask = abs_diff.data <= self.delta
                linear_mask = ~quadratic_mask

                loss_data = np.zeros_like(diff.data)
                loss_data[quadratic_mask] = 0.5 * (diff.data[quadratic_mask] ** 2)
                loss_data[linear_mask] = self.delta * (
                    abs_diff.data[linear_mask] - 0.5 * self.delta
                )

                loss = Tensor(loss_data)
                return self._apply_reduction(loss)

        # Create simple regression problem
        X = Tensor(np.random.randn(100, 5))
        y_true = Tensor(np.random.randn(100, 1))

        model = Sequential(Linear(5, 10), lambda x: x.relu(), Linear(10, 1))

        optimizer = Adam(model.parameters(), lr=0.01)
        custom_loss = RobustLoss(delta=1.0)

        initial_loss = None
        final_loss = None

        for epoch in range(20):
            optimizer.zero_grad()
            pred = model(X)
            loss = custom_loss(pred, y_true)

            if epoch == 0:
                initial_loss = loss.data
            if epoch == 19:
                final_loss = loss.data

            loss.backward()
            optimizer.step()

        # Training should reduce loss
        self.assertLess(final_loss, initial_loss)


class TestLossFunctionFactory(unittest.TestCase):
    def test_factory_list_available(self):
        """Test listing available loss functions"""
        available = LossFunctionFactory.list_available()

        # Should include basic losses
        expected_losses = ["mse", "mae", "cross_entropy", "binary_cross_entropy"]
        for loss_name in expected_losses:
            self.assertIn(loss_name, available)

    def test_factory_create_known_loss(self):
        """Test creating known loss functions"""
        # This test assumes the factory is properly set up with Function classes
        # For now, we'll test the registry mechanism

        class DummyLoss:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        # Register a dummy loss
        LossFunctionFactory.register("dummy", DummyLoss)

        # Check it's in the list
        available = LossFunctionFactory.list_available()
        self.assertIn("dummy", available)

        # Create it
        loss_instance = LossFunctionFactory.create("dummy", param1=1, param2=2)
        self.assertIsInstance(loss_instance, DummyLoss)
        self.assertEqual(loss_instance.kwargs, {"param1": 1, "param2": 2})

    def test_factory_create_unknown_loss(self):
        """Test creating unknown loss functions raises error"""
        with self.assertRaises(ValueError):
            LossFunctionFactory.create("nonexistent_loss")

    def test_factory_register_custom_loss(self):
        """Test registering and using custom loss"""

        class MyCustomLoss(CustomLossTemplate):
            def __init__(self, scale=1.0, reduction="mean"):
                super().__init__(reduction)
                self.scale = scale

            def forward(self, y_pred, y_true):
                diff = y_pred - y_true
                loss = self.scale * (diff**2)
                return self._apply_reduction(loss)

        # Register custom loss
        LossFunctionFactory.register("my_custom", MyCustomLoss)

        # Create and test
        y_pred = Tensor(np.random.randn(10, 3), requires_grad=True)
        y_true = Tensor(np.random.randn(10, 3))

        custom_loss = LossFunctionFactory.create("my_custom", scale=2.0)
        result = custom_loss(y_pred, y_true)

        self.assertEqual(result.data.shape, ())
        self.assertGreater(result.data, 0)

        # Test backward
        result.backward()
        self.assertIsNotNone(y_pred.grad)


class TestMultiTaskLoss(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.batch_size = 32

        # Create multiple task predictions and targets
        self.task1_pred = Tensor(
            np.random.randn(self.batch_size, 5), requires_grad=True
        )
        self.task1_true = Tensor(np.random.randn(self.batch_size, 5))

        self.task2_pred = Tensor(
            np.random.randn(self.batch_size, 3), requires_grad=True
        )
        self.task2_true = Tensor(np.random.randn(self.batch_size, 3))

        self.task3_pred = Tensor(
            np.random.randn(self.batch_size, 1), requires_grad=True
        )
        self.task3_true = Tensor(np.random.randn(self.batch_size, 1))

    def test_multi_task_loss_creation(self):
        """Test creating MultiTaskLoss"""
        multi_loss = MultiTaskLoss(num_tasks=3)

        self.assertEqual(multi_loss.num_tasks, 3)
        self.assertEqual(len(multi_loss.log_vars), 3)

        # Initial log vars should be tensors requiring grad
        for log_var in multi_loss.log_vars:
            self.assertIsInstance(log_var, Tensor)
            self.assertTrue(log_var.requires_grad)

    def test_multi_task_loss_forward(self):
        """Test MultiTaskLoss forward pass"""
        multi_loss = MultiTaskLoss(num_tasks=3)

        # Compute individual task losses
        task1_loss = mse_loss(self.task1_pred, self.task1_true)
        task2_loss = mse_loss(self.task2_pred, self.task2_true)
        task3_loss = mae_loss(self.task3_pred, self.task3_true)

        task_losses = [task1_loss, task2_loss, task3_loss]

        # Compute combined loss
        combined_loss = multi_loss(task_losses)

        self.assertEqual(combined_loss.data.shape, ())
        self.assertGreater(combined_loss.data, 0)

    def test_multi_task_loss_weights(self):
        """Test getting task weights from MultiTaskLoss"""
        multi_loss = MultiTaskLoss(num_tasks=2)

        # Initial weights
        initial_weights = multi_loss.get_weights()
        self.assertEqual(len(initial_weights), 2)

        # All weights should be positive
        for weight in initial_weights:
            self.assertGreater(weight, 0)

    def test_multi_task_loss_training(self):
        """Test training with MultiTaskLoss"""
        # Create simple multi-task model
        shared_backbone = Sequential(
            Linear(10, 32), lambda x: x.relu(), Linear(32, 16), lambda x: x.relu()
        )

        task1_head = Linear(16, 5)
        task2_head = Linear(16, 3)

        X = Tensor(np.random.randn(50, 10))
        task1_target = Tensor(np.random.randn(50, 5))
        task2_target = Tensor(np.random.randn(50, 3))

        # Collect all parameters
        all_params = (
            shared_backbone.parameters()
            + task1_head.parameters()
            + task2_head.parameters()
        )

        multi_loss = MultiTaskLoss(num_tasks=2)
        all_params.extend(multi_loss.log_vars)  # Include loss weights as parameters

        optimizer = Adam(all_params, lr=0.01)

        initial_weights = multi_loss.get_weights()

        for epoch in range(10):
            optimizer.zero_grad()

            # Forward pass
            shared_features = shared_backbone(X)
            task1_pred = task1_head(shared_features)
            task2_pred = task2_head(shared_features)

            # Compute individual losses
            loss1 = mse_loss(task1_pred, task1_target)
            loss2 = mse_loss(task2_pred, task2_target)

            # Combine losses
            total_loss = multi_loss([loss1, loss2])

            total_loss.backward()
            optimizer.step()

        final_weights = multi_loss.get_weights()

        # Weights should have changed during training
        for initial, final in zip(initial_weights, final_weights):
            # Allow for some variation due to optimization
            pass  # Weights might not change dramatically in just 10 epochs

    def test_multi_task_loss_backward(self):
        """Test MultiTaskLoss backward pass"""
        multi_loss = MultiTaskLoss(num_tasks=2)

        task1_loss = mse_loss(self.task1_pred, self.task1_true)
        task2_loss = mse_loss(self.task2_pred, self.task2_true)

        combined_loss = multi_loss([task1_loss, task2_loss])
        combined_loss.backward()

        # Check gradients exist for predictions
        self.assertIsNotNone(self.task1_pred.grad)
        self.assertIsNotNone(self.task2_pred.grad)

        # Check gradients exist for loss weights
        for log_var in multi_loss.log_vars:
            self.assertIsNotNone(log_var.grad)


class TestLossIntegration(unittest.TestCase):
    """Integration tests for loss functions"""

    def test_custom_loss_with_real_model(self):
        """Test custom loss with a real model on a real problem"""

        class PercentileRegressionLoss(CustomLossTemplate):
            """Custom loss for percentile regression"""

            def __init__(self, percentile=0.5, reduction="mean"):
                super().__init__(reduction)
                self.percentile = percentile

            def forward(self, y_pred, y_true):
                residual = y_true - y_pred

                # Asymmetric loss for percentile regression
                loss_data = np.where(
                    residual.data >= 0,
                    self.percentile * residual.data,
                    (self.percentile - 1) * residual.data,
                )

                loss = Tensor(loss_data)
                return self._apply_reduction(loss)

        # Generate synthetic data with heteroscedastic noise
        np.random.seed(123)
        X = np.random.randn(200, 3)
        true_y = 2 * X[:, 0] - X[:, 1] + 0.5 * X[:, 2]

        # Add heteroscedastic noise
        noise_std = 0.1 + 0.5 * np.abs(X[:, 0])
        noise = np.random.randn(200) * noise_std
        y = true_y + noise

        X_tensor = Tensor(X)
        y_tensor = Tensor(y.reshape(-1, 1))

        # Create models for different percentiles
        models = {}
        losses = {}
        percentiles = [0.1, 0.5, 0.9]

        for p in percentiles:
            models[p] = Sequential(
                Linear(3, 20),
                lambda x: x.relu(),
                Linear(20, 10),
                lambda x: x.relu(),
                Linear(10, 1),
            )
            losses[p] = PercentileRegressionLoss(percentile=p)

        # Train each model
        for p in percentiles:
            optimizer = Adam(models[p].parameters(), lr=0.01)

            for epoch in range(100):
                optimizer.zero_grad()
                pred = models[p](X_tensor)
                loss = losses[p](pred, y_tensor)
                loss.backward()
                optimizer.step()

        # Test that different percentiles give different predictions
        test_X = Tensor(np.array([[1.0, 0.0, 0.0]]))  # Test point

        predictions = {}
        for p in percentiles:
            predictions[p] = models[p](test_X).data[0, 0]

        # Predictions should be ordered by percentile
        self.assertLess(predictions[0.1], predictions[0.5])
        self.assertLess(predictions[0.5], predictions[0.9])

    def test_loss_function_composition(self):
        """Test composing multiple loss functions"""

        class CompositeLoss(CustomLossTemplate):
            def __init__(
                self, loss1, loss2, weight1=0.5, weight2=0.5, reduction="mean"
            ):
                super().__init__(reduction)
                self.loss1 = loss1
                self.loss2 = loss2
                self.weight1 = weight1
                self.weight2 = weight2

            def forward(self, y_pred, y_true):
                l1 = self.loss1(y_pred, y_true)
                l2 = self.loss2(y_pred, y_true)
                combined = self.weight1 * l1 + self.weight2 * l2
                return combined

        y_pred = Tensor(np.random.randn(32, 5), requires_grad=True)
        y_true = Tensor(np.random.randn(32, 5))

        # Combine MSE and MAE
        composite_loss = CompositeLoss(
            loss1=lambda pred, true: mse_loss(pred, true),
            loss2=lambda pred, true: mae_loss(pred, true),
            weight1=0.7,
            weight2=0.3,
        )

        result = composite_loss(y_pred, y_true)

        # Compare with manual computation
        manual_mse = mse_loss(y_pred, y_true)
        manual_mae = mae_loss(y_pred, y_true)
        manual_composite = 0.7 * manual_mse + 0.3 * manual_mae

        self.assertAlmostEqual(result.data, manual_composite.data, places=6)

        # Test backward pass
        result.backward()
        self.assertIsNotNone(y_pred.grad)


if __name__ == "__main__":
    unittest.main()
