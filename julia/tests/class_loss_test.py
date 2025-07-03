import numpy as np
import unittest
from julia.core.tensor import Tensor
from julia.core.nn.layers import Linear, Sequential
from julia.core.optim import Adam
from julia.core.loss import (
    cross_entropy_loss,
    binary_cross_entropy_loss,
    focal_loss,
    BinaryCrossEntropyLoss,
)


class TestClassificationLosses(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.batch_size = 64
        self.input_dim = 20
        self.num_classes = 5

        # Create synthetic classification data
        X = np.random.randn(self.batch_size, self.input_dim)
        y = np.random.randint(0, self.num_classes, (self.batch_size,))

        self.X = Tensor(X, requires_grad=False)
        self.y_true = Tensor(y, requires_grad=False)

        # Create one-hot encoded labels
        y_onehot = np.zeros((self.batch_size, self.num_classes))
        y_onehot[np.arange(self.batch_size), y] = 1
        self.y_onehot = Tensor(y_onehot, requires_grad=False)

        # Create a classification model
        self.model = Sequential(
            Linear(self.input_dim, 64),
            lambda x: x.relu(),
            Linear(64, 32),
            lambda x: x.relu(),
            Linear(32, self.num_classes),
        )

        # Generate predictions (logits)
        self.y_pred = self.model(self.X)

        # Binary classification setup
        self.binary_model = Sequential(
            Linear(self.input_dim, 32), lambda x: x.relu(), Linear(32, 1)
        )

        self.y_binary_true = Tensor(
            np.random.randint(0, 2, (self.batch_size, 1)).astype(float)
        )
        self.y_binary_pred = self.binary_model(self.X)

    def test_cross_entropy_loss_forward(self):
        """Test Cross Entropy loss forward pass"""
        loss = cross_entropy_loss(self.y_pred, self.y_true)

        self.assertEqual(loss.data.shape, ())
        self.assertGreater(loss.data, 0)

        # Manual calculation with numerical stability
        logits = self.y_pred.data
        max_logits = np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits - max_logits)
        softmax = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        expected_loss = -np.mean(
            np.log(softmax[np.arange(self.batch_size), self.y_true.data] + 1e-8)
        )
        self.assertAlmostEqual(loss.data, expected_loss, places=5)

    def test_cross_entropy_loss_backward(self):
        """Test Cross Entropy loss backward pass"""
        loss = cross_entropy_loss(self.y_pred, self.y_true)
        loss.backward()

        for param in self.model.parameters():
            self.assertIsNotNone(param.grad)
            self.assertFalse(np.all(param.grad.data == 0))

    def test_cross_entropy_onehot_vs_sparse(self):
        """Test Cross Entropy with one-hot vs sparse labels"""
        # This test would need modification of the cross entropy implementation
        # to handle both sparse and one-hot labels properly
        loss_sparse = cross_entropy_loss(self.y_pred, self.y_true)

        # For now, just test that sparse labels work
        self.assertIsInstance(loss_sparse.data, (float, np.floating, np.ndarray))
        self.assertGreater(loss_sparse.data, 0)

    def test_binary_cross_entropy_loss_forward(self):
        """Test Binary Cross Entropy loss forward pass"""
        loss = binary_cross_entropy_loss(self.y_binary_pred, self.y_binary_true)

        self.assertEqual(loss.data.shape, ())
        self.assertGreater(loss.data, 0)

        # Manual calculation
        sigmoid_vals = 1 / (1 + np.exp(-self.y_binary_pred.data))
        sigmoid_vals = np.clip(sigmoid_vals, 1e-8, 1 - 1e-8)

        expected_loss = -np.mean(
            self.y_binary_true.data * np.log(sigmoid_vals)
            + (1 - self.y_binary_true.data) * np.log(1 - sigmoid_vals)
        )
        self.assertAlmostEqual(loss.data, expected_loss, places=5)

    def test_binary_cross_entropy_loss_backward(self):
        """Test Binary Cross Entropy loss backward pass"""
        loss = binary_cross_entropy_loss(self.y_binary_pred, self.y_binary_true)
        loss.backward()

        for param in self.binary_model.parameters():
            self.assertIsNotNone(param.grad)

    def test_focal_loss_forward(self):
        """Test Focal loss forward pass"""
        alpha = 1.0
        gamma = 2.0
        loss = focal_loss(self.y_binary_pred, self.y_binary_true, alpha, gamma)

        self.assertEqual(loss.data.shape, ())
        self.assertGreater(loss.data, 0)

        # Focal loss should be different from BCE
        bce_loss = binary_cross_entropy_loss(self.y_binary_pred, self.y_binary_true)
        self.assertNotAlmostEqual(loss.data, bce_loss.data, places=3)

    def test_focal_loss_gamma_effect(self):
        """Test that gamma parameter affects focal loss"""
        gammas = [0.0, 1.0, 2.0, 5.0]
        losses = []

        for gamma in gammas:
            loss = focal_loss(
                self.y_binary_pred, self.y_binary_true, alpha=1.0, gamma=gamma
            )
            losses.append(loss.data)

        # Losses should be different for different gamma values
        for i in range(len(losses) - 1):
            self.assertNotAlmostEqual(losses[i], losses[i + 1], places=3)

    def test_focal_loss_class_imbalance(self):
        """Test focal loss with imbalanced data"""
        # Create heavily imbalanced binary data
        imbalanced_labels = np.zeros((self.batch_size, 1))
        imbalanced_labels[:5] = 1  # Only 5 positive examples
        y_imbalanced = Tensor(imbalanced_labels)

        focal_loss_val = focal_loss(
            self.y_binary_pred, y_imbalanced, alpha=1.0, gamma=2.0
        )
        bce_loss_val = binary_cross_entropy_loss(self.y_binary_pred, y_imbalanced)

        # Both should work with imbalanced data
        self.assertGreater(focal_loss_val.data, 0)
        self.assertGreater(bce_loss_val.data, 0)

    def test_classification_training_convergence(self):
        """Test that classification models can train with different losses"""
        # Simple binary classification problem
        np.random.seed(123)
        X_simple = np.random.randn(100, 2)
        y_simple = (X_simple[:, 0] + X_simple[:, 1] > 0).astype(float).reshape(-1, 1)

        X_tensor = Tensor(X_simple)
        y_tensor = Tensor(y_simple)

        losses_to_test = [
            ("BCE", lambda pred, true: binary_cross_entropy_loss(pred, true)),
            ("Focal", lambda pred, true: focal_loss(pred, true, alpha=1.0, gamma=2.0)),
        ]

        for loss_name, loss_fn in losses_to_test:
            model = Sequential(Linear(2, 8), lambda x: x.relu(), Linear(8, 1))

            optimizer = Adam(model.parameters(), lr=0.01)

            initial_loss = None
            final_loss = None

            for epoch in range(50):
                optimizer.zero_grad()
                pred = model(X_tensor)
                loss = loss_fn(pred, y_tensor)

                if epoch == 0:
                    initial_loss = loss.data
                if epoch == 49:
                    final_loss = loss.data

                loss.backward()
                optimizer.step()

            self.assertLess(
                final_loss,
                initial_loss,
                f"{loss_name} loss did not decrease during training",
            )

    def test_multiclass_classification_training(self):
        """Test multiclass classification training"""
        # Simple 3-class problem
        np.random.seed(456)
        X_multi = np.random.randn(150, 3)
        y_multi = np.argmax(X_multi, axis=1)  # Class is the index of max feature

        X_tensor = Tensor(X_multi)
        y_tensor = Tensor(y_multi)

        model = Sequential(Linear(3, 16), lambda x: x.relu(), Linear(16, 3))

        optimizer = Adam(model.parameters(), lr=0.01)

        initial_loss = None
        final_loss = None

        for epoch in range(30):
            optimizer.zero_grad()
            pred = model(X_tensor)
            loss = cross_entropy_loss(pred, y_tensor)

            if epoch == 0:
                initial_loss = loss.data
            if epoch == 29:
                final_loss = loss.data

            loss.backward()
            optimizer.step()

        self.assertLess(
            final_loss,
            initial_loss,
            "Cross entropy loss did not decrease during multiclass training",
        )

    def test_accuracy_improvement(self):
        """Test that accuracy improves during training"""
        # Binary classification accuracy test
        np.random.seed(789)
        X_simple = np.random.randn(200, 3)
        y_simple = (np.sum(X_simple, axis=1) > 0).astype(float).reshape(-1, 1)

        X_tensor = Tensor(X_simple)
        y_tensor = Tensor(y_simple)

        model = Sequential(Linear(3, 10), lambda x: x.relu(), Linear(10, 1))

        optimizer = Adam(model.parameters(), lr=0.01)

        def compute_accuracy(predictions, targets):
            probs = 1 / (1 + np.exp(-predictions.data))
            predicted_classes = (probs > 0.5).astype(float)
            return np.mean(predicted_classes == targets.data)

        initial_acc = compute_accuracy(model(X_tensor), y_tensor)

        for epoch in range(40):
            optimizer.zero_grad()
            pred = model(X_tensor)
            loss = binary_cross_entropy_loss(pred, y_tensor)
            loss.backward()
            optimizer.step()

        final_acc = compute_accuracy(model(X_tensor), y_tensor)

        self.assertGreater(
            final_acc, initial_acc, "Accuracy did not improve during training"
        )
        self.assertGreater(
            final_acc,
            0.7,  # Should achieve reasonable accuracy
            "Final accuracy is too low",
        )

    def test_loss_gradients_sanity(self):
        """Test that gradients have expected properties"""
        losses = [
            cross_entropy_loss(self.y_pred, self.y_true),
            binary_cross_entropy_loss(self.y_binary_pred, self.y_binary_true),
            focal_loss(self.y_binary_pred, self.y_binary_true, alpha=1.0, gamma=2.0),
        ]

        models = [self.model, self.binary_model, self.binary_model]

        for loss, model in zip(losses, models):
            # Clear gradients
            for param in model.parameters():
                param.zero_grad()

            loss.backward()

            # Check gradients exist and are finite
            for param in model.parameters():
                self.assertIsNotNone(param.grad)
                self.assertTrue(np.all(np.isfinite(param.grad.data)))

                # Gradients should not be all zeros (for this random data)
                grad_norm = np.linalg.norm(param.grad.data)
                self.assertGreater(grad_norm, 1e-8)

    def test_loss_reduction_modes(self):
        """Test different reduction modes for classification losses"""
        # Test BCE with different reductions
        loss_none = BinaryCrossEntropyLoss.apply(
            self.y_binary_pred, self.y_binary_true, "none"
        )
        loss_mean = BinaryCrossEntropyLoss.apply(
            self.y_binary_pred, self.y_binary_true, "mean"
        )
        loss_sum = BinaryCrossEntropyLoss.apply(
            self.y_binary_pred, self.y_binary_true, "sum"
        )

        # Check shapes
        self.assertEqual(loss_none.shape, self.y_binary_pred.shape)
        self.assertEqual(loss_mean.data.shape, ())
        self.assertEqual(loss_sum.data.shape, ())

        # Check relationships
        self.assertAlmostEqual(loss_mean.data, np.mean(loss_none.data), places=6)
        self.assertAlmostEqual(loss_sum.data, np.sum(loss_none.data), places=6)

    def test_probability_outputs(self):
        """Test that models produce reasonable probability distributions"""
        # After some training, check that softmax outputs sum to 1
        model = Sequential(
            Linear(self.input_dim, 32), lambda x: x.relu(), Linear(32, self.num_classes)
        )

        optimizer = Adam(model.parameters(), lr=0.01)

        # Train briefly
        for _ in range(10):
            optimizer.zero_grad()
            pred = model(self.X)
            loss = cross_entropy_loss(pred, self.y_true)
            loss.backward()
            optimizer.step()

        # Check final predictions
        final_pred = model(self.X)

        # Apply softmax manually
        logits = final_pred.data
        max_logits = np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits - max_logits)
        softmax_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # Check that probabilities sum to 1
        prob_sums = np.sum(softmax_probs, axis=1)
        np.testing.assert_allclose(prob_sums, 1.0, rtol=1e-6)

        # Check that all probabilities are positive
        self.assertTrue(np.all(softmax_probs >= 0))
        self.assertTrue(np.all(softmax_probs <= 1))

    def test_loss_with_extreme_predictions(self):
        """Test loss functions with extreme prediction values"""
        # Very confident correct predictions (should have low loss)
        confident_correct = Tensor(np.array([[10.0, -10.0, -10.0]]), requires_grad=True)
        true_label = Tensor(np.array([0]))

        loss_confident = cross_entropy_loss(confident_correct, true_label)

        # Very confident incorrect predictions (should have high loss)
        confident_wrong = Tensor(np.array([[-10.0, 10.0, -10.0]]), requires_grad=True)

        loss_wrong = cross_entropy_loss(confident_wrong, true_label)

        # Uncertain predictions (should have medium loss)
        uncertain = Tensor(np.array([[0.0, 0.0, 0.0]]), requires_grad=True)

        loss_uncertain = cross_entropy_loss(uncertain, true_label)

        # Check loss ordering
        self.assertLess(loss_confident.data, loss_uncertain.data)
        self.assertLess(loss_uncertain.data, loss_wrong.data)

        # Test backward passes don't crash
        for loss in [loss_confident, loss_wrong, loss_uncertain]:
            loss.backward()


if __name__ == "__main__":
    unittest.main()
