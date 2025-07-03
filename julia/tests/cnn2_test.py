import numpy as np
import unittest
from julia.core.tensor import Tensor
from julia.core.nn.layers import (
    Linear,
    Flatten,
    Dropout,
    Sequential,
    BatchNorm2D,
    Layer,
)
from julia.core.nn.conv import Conv2D
from julia.core.nn.pooling import MaxPool2D


class CNNTestModel(Sequential):
    """A simple CNN model for testing"""

    def __init__(self, in_channels=3, num_classes=10):
        # Define the CNN architecture
        conv1 = Conv2D(
            in_channels=in_channels, out_channels=16, kernel_size=3, padding=1
        )
        bn1 = BatchNorm2D(num_features=16)
        act1 = lambda x: x.relu()
        pool1 = MaxPool2D(kernel_size=2, stride=2)

        conv2 = Conv2D(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        bn2 = BatchNorm2D(num_features=32)
        act2 = lambda x: x.relu()
        pool2 = MaxPool2D(kernel_size=2, stride=2)

        flatten = Flatten()
        fc1 = Linear(
            32 * 8 * 8, 128
        )  # Assuming 32x32 input -> 8x8 after pooling layers
        act3 = lambda x: x.relu()
        dropout = Dropout(0.5)
        fc2 = Linear(128, num_classes)

        # Initialize Sequential with layers
        super().__init__(
            conv1,
            bn1,
            act1,
            pool1,
            conv2,
            bn2,
            act2,
            pool2,
            flatten,
            fc1,
            act3,
            dropout,
            fc2,
        )


class FeatureExtractor(Layer):
    def __init__(self, model, layer_indices):
        super().__init__()
        # Ensure model is a Sequential instance or similar that has _layers_list
        if not isinstance(model, Sequential) or not hasattr(model, "_layers_list"):
            raise TypeError(
                "FeatureExtractor expects a Sequential model with a '_layers_list' attribute"
            )
        self.model = model
        self.layer_indices = layer_indices
        self.features = {}

    def forward(self, x):
        features = {}
        activation = x

        # Access the internal list from Sequential that holds layers/functions in order
        if not hasattr(self.model, "_layers_list"):
            # This check is redundant if the __init__ check passes, but safe
            raise AttributeError(
                "Model passed to FeatureExtractor does not have '_layers_list'"
            )

        # Iterate over the correct attribute: _layers_list
        for i, layer_or_fn in enumerate(self.model._layers_list):
            activation = layer_or_fn(activation)  # Call layer or function
            if i in self.layer_indices:
                # Store feature using the sequential index 'i'
                features[i] = activation

        self.features = features
        return activation


class MSELoss:
    """Simple MSE loss function"""

    @staticmethod
    def backward(pred, target):
        """Return gradients of MSE loss with respect to predictions"""
        batch_size = pred.shape[0]
        return 2 * (pred - target) / batch_size


class TestCNNModel(unittest.TestCase):
    def test_cnn_forward_dimensions(self):
        """Test that a CNN forward pass produces correct output dimensions"""
        # Create model
        cnn = CNNTestModel(in_channels=3, num_classes=10)

        # Create a batch of test images (3 channels, 32x32)
        batch_size = 4
        images = Tensor(np.random.randn(batch_size, 3, 32, 32))

        # Forward pass
        cnn.train(True)
        output = cnn(images)

        # Check output dimensions
        self.assertEqual(output.shape, (batch_size, 10))

    def test_cnn_intermediate_shapes(self):
        """Test the shapes of intermediate features in a CNN"""
        # Create model
        cnn = CNNTestModel(in_channels=3, num_classes=10)

        # Define layers to extract features from
        extract_layers = [0, 3, 7, 8, 11]  # Conv1, Pool1, Pool2, Flatten, FC1
        extractor = FeatureExtractor(cnn, extract_layers)

        # Create a test image
        batch_size = 2
        images = Tensor(np.random.randn(batch_size, 3, 32, 32))

        # Forward pass
        _ = extractor(images)

        # Check intermediate feature shapes
        expected_shapes = {
            0: (batch_size, 16, 32, 32),  # After Conv1
            3: (batch_size, 16, 16, 16),  # After Pool1
            7: (batch_size, 32, 8, 8),  # After Pool2
            8: (batch_size, 2048),  # After Flatten (32*8*8 = 2048)
            11: (batch_size, 128),  # After FC1
        }

        for layer_idx, expected_shape in expected_shapes.items():
            self.assertEqual(extractor.features[layer_idx].shape, expected_shape)

    def test_cnn_eval_mode(self):
        """Test that dropout and batch normalization behave differently in eval mode"""
        # Create model
        cnn = CNNTestModel(in_channels=3, num_classes=10)

        # Create a test image
        images = Tensor(np.random.randn(2, 3, 32, 32))

        # Forward pass in training mode
        cnn.train(True)
        output_train = cnn(images)

        # Forward pass in eval mode
        cnn.train(False)
        output_eval = cnn(images)

        # In eval mode, with the same input, Dropout is disabled
        # So the outputs should typically be different
        self.assertFalse(np.allclose(output_train.data, output_eval.data))

    def test_cnn_parameter_count(self):
        """Test that the CNN has the expected number of parameters"""
        # Create model
        cnn = CNNTestModel(in_channels=3, num_classes=10)

        # Count parameters
        param_count = sum(np.prod(param.shape) for param in cnn.parameters())

        # Expected parameter count:
        # Conv1: (16 * 3 * 3 * 3) + 16 = 448
        # BN1: 16 * 2 = 32 (gamma and beta)
        # Conv2: (32 * 16 * 3 * 3) + 32 = 4640
        # BN2: 32 * 2 = 64 (gamma and beta)
        # FC1: (2048 * 128) + 128 = 262,272
        # FC2: (128 * 10) + 10 = 1,290
        # Total: 268,746
        expected_count = 268746

        self.assertEqual(param_count, expected_count)

    def test_cnn_direct_param_access(self):
        """Test direct parameter access and modification"""
        # Create a minimal model - just the first few layers
        model = Sequential(
            Conv2D(in_channels=3, out_channels=4, kernel_size=3, padding=1),
            lambda x: x.relu(),
        )

        # Access and modify parameters directly
        for param in model.parameters():
            # Store original value
            orig_data = param.data.copy()

            # Manually update parameter
            param.data += 0.1

            # Verify parameter was updated
            self.assertFalse(np.array_equal(param.data, orig_data))

        # This test verifies we can access and modify parameters directly
        self.assertTrue(True)

    def test_simple_loss_and_grad(self):
        """Test a very simple loss and gradient calculation"""
        # Create a minimal model - just a single Linear layer
        model = Linear(in_features=3, out_features=2)

        # Access parameters directly
        weight = model.weight
        bias = model.bias

        # Create input
        x = Tensor(np.array([[1.0, 2.0, 3.0]]), requires_grad=True)

        # Forward pass
        y = model(x)

        # Create target
        target = Tensor(np.array([[0.5, 0.5]]))

        # Calculate squared error
        error = y - target
        loss = (error * error).data.mean()

        # Calculate gradient of loss w.r.t. output manually
        output_grad = 2 * error.data / np.prod(error.shape)

        # Set gradient on y and do backward
        y.grad = Tensor(output_grad)

        # Call backward on y
        y.backward(y.grad)

        # Check that x now has a gradient
        self.assertIsNotNone(x.grad)

        # Check weight and bias gradients exist
        self.assertIsNotNone(weight.grad)
        self.assertIsNotNone(bias.grad)


if __name__ == "__main__":
    unittest.main()
