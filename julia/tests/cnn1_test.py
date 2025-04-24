import numpy as np
from julia.core.tensor import Tensor
from julia.core.nn.layers import Linear, Flatten, Dropout, Sequential, BatchNorm2D
from julia.core.nn.conv import Conv2D
from julia.core.nn.pooling import MaxPool2D
from julia.core.optim import Adam
import unittest

class TestCNN(unittest.TestCase):
    def test_cnn_forward(self):
        """Test a simple CNN forward pass"""
        # Create a small CNN model with dimensions we can trace manually
        model = Sequential(
            # Input: 3x32x32
            Conv2D(in_channels=3, out_channels=16, kernel_size=3, padding=1),  # Output: 16x32x32
            BatchNorm2D(16),
            lambda x: x.relu(),
            MaxPool2D(kernel_size=2, stride=2),  # Output: 16x16x16
            
            Conv2D(in_channels=16, out_channels=32, kernel_size=3, padding=1),  # Output: 32x16x16
            BatchNorm2D(32),
            lambda x: x.relu(),
            MaxPool2D(kernel_size=2, stride=2),  # Output: 32x8x8
            
            Flatten(),  # Output: 32*8*8 = 2048
            Linear(2048, 128),  # Adjusted to match the flattened dimensions
            lambda x: x.relu(),
            Dropout(0.5),
            Linear(128, 10)
        )
        
        # Create a random batch of images
        batch_size = 4
        x = Tensor(np.random.randn(batch_size, 3, 32, 32))
        
        # Forward pass
        output = model(x)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, 10))
        
        # Manually trace dimensions through the network
        # Input: batch_size x 3 x 32 x 32
        # After Conv2D(3, 16): batch_size x 16 x 32 x 32
        # After MaxPool2D: batch_size x 16 x 16 x 16
        # After Conv2D(16, 32): batch_size x 32 x 16 x 16
        # After MaxPool2D: batch_size x 32 x 8 x 8
        # After Flatten: batch_size x (32*8*8) = batch_size x 2048
        # After Linear(2048, 128): batch_size x 128
        # After Linear(128, 10): batch_size x 10
        
        # For a more detailed test, you could extract and check intermediate shapes
        # by adding hooks or by calling each layer individually

if __name__ == '__main__':
    unittest.main()
