import numpy as np
import unittest
from julia.core.tensor import Tensor
from julia.core.nn.conv import Conv2D, DepthwiseSeparableConv2D, DilatedConv2D
from julia.core.nn.pooling import MaxPool2D, AvgPool2D


class TestConvLayers(unittest.TestCase):
    def test_conv2d_forward(self):
        """Test the forward pass of Conv2D layer"""
        batch_size = 2
        in_channels = 3
        out_channels = 4
        height = 8
        width = 8
        kernel_size = 3

        # Create a Conv2D layer
        conv = Conv2D(in_channels, out_channels, kernel_size, padding=1)

        # Set deterministic weights and bias for testing
        conv.weight = Tensor(
            np.ones((out_channels, in_channels, kernel_size, kernel_size)),
            requires_grad=True,
        )
        conv.bias = Tensor(np.ones(out_channels), requires_grad=True)

        # Create input
        x = Tensor(np.ones((batch_size, in_channels, height, width)))

        # Forward pass
        output = conv(x)
        output_data = output.data
        # 1. Check output shape
        # With padding=1, stride=1, kernel_size=3, output H/W should match input H/W
        # H_out = (8 + 2*1 - 3) // 1 + 1 = 7 // 1 + 1 = 8
        # W_out = (8 + 2*1 - 3) // 1 + 1 = 7 // 1 + 1 = 8
        expected_shape = (batch_size, out_channels, height, width)
        self.assertEqual(output.shape, expected_shape, "Output shape mismatch")

        # 2. Calculate Expected Values for different regions based on padding
        k_h, k_w = (kernel_size, kernel_size)
        bias_val = 1.0  # Since conv.bias is all ones

        # Center: Full receptive field (3x3=9 ones) * in_channels + bias
        expected_center_value = (k_h * k_w * in_channels) + bias_val  # 9 * 3 + 1 = 28.0

        # Corner: Receptive field overlaps padding (4 non-zero values: 2x2 patch of 1s) * in_channels + bias
        ones_in_corner_receptive_field = 4
        expected_corner_value = (
            ones_in_corner_receptive_field * in_channels
        ) + bias_val  # 4 * 3 + 1 = 13.0

        # Edge (Non-Corner): Receptive field overlaps padding (6 non-zero values: 2x3 or 3x2 patch of 1s) * in_channels + bias
        ones_in_edge_receptive_field = 6
        expected_edge_value = (
            ones_in_edge_receptive_field * in_channels
        ) + bias_val  # 6 * 3 + 1 = 19.0

        print("\n--- Testing Conv2D Forward ---")
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Expected Center Value: {expected_center_value}")
        print(f"Expected Edge Value: {expected_edge_value}")
        print(f"Expected Corner Value: {expected_corner_value}")

        # 3. Check specific pixel values (using first batch, first output channel)
        # Check top-left corner
        actual_corner = output_data[0, 0, 0, 0]
        print(f"Actual Corner [0,0,0,0]: {actual_corner}")
        self.assertAlmostEqual(
            actual_corner,
            expected_corner_value,
            places=5,
            msg="Corner pixel [0,0,0,0] mismatch.",
        )

        # Check top edge (non-corner)
        actual_top_edge = output_data[0, 0, 0, 1]
        print(f"Actual Top Edge [0,0,0,1]: {actual_top_edge}")
        self.assertAlmostEqual(
            actual_top_edge,
            expected_edge_value,
            places=5,
            msg="Top edge pixel [0,0,0,1] mismatch.",
        )

        # Check left edge (non-corner)
        actual_left_edge = output_data[0, 0, 1, 0]
        print(f"Actual Left Edge [0,0,1,0]: {actual_left_edge}")
        self.assertAlmostEqual(
            actual_left_edge,
            expected_edge_value,
            places=5,
            msg="Left edge pixel [0,0,1,0] mismatch.",
        )

        # Check a central pixel
        actual_center = output_data[0, 0, 1, 1]
        print(f"Actual Center [0,0,1,1]: {actual_center}")
        self.assertAlmostEqual(
            actual_center,
            expected_center_value,
            places=5,
            msg="Center pixel [0,0,1,1] mismatch.",
        )

        # 4. Check the entire central region matches the expected center value
        # With padding=1, the border of size 1 is affected. Center is [:, :, 1:-1, 1:-1]
        center_output_region = output_data[:, :, 1:-1, 1:-1]
        self.assertTrue(
            np.allclose(center_output_region, expected_center_value),
            msg=f"Central region values mismatch. Expected all to be {expected_center_value}",
        )
        print(
            f"Central region check passed (all values approx {expected_center_value})."
        )

        # 5. Optional: Check all corners and edges explicitly if desired
        # Check all 4 corners of the first image, first channel
        corners = [
            output_data[0, 0, 0, 0],
            output_data[0, 0, 0, -1],
            output_data[0, 0, -1, 0],
            output_data[0, 0, -1, -1],
        ]
        self.assertTrue(
            np.allclose(corners, expected_corner_value),
            msg=f"Not all corner values match expected {expected_corner_value}",
        )
        print(
            f"All 4 corner values check passed (all values approx {expected_corner_value})."
        )

        # Check top/bottom edge regions (excluding corners)
        top_edge_region = output_data[:, :, 0, 1:-1]
        bottom_edge_region = output_data[:, :, -1, 1:-1]
        self.assertTrue(
            np.allclose(top_edge_region, expected_edge_value),
            msg=f"Top edge region values mismatch. Expected all to be {expected_edge_value}",
        )
        self.assertTrue(
            np.allclose(bottom_edge_region, expected_edge_value),
            msg=f"Bottom edge region values mismatch. Expected all to be {expected_edge_value}",
        )

        # Check left/right edge regions (excluding corners)
        left_edge_region = output_data[:, :, 1:-1, 0]
        right_edge_region = output_data[:, :, 1:-1, -1]
        self.assertTrue(
            np.allclose(left_edge_region, expected_edge_value),
            msg=f"Left edge region values mismatch. Expected all to be {expected_edge_value}",
        )
        self.assertTrue(
            np.allclose(right_edge_region, expected_edge_value),
            msg=f"Right edge region values mismatch. Expected all to be {expected_edge_value}",
        )
        print(
            f"All edge regions check passed (all values approx {expected_edge_value})."
        )

    def test_conv2d_with_stride(self):
        """Test Conv2D with stride"""
        batch_size = 2
        in_channels = 3
        out_channels = 4
        height = 8
        width = 8
        kernel_size = 3
        stride = 2

        # Create a Conv2D layer with stride
        conv = Conv2D(in_channels, out_channels, kernel_size, stride=stride)

        # Set deterministic weights for testing
        conv.weight = Tensor(
            np.ones((out_channels, in_channels, kernel_size, kernel_size)),
            requires_grad=True,
        )
        conv.bias = Tensor(np.ones(out_channels), requires_grad=True)

        # Create input
        x = Tensor(np.ones((batch_size, in_channels, height, width)))

        # Forward pass
        output = conv(x)

        # Check output shape
        # With stride=2 and kernel_size=3, output dimensions should be (8-3)/2 + 1 = 3
        expected_height = (height - kernel_size) // stride + 1
        expected_width = (width - kernel_size) // stride + 1
        expected_shape = (batch_size, out_channels, expected_height, expected_width)

        self.assertEqual(output.shape, expected_shape)

        # Verify values are as expected
        expected_value = kernel_size * kernel_size * in_channels + 1
        self.assertTrue(np.allclose(output.data, expected_value))

    def test_maxpool2d(self):
        """Test MaxPool2D layer"""
        batch_size = 2
        channels = 3
        height = 8
        width = 8
        pool_size = 2

        # Create a MaxPool2D layer
        pool = MaxPool2D(kernel_size=pool_size)

        # Create input with increasing values
        x_data = np.zeros((batch_size, channels, height, width))
        for h in range(height):
            for w in range(width):
                x_data[:, :, h, w] = h * width + w

        x = Tensor(x_data)

        # Forward pass
        output = pool(x)

        # Check output shape
        expected_height = height // pool_size
        expected_width = width // pool_size
        expected_shape = (batch_size, channels, expected_height, expected_width)

        self.assertEqual(output.shape, expected_shape)

        # Verify max pooling behavior
        # In each 2x2 window, the max value should be at the bottom right
        for b in range(batch_size):
            for c in range(channels):
                for h in range(expected_height):
                    for w in range(expected_width):
                        pool_max = output.data[b, c, h, w]
                        expected_max = x_data[
                            b,
                            c,
                            h * pool_size + pool_size - 1,
                            w * pool_size + pool_size - 1,
                        ]
                        self.assertEqual(pool_max, expected_max)

    def test_avgpool2d(self):
        """Test AvgPool2D layer"""
        batch_size = 2
        channels = 3
        height = 8
        width = 8
        pool_size = 2

        # Create an AvgPool2D layer
        pool = AvgPool2D(kernel_size=pool_size)

        # Create input
        x = Tensor(np.ones((batch_size, channels, height, width)))

        # Forward pass
        output = pool(x)

        # Check output shape
        expected_height = height // pool_size
        expected_width = width // pool_size
        expected_shape = (batch_size, channels, expected_height, expected_width)

        self.assertEqual(output.shape, expected_shape)

        # For all ones input, the average should also be one
        self.assertTrue(np.allclose(output.data, 1.0))

        # Test with non-uniform input
        x_data = np.zeros((batch_size, channels, height, width))
        for h in range(height):
            for w in range(width):
                x_data[:, :, h, w] = h * width + w

        x = Tensor(x_data)
        output = pool(x)

        # Verify average pooling behavior
        for b in range(batch_size):
            for c in range(channels):
                for h in range(expected_height):
                    for w in range(expected_width):
                        # Calculate expected average of 2x2 window
                        window_sum = 0
                        for ph in range(pool_size):
                            for pw in range(pool_size):
                                window_sum += x_data[
                                    b, c, h * pool_size + ph, w * pool_size + pw
                                ]
                        expected_avg = window_sum / (pool_size * pool_size)
                        self.assertAlmostEqual(
                            output.data[b, c, h, w], expected_avg, places=5
                        )

    def test_depthwise_separable_conv2d(self):
        """Test DepthwiseSeparableConv2D layer"""
        batch_size = 2
        in_channels = 3
        out_channels = 6
        height = 8
        width = 8
        kernel_size = 3

        # Create a DepthwiseSeparableConv2D layer
        conv = DepthwiseSeparableConv2D(
            in_channels, out_channels, kernel_size, padding=1
        )

        # Create input
        x = Tensor(np.ones((batch_size, in_channels, height, width)))

        # Forward pass
        output = conv(x)

        # Check output shape - should match input spatial dimensions
        expected_shape = (batch_size, out_channels, height, width)
        self.assertEqual(output.shape, expected_shape)

    def test_dilated_conv2d(self):
        """Test DilatedConv2D layer"""
        batch_size = 2
        in_channels = 3
        out_channels = 4
        height = 8
        width = 8
        kernel_size = 3
        dilation = 2

        # Create a DilatedConv2D layer
        conv = DilatedConv2D(in_channels, out_channels, kernel_size, dilation=dilation)

        # Create input
        x = Tensor(np.ones((batch_size, in_channels, height, width)))

        # Forward pass
        output = conv(x)

        # Check output shape
        # With dilation=2, effective kernel size is 5 (3 + (3-1))
        # Output size should be (8 - 5 + 1) = 4
        expected_height = height - (kernel_size - 1) * dilation - 1 + 1
        expected_width = width - (kernel_size - 1) * dilation - 1 + 1
        expected_shape = (batch_size, out_channels, expected_height, expected_width)

        self.assertEqual(output.shape, expected_shape)


if __name__ == "__main__":
    unittest.main()
