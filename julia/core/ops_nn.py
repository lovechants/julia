import numpy as np
from julia.core.tensor import Tensor, _ensure_tensor, Function


# Starting with Conv functions
class Conv1DFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding):
        """
        Foward pass for Conv1DFunction

        Args:
            input: Input tensor fo shape (batch_size, in_channels, seq_len)
            weight: Weight tensor of shape (out_channel, in_channels, kernel_size)
            bias: Bias tensor of shape (out_channel) or None
            stride: Conv stride
            padding: Conv padding

        Returns:
            Output tensor of shape (batch_size, out_channels, out_len)
        """
        input = _ensure_tensor(input)
        weight = _ensure_tensor(weight)
        if bias is not None:
            bias = _ensure_tensor(bias)

        batch_size, in_channels, seq_len = input.shape
        out_channels, _, kernel_size = weight.shape

        out_len = (seq_len + 2 * padding - kernel_size) // stride + 1

        if padding > 0:
            input_padded = np.pad(
                input.data, ((0, 0), (0, 0), (padding, padding)), mode="constant"
            )
        else:
            input_padded = input.data

        output_data = np.zeros((batch_size, out_channels, out_len))

        for b in range(batch_size):
            for c_out in range(out_channels):
                for c_in in range(in_channels):
                    for i in range(out_len):
                        i_start = i * stride
                        region = input_padded[b, c_in, i_start : i_start + kernel_size]
                        output_data[b, c_out, i] += np.sum(
                            region * weight.data[c_out, c_in, :]
                        )

        if bias is not None:
            reshaped_bias = bias.data.reshape(1, out_channels, 1)
            output_data += reshaped_bias

        ctx.save_for_backwards(input, weight, bias)
        ctx.save_data(
            stride=stride,
            padding=padding,
            input_shape=input.shape,
            weight_shape=weight.shape,
            output_shape=output_data.shape,
            padded_data=input_padded,
        )

        return Tensor(
            output_data, requires_grad=input.requires_grad or weight.requires_grad
        )

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for Conv1D.

        Args:
            grad_output: Gradient of loss with respect to output

        Returns:
            Gradients for input, weight, bias, stride, and padding
        """
        input, weight, bias = ctx.saved_tensors
        stride = ctx.saved_data["stride"]
        padding = ctx.saved_data["padding"]
        input_shape = ctx.saved_data["input_shape"]
        weight_shape = ctx.saved_data["weight_shape"]
        output_shape = ctx.saved_data["output_shape"]
        input_padded = ctx.saved_data["padded_data"]

        batch_size, in_channels, seq_len = input_shape
        out_channels, _, kernel_size = weight_shape
        _, _, out_len = output_shape

        grad_input_data = np.zeros_like(input.data)
        grad_weight_data = np.zeros_like(weight.data)
        grad_bias_data = None if bias is None else np.zeros_like(bias.data)

        if bias is not None and bias.requires_grad:
            grad_bias_data = np.sum(grad_output.data, axis=(0, 2))

        if weight.requires_grad:
            for b in range(batch_size):
                for c_out in range(out_channels):
                    for c_in in range(in_channels):
                        for i in range(out_len):
                            i_start = i * stride
                            region = input_padded[
                                b, c_in, i_start : i_start + kernel_size
                            ]
                            grad_weight_data[c_out, c_in, :] += (
                                region * grad_output.data[b, c_out, i]
                            )

        if input.requires_grad:
            grad_input_padded = np.zeros_like(input_padded)

            for b in range(batch_size):
                for c_out in range(out_channels):
                    for c_in in range(in_channels):
                        for i in range(out_len):
                            i_start = i * stride
                            for k in range(kernel_size):
                                grad_input_padded[b, c_in, i_start + k] += (
                                    weight.data[c_out, c_in, k]
                                    * grad_output.data[b, c_out, i]
                                )

            if padding > 0:
                grad_input_data = grad_input_padded[:, :, padding:-padding]
            else:
                grad_input_data = grad_input_padded

        grad_input = Tensor(grad_input_data) if input.requires_grad else None
        grad_weight = Tensor(grad_weight_data) if weight.requires_grad else None
        grad_bias = (
            Tensor(grad_bias_data) if bias is not None and bias.requires_grad else None
        )

        return grad_input, grad_weight, grad_bias, None, None


class Conv2DFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding):
        """
        Forward pass for Conv2D.

        Args:
            input: Input tensor of shape (batch_size, in_channels, height, width)
            weight: Weight tensor of shape (out_channels, in_channels, kernel_h, kernel_w)
            bias: Bias tensor of shape (out_channels) or None
            stride: Convolution stride (tuple of 2 integers)
            padding: Convolution padding (tuple of 2 integers)

        Returns:
            Output tensor of shape (batch_size, out_channels, out_height, out_width)
        """
        input = _ensure_tensor(input)
        weight = _ensure_tensor(weight)
        if bias is not None:
            bias = _ensure_tensor(bias)

        batch_size, in_channels, height, width = input.shape
        out_channels, _, kernel_h, kernel_w = weight.shape

        stride_h, stride_w = stride
        padding_h, padding_w = padding

        out_height = (height + 2 * padding_h - kernel_h) // stride_h + 1
        out_width = (width + 2 * padding_w - kernel_w) // stride_w + 1

        if padding_h > 0 or padding_w > 0:
            input_padded = np.pad(
                input.data,
                ((0, 0), (0, 0), (padding_h, padding_h), (padding_w, padding_w)),
                mode="constant",
            )
        else:
            input_padded = input.data

        output_data = np.zeros((batch_size, out_channels, out_height, out_width))

        # Im2col approach for faster computation
        def im2col(x, h_out, w_out):
            n, c, h, w = x.shape
            col = np.ndarray(
                (n * h_out * w_out, c * kernel_h * kernel_w), dtype=x.dtype
            )

            for y in range(h_out):
                y_start = y * stride_h
                y_end = y_start + kernel_h
                for x_pos in range(w_out):
                    x_start = x_pos * stride_w
                    x_end = x_start + kernel_w

                    patch = x[:, :, y_start:y_end, x_start:x_end]
                    col[y * w_out + x_pos :: h_out * w_out, :] = patch.reshape(n, -1)
            return col

        # Use im2col for convolution
        x_col = im2col(input_padded, out_height, out_width)

        # Reshape weight for matrix multiplication
        weight_col = weight.data.reshape(out_channels, -1)

        # Compute convolution as matrix multiplication
        output_data_flat = np.matmul(x_col, weight_col.T)

        # Reshape output
        output_data = output_data_flat.reshape(
            batch_size, out_height, out_width, out_channels
        )
        output_data = output_data.transpose(0, 3, 1, 2)  # NHWC -> NCHW

        if bias is not None:
            reshaped_bias = bias.data.reshape(1, out_channels, 1, 1)
            output_data += reshaped_bias

        ctx.save_for_backwards(input, weight, bias)
        ctx.save_data(
            stride=stride,
            padding=padding,
            input_shape=input.shape,
            weight_shape=weight.shape,
            output_shape=output_data.shape,
            x_col=x_col,
            input_padded=input_padded,
        )

        return Tensor(
            output_data, requires_grad=input.requires_grad or weight.requires_grad
        )

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for Conv2D.

        Args:
            grad_output: Gradient of loss with respect to output

        Returns:
            Gradients for input, weight, bias, stride, and padding
        """
        input, weight, bias = ctx.saved_tensors
        stride = ctx.saved_data["stride"]
        padding = ctx.saved_data["padding"]
        input_shape = ctx.saved_data["input_shape"]
        weight_shape = ctx.saved_data["weight_shape"]
        output_shape = ctx.saved_data["output_shape"]
        x_col = ctx.saved_data["x_col"]
        input_padded = ctx.saved_data["input_padded"]

        batch_size, in_channels, height, width = input_shape
        out_channels, _, kernel_h, kernel_w = weight_shape
        _, _, out_height, out_width = output_shape

        stride_h, stride_w = stride
        padding_h, padding_w = padding

        grad_input_data = np.zeros_like(input.data)
        grad_weight_data = np.zeros_like(weight.data)
        grad_bias_data = None if bias is None else np.zeros_like(bias.data)

        if bias is not None and bias.requires_grad:
            grad_bias_data = np.sum(grad_output.data, axis=(0, 2, 3))

        if weight.requires_grad:
            grad_out_reshaped = grad_output.data.transpose(0, 2, 3, 1).reshape(
                -1, out_channels
            )

            grad_weight_data_flat = np.matmul(grad_out_reshaped.T, x_col)
            grad_weight_data = grad_weight_data_flat.reshape(weight_shape)

        if input.requires_grad:
            weight_reshaped = weight.data.reshape(out_channels, -1).T
            grad_out_reshaped = grad_output.data.transpose(0, 2, 3, 1).reshape(
                -1, out_channels
            )

            grad_col = np.matmul(grad_out_reshaped, weight_reshaped.T)

            grad_input_padded = np.zeros_like(input_padded)

            grad_col_reshaped = grad_col.reshape(batch_size, out_height, out_width, -1)

            # Perform col2im
            for h in range(kernel_h):
                for w in range(kernel_w):
                    for c in range(in_channels):
                        # Extract the appropriate slice of grad_col_reshaped
                        col_slice = grad_col_reshaped[
                            :, :, :, c * kernel_h * kernel_w + h * kernel_w + w
                        ]

                        # Apply to padded gradient
                        for y in range(out_height):
                            for x in range(out_width):
                                grad_input_padded[
                                    :, c, y * stride_h + h, x * stride_w + w
                                ] += col_slice[:, y, x]

            # Extract the gradient from padded version if padding was used
            if padding_h > 0 or padding_w > 0:
                grad_input_data = grad_input_padded[
                    :, :, padding_h : padding_h + height, padding_w : padding_w + width
                ]
            else:
                grad_input_data = grad_input_padded

        # Create Tensor objects for gradients
        grad_input = Tensor(grad_input_data) if input.requires_grad else None
        grad_weight = Tensor(grad_weight_data) if weight.requires_grad else None
        grad_bias = (
            Tensor(grad_bias_data) if bias is not None and bias.requires_grad else None
        )

        # Return gradients (None for stride and padding as they're not learnable)
        return grad_input, grad_weight, grad_bias, None, None


class Conv3DFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding):
        """
        Forward pass for Conv3D.

        Args:
            input: Input tensor of shape (batch_size, in_channels, depth, height, width)
            weight: Weight tensor of shape (out_channels, in_channels, kernel_d, kernel_h, kernel_w)
            bias: Bias tensor of shape (out_channels) or None
            stride: Convolution stride (tuple of 3 integers)
            padding: Convolution padding (tuple of 3 integers)

        Returns:
            Output tensor of shape (batch_size, out_channels, out_depth, out_height, out_width)
        """
        input = _ensure_tensor(input)
        weight = _ensure_tensor(weight)
        if bias is not None:
            bias = _ensure_tensor(bias)

        batch_size, in_channels, depth, height, width = input.shape
        out_channels, _, kernel_d, kernel_h, kernel_w = weight.shape

        stride_d, stride_h, stride_w = stride
        padding_d, padding_h, padding_w = padding

        out_depth = (depth + 2 * padding_d - kernel_d) // stride_d + 1
        out_height = (height + 2 * padding_h - kernel_h) // stride_h + 1
        out_width = (width + 2 * padding_w - kernel_w) // stride_w + 1

        if padding_d > 0 or padding_h > 0 or padding_w > 0:
            input_padded = np.pad(
                input.data,
                (
                    (0, 0),
                    (0, 0),
                    (padding_d, padding_d),
                    (padding_h, padding_h),
                    (padding_w, padding_w),
                ),
                mode="constant",
            )
        else:
            input_padded = input.data

        output_data = np.zeros(
            (batch_size, out_channels, out_depth, out_height, out_width)
        )

        for b in range(batch_size):
            for c_out in range(out_channels):
                for c_in in range(in_channels):
                    for d in range(out_depth):
                        d_start = d * stride_d
                        for h in range(out_height):
                            h_start = h * stride_h
                            for w in range(out_width):
                                w_start = w * stride_w

                                region = input_padded[
                                    b,
                                    c_in,
                                    d_start : d_start + kernel_d,
                                    h_start : h_start + kernel_h,
                                    w_start : w_start + kernel_w,
                                ]
                                output_data[b, c_out, d, h, w] += np.sum(
                                    region * weight.data[c_out, c_in]
                                )

        if bias is not None:
            reshaped_bias = bias.data.reshape(1, out_channels, 1, 1, 1)
            output_data += reshaped_bias

        ctx.save_for_backwards(input, weight, bias)
        ctx.save_data(
            stride=stride,
            padding=padding,
            input_shape=input.shape,
            weight_shape=weight.shape,
            output_shape=output_data.shape,
            input_padded=input_padded,
        )

        return Tensor(
            output_data, requires_grad=input.requires_grad or weight.requires_grad
        )

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for Conv3D.

        Args:
            grad_output: Gradient of loss with respect to output

        Returns:
            Gradients for input, weight, bias, stride, and padding
        """
        input, weight, bias = ctx.saved_tensors
        stride = ctx.saved_data["stride"]
        padding = ctx.saved_data["padding"]
        input_shape = ctx.saved_data["input_shape"]
        weight_shape = ctx.saved_data["weight_shape"]
        output_shape = ctx.saved_data["output_shape"]
        input_padded = ctx.saved_data["input_padded"]

        batch_size, in_channels, depth, height, width = input_shape
        out_channels, _, kernel_d, kernel_h, kernel_w = weight_shape
        _, _, out_depth, out_height, out_width = output_shape

        stride_d, stride_h, stride_w = stride
        padding_d, padding_h, padding_w = padding

        grad_input_data = np.zeros_like(input.data)
        grad_weight_data = np.zeros_like(weight.data)
        grad_bias_data = None if bias is None else np.zeros_like(bias.data)

        if bias is not None and bias.requires_grad:
            grad_bias_data = np.sum(grad_output.data, axis=(0, 2, 3, 4))

        if weight.requires_grad:
            for b in range(batch_size):
                for c_out in range(out_channels):
                    for c_in in range(in_channels):
                        for d in range(out_depth):
                            d_start = d * stride_d
                            for h in range(out_height):
                                h_start = h * stride_h
                                for w in range(out_width):
                                    w_start = w * stride_w

                                    region = input_padded[
                                        b,
                                        c_in,
                                        d_start : d_start + kernel_d,
                                        h_start : h_start + kernel_h,
                                        w_start : w_start + kernel_w,
                                    ]

                                    grad_weight_data[c_out, c_in] += (
                                        region * grad_output.data[b, c_out, d, h, w]
                                    )

        if input.requires_grad:
            grad_input_padded = np.zeros_like(input_padded)

            for b in range(batch_size):
                for c_out in range(out_channels):
                    for c_in in range(in_channels):
                        for d in range(out_depth):
                            d_start = d * stride_d
                            for h in range(out_height):
                                h_start = h * stride_h
                                for w in range(out_width):
                                    w_start = w * stride_w

                                    grad_input_padded[
                                        b,
                                        c_in,
                                        d_start : d_start + kernel_d,
                                        h_start : h_start + kernel_h,
                                        w_start : w_start + kernel_w,
                                    ] += (
                                        weight.data[c_out, c_in]
                                        * grad_output.data[b, c_out, d, h, w]
                                    )

            if padding_d > 0 or padding_h > 0 or padding_w > 0:
                grad_input_data = grad_input_padded[
                    :,
                    :,
                    padding_d : padding_d + depth,
                    padding_h : padding_h + height,
                    padding_w : padding_w + width,
                ]
            else:
                grad_input_data = grad_input_padded

        grad_input = Tensor(grad_input_data) if input.requires_grad else None
        grad_weight = Tensor(grad_weight_data) if weight.requires_grad else None
        grad_bias = (
            Tensor(grad_bias_data) if bias is not None and bias.requires_grad else None
        )

        return grad_input, grad_weight, grad_bias, None, None


class TransposedConv2DFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding, output_padding):
        """
        Forward pass for TransposedConv2D.

        Args:
            input: Input tensor of shape (batch_size, in_channels, height, width)
            weight: Weight tensor of shape (in_channels, out_channels, kernel_h, kernel_w)
            bias: Bias tensor of shape (out_channels) or None
            stride: Convolution stride (tuple of 2 integers)
            padding: Convolution padding (tuple of 2 integers)
            output_padding: Additional padding for output (tuple of 2 integers)

        Returns:
            Output tensor of shape (batch_size, out_channels, out_height, out_width)
        """
        input = _ensure_tensor(input)
        weight = _ensure_tensor(weight)
        if bias is not None:
            bias = _ensure_tensor(bias)

        batch_size, in_channels, height, width = input.shape
        in_channels, out_channels, kernel_h, kernel_w = weight.shape

        stride_h, stride_w = stride
        padding_h, padding_w = padding
        output_padding_h, output_padding_w = output_padding

        # Calculate output dimensions
        out_height = (
            (height - 1) * stride_h - 2 * padding_h + kernel_h + output_padding_h
        )
        out_width = (width - 1) * stride_w - 2 * padding_w + kernel_w + output_padding_w

        output_data = np.zeros((batch_size, out_channels, out_height, out_width))

        # Perform transposed convolution
        for b in range(batch_size):
            for c_out in range(out_channels):
                for c_in in range(in_channels):
                    for h in range(height):
                        for w in range(width):
                            # Calculate output position in transposed convolution
                            h_start = h * stride_h - padding_h
                            w_start = w * stride_w - padding_w

                            # For each position in the kernel
                            for kh in range(kernel_h):
                                for kw in range(kernel_w):
                                    h_out_pos = h_start + kh
                                    w_out_pos = w_start + kw

                                    # Only update if within output bounds
                                    if (0 <= h_out_pos < out_height) and (
                                        0 <= w_out_pos < out_width
                                    ):
                                        # Note weight indexing is flipped for transposed convolution
                                        output_data[b, c_out, h_out_pos, w_out_pos] += (
                                            input.data[b, c_in, h, w]
                                            * weight.data[c_in, c_out, kh, kw]
                                        )

        if bias is not None:
            reshaped_bias = bias.data.reshape(1, out_channels, 1, 1)
            output_data += reshaped_bias

        ctx.save_for_backwards(input, weight, bias)
        ctx.save_data(
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            input_shape=input.shape,
            weight_shape=weight.shape,
            output_shape=output_data.shape,
        )

        return Tensor(
            output_data, requires_grad=input.requires_grad or weight.requires_grad
        )

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for TransposedConv2D.

        Args:
            grad_output: Gradient of loss with respect to output

        Returns:
            Gradients for input, weight, bias, stride, padding, and output_padding
        """
        input, weight, bias = ctx.saved_tensors
        stride = ctx.saved_data["stride"]
        padding = ctx.saved_data["padding"]
        output_padding = ctx.saved_data["output_padding"]
        input_shape = ctx.saved_data["input_shape"]
        weight_shape = ctx.saved_data["weight_shape"]
        output_shape = ctx.saved_data["output_shape"]

        batch_size, in_channels, height, width = input_shape
        in_channels, out_channels, kernel_h, kernel_w = weight_shape
        _, _, out_height, out_width = output_shape

        stride_h, stride_w = stride
        padding_h, padding_w = padding

        grad_input_data = np.zeros_like(input.data)
        grad_weight_data = np.zeros_like(weight.data)
        grad_bias_data = None if bias is None else np.zeros_like(bias.data)

        if bias is not None and bias.requires_grad:
            grad_bias_data = np.sum(grad_output.data, axis=(0, 2, 3))

        if weight.requires_grad:
            for b in range(batch_size):
                for c_out in range(out_channels):
                    for c_in in range(in_channels):
                        for h in range(height):
                            for w in range(width):
                                h_start = h * stride_h - padding_h
                                w_start = w * stride_w - padding_w

                                for kh in range(kernel_h):
                                    for kw in range(kernel_w):
                                        h_out_pos = h_start + kh
                                        w_out_pos = w_start + kw

                                        if (0 <= h_out_pos < out_height) and (
                                            0 <= w_out_pos < out_width
                                        ):
                                            grad_weight_data[c_in, c_out, kh, kw] += (
                                                input.data[b, c_in, h, w]
                                                * grad_output.data[
                                                    b, c_out, h_out_pos, w_out_pos
                                                ]
                                            )

        # Compute gradient for input - effectively a normal convolution
        if input.requires_grad:
            for b in range(batch_size):
                for c_in in range(in_channels):
                    for h in range(height):
                        for w in range(width):
                            h_start = h * stride_h - padding_h
                            w_start = w * stride_w - padding_w

                            for c_out in range(out_channels):
                                for kh in range(kernel_h):
                                    for kw in range(kernel_w):
                                        h_out_pos = h_start + kh
                                        w_out_pos = w_start + kw

                                        if (0 <= h_out_pos < out_height) and (
                                            0 <= w_out_pos < out_width
                                        ):
                                            grad_input_data[b, c_in, h, w] += (
                                                weight.data[c_in, c_out, kh, kw]
                                                * grad_output.data[
                                                    b, c_out, h_out_pos, w_out_pos
                                                ]
                                            )

        grad_input = Tensor(grad_input_data) if input.requires_grad else None
        grad_weight = Tensor(grad_weight_data) if weight.requires_grad else None
        grad_bias = (
            Tensor(grad_bias_data) if bias is not None and bias.requires_grad else None
        )

        return grad_input, grad_weight, grad_bias, None, None, None


class DepthwiseSeparableConv2DFunction(Function):
    @staticmethod
    def forward(
        ctx,
        input,
        depthwise_weight,
        pointwise_weight,
        bias,
        stride,
        padding,
        depth_multiplier,
    ):
        """
        Forward pass for DepthwiseSeparableConv2D.

        Args:
            input: Input tensor of shape (batch_size, in_channels, height, width)
            depthwise_weight: Depthwise weight tensor of shape (depth_channels, 1, kernel_h, kernel_w)
            pointwise_weight: Pointwise weight tensor of shape (out_channels, depth_channels, 1, 1)
            bias: Bias tensor of shape (out_channels) or None
            stride: Convolution stride (tuple of 2 integers)
            padding: Convolution padding (tuple of 2 integers)
            depth_multiplier: Multiplier for the depth dimension

        Returns:
            Output tensor of shape (batch_size, out_channels, out_height, out_width)
        """
        input = _ensure_tensor(input)
        depthwise_weight = _ensure_tensor(depthwise_weight)
        pointwise_weight = _ensure_tensor(pointwise_weight)
        if bias is not None:
            bias = _ensure_tensor(bias)

        batch_size, in_channels, height, width = input.shape
        depth_channels, _, kernel_h, kernel_w = depthwise_weight.shape
        out_channels, _, _, _ = pointwise_weight.shape

        stride_h, stride_w = stride
        padding_h, padding_w = padding

        out_height = (height + 2 * padding_h - kernel_h) // stride_h + 1
        out_width = (width + 2 * padding_w - kernel_w) // stride_w + 1

        if padding_h > 0 or padding_w > 0:
            input_padded = np.pad(
                input.data,
                ((0, 0), (0, 0), (padding_h, padding_h), (padding_w, padding_w)),
                mode="constant",
            )
        else:
            input_padded = input.data

        # Initialize intermediate output for depthwise convolution
        depth_out = np.zeros((batch_size, depth_channels, out_height, out_width))

        for b in range(batch_size):
            for c_in in range(in_channels):
                for c_depth in range(depth_multiplier):
                    c_out = c_in * depth_multiplier + c_depth
                    for h in range(out_height):
                        h_start = h * stride_h
                        for w in range(out_width):
                            w_start = w * stride_w

                            region = input_padded[
                                b,
                                c_in,
                                h_start : h_start + kernel_h,
                                w_start : w_start + kernel_w,
                            ]

                            # Apply depthwise filter
                            depth_out[b, c_out, h, w] = np.sum(
                                region * depthwise_weight.data[c_out, 0]
                            )

        # Perform pointwise convolution (1x1 convolution to mix channels)
        output_data = np.zeros((batch_size, out_channels, out_height, out_width))

        for b in range(batch_size):
            for h in range(out_height):
                for w in range(out_width):
                    # Extract the depth channels at this position
                    depth_channels_data = depth_out[b, :, h, w]

                    # For each output channel
                    for c_out in range(out_channels):
                        # Apply 1x1 convolution
                        output_data[b, c_out, h, w] = np.sum(
                            depth_channels_data * pointwise_weight.data[c_out, :, 0, 0]
                        )

        if bias is not None:
            reshaped_bias = bias.data.reshape(1, out_channels, 1, 1)
            output_data += reshaped_bias

        ctx.save_for_backwards(input, depthwise_weight, pointwise_weight, bias)
        ctx.save_data(
            stride=stride,
            padding=padding,
            depth_multiplier=depth_multiplier,
            input_shape=input.shape,
            depthwise_weight_shape=depthwise_weight.shape,
            pointwise_weight_shape=pointwise_weight.shape,
            output_shape=output_data.shape,
            depth_out=depth_out,
            input_padded=input_padded,
        )

        return Tensor(
            output_data,
            requires_grad=input.requires_grad
            or depthwise_weight.requires_grad
            or pointwise_weight.requires_grad,
        )

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for DepthwiseSeparableConv2D.

        Args:
            grad_output: Gradient of loss with respect to output

        Returns:
            Gradients for input, depthwise_weight, pointwise_weight, bias, stride, padding, and depth_multiplier
        """
        input, depthwise_weight, pointwise_weight, bias = ctx.saved_tensors
        stride = ctx.saved_data["stride"]
        padding = ctx.saved_data["padding"]
        depth_multiplier = ctx.saved_data["depth_multiplier"]
        input_shape = ctx.saved_data["input_shape"]
        depthwise_weight_shape = ctx.saved_data["depthwise_weight_shape"]
        pointwise_weight_shape = ctx.saved_data["pointwise_weight_shape"]
        output_shape = ctx.saved_data["output_shape"]
        depth_out = ctx.saved_data["depth_out"]
        input_padded = ctx.saved_data["input_padded"]

        batch_size, in_channels, height, width = input_shape
        depth_channels, _, kernel_h, kernel_w = depthwise_weight_shape
        out_channels, _, _, _ = pointwise_weight_shape
        _, _, out_height, out_width = output_shape

        stride_h, stride_w = stride
        padding_h, padding_w = padding

        grad_input_data = np.zeros_like(input.data)
        grad_depthwise_weight_data = np.zeros_like(depthwise_weight.data)
        grad_pointwise_weight_data = np.zeros_like(pointwise_weight.data)
        grad_bias_data = None if bias is None else np.zeros_like(bias.data)

        if bias is not None and bias.requires_grad:
            grad_bias_data = np.sum(grad_output.data, axis=(0, 2, 3))

        # First, compute gradient for pointwise convolution
        grad_depth_out = np.zeros_like(depth_out)

        if pointwise_weight.requires_grad:
            for b in range(batch_size):
                for h in range(out_height):
                    for w in range(out_width):
                        depth_channels_data = depth_out[b, :, h, w]

                        for c_out in range(out_channels):
                            # Gradient for pointwise weights
                            grad_pointwise_weight_data[c_out, :, 0, 0] += (
                                depth_channels_data * grad_output.data[b, c_out, h, w]
                            )

                            # Gradient for depth output
                            grad_depth_out[b, :, h, w] += (
                                pointwise_weight.data[c_out, :, 0, 0]
                                * grad_output.data[b, c_out, h, w]
                            )

        # Compute gradient for depthwise convolution
        if depthwise_weight.requires_grad or input.requires_grad:
            grad_input_padded = np.zeros_like(input_padded)

            for b in range(batch_size):
                for c_in in range(in_channels):
                    for c_depth in range(depth_multiplier):
                        c_out = c_in * depth_multiplier + c_depth

                        if depthwise_weight.requires_grad:
                            for h in range(out_height):
                                h_start = h * stride_h
                                for w in range(out_width):
                                    w_start = w * stride_w

                                    # Extract region for gradient computation
                                    region = input_padded[
                                        b,
                                        c_in,
                                        h_start : h_start + kernel_h,
                                        w_start : w_start + kernel_w,
                                    ]

                                    # Gradient for depthwise weights
                                    grad_depthwise_weight_data[c_out, 0] += (
                                        region * grad_depth_out[b, c_out, h, w]
                                    )

                        if input.requires_grad:
                            for h in range(out_height):
                                h_start = h * stride_h
                                for w in range(out_width):
                                    w_start = w * stride_w

                                    # Gradient for input
                                    grad_input_padded[
                                        b,
                                        c_in,
                                        h_start : h_start + kernel_h,
                                        w_start : w_start + kernel_w,
                                    ] += (
                                        depthwise_weight.data[c_out, 0]
                                        * grad_depth_out[b, c_out, h, w]
                                    )

            if padding_h > 0 or padding_w > 0:
                grad_input_data = grad_input_padded[
                    :, :, padding_h : padding_h + height, padding_w : padding_w + width
                ]
            else:
                grad_input_data = grad_input_padded

        grad_input = Tensor(grad_input_data) if input.requires_grad else None
        grad_depthwise_weight = (
            Tensor(grad_depthwise_weight_data)
            if depthwise_weight.requires_grad
            else None
        )
        grad_pointwise_weight = (
            Tensor(grad_pointwise_weight_data)
            if pointwise_weight.requires_grad
            else None
        )
        grad_bias = (
            Tensor(grad_bias_data) if bias is not None and bias.requires_grad else None
        )

        return (
            grad_input,
            grad_depthwise_weight,
            grad_pointwise_weight,
            grad_bias,
            None,
            None,
            None,
        )


class DilatedConv2DFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding, dilation):
        """
        Forward pass for DilatedConv2D.

        Args:
            input: Input tensor of shape (batch_size, in_channels, height, width)
            weight: Weight tensor of shape (out_channels, in_channels, kernel_h, kernel_w)
            bias: Bias tensor of shape (out_channels) or None
            stride: Convolution stride (tuple of 2 integers)
            padding: Convolution padding (tuple of 2 integers)
            dilation: Dilation factor (tuple of 2 integers)

        Returns:
            Output tensor of shape (batch_size, out_channels, out_height, out_width)
        """
        input = _ensure_tensor(input)
        weight = _ensure_tensor(weight)
        if bias is not None:
            bias = _ensure_tensor(bias)

        batch_size, in_channels, height, width = input.shape
        out_channels, _, kernel_h, kernel_w = weight.shape

        stride_h, stride_w = stride
        padding_h, padding_w = padding
        dilation_h, dilation_w = dilation

        out_height = (
            height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1
        ) // stride_h + 1
        out_width = (
            width + 2 * padding_w - dilation_w * (kernel_w - 1) - 1
        ) // stride_w + 1

        if padding_h > 0 or padding_w > 0:
            input_padded = np.pad(
                input.data,
                ((0, 0), (0, 0), (padding_h, padding_h), (padding_w, padding_w)),
                mode="constant",
            )
        else:
            input_padded = input.data

        output_data = np.zeros((batch_size, out_channels, out_height, out_width))

        for b in range(batch_size):
            for c_out in range(out_channels):
                for h in range(out_height):
                    h_start = h * stride_h
                    for w in range(out_width):
                        w_start = w * stride_w

                        val = 0.0

                        for c_in in range(in_channels):
                            for kh in range(kernel_h):
                                h_offset = kh * dilation_h

                                for kw in range(kernel_w):
                                    w_offset = kw * dilation_w

                                    h_pos = h_start + h_offset
                                    w_pos = w_start + w_offset

                                    if (
                                        h_pos < input_padded.shape[2]
                                        and w_pos < input_padded.shape[3]
                                    ):
                                        input_val = input_padded[b, c_in, h_pos, w_pos]
                                        kernel_val = weight.data[c_out, c_in, kh, kw]
                                        val += input_val * kernel_val

                        output_data[b, c_out, h, w] = val

        if bias is not None:
            reshaped_bias = bias.data.reshape(1, out_channels, 1, 1)
            output_data += reshaped_bias

        ctx.save_for_backwards(input, weight, bias)
        ctx.save_data(
            stride=stride,
            padding=padding,
            dilation=dilation,
            input_shape=input.shape,
            weight_shape=weight.shape,
            output_shape=output_data.shape,
            input_padded=input_padded,
        )

        return Tensor(
            output_data, requires_grad=input.requires_grad or weight.requires_grad
        )

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for DilatedConv2D.

        Args:
            grad_output: Gradient of loss with respect to output

        Returns:
            Gradients for input, weight, bias, stride, padding, and dilation
        """
        input, weight, bias = ctx.saved_tensors
        stride = ctx.saved_data["stride"]
        padding = ctx.saved_data["padding"]
        dilation = ctx.saved_data["dilation"]
        input_shape = ctx.saved_data["input_shape"]
        weight_shape = ctx.saved_data["weight_shape"]
        output_shape = ctx.saved_data["output_shape"]
        input_padded = ctx.saved_data["input_padded"]

        batch_size, in_channels, height, width = input_shape
        out_channels, _, kernel_h, kernel_w = weight_shape
        _, _, out_height, out_width = output_shape

        stride_h, stride_w = stride
        padding_h, padding_w = padding
        dilation_h, dilation_w = dilation

        grad_input_data = np.zeros_like(input.data)
        grad_weight_data = np.zeros_like(weight.data)
        grad_bias_data = None if bias is None else np.zeros_like(bias.data)

        if bias is not None and bias.requires_grad:
            # Sum gradient over batch and spatial dimensions
            grad_bias_data = np.sum(grad_output.data, axis=(0, 2, 3))

        # Compute gradient for weights
        if weight.requires_grad:
            for b in range(batch_size):
                for c_out in range(out_channels):
                    for c_in in range(in_channels):
                        for h in range(out_height):
                            h_start = h * stride_h
                            for w in range(out_width):
                                w_start = w * stride_w

                                # For each kernel position
                                for kh in range(kernel_h):
                                    h_offset = kh * dilation_h
                                    for kw in range(kernel_w):
                                        w_offset = kw * dilation_w

                                        # Get input value at dilated position
                                        h_pos = h_start + h_offset
                                        w_pos = w_start + w_offset

                                        if (
                                            h_pos < input_padded.shape[2]
                                            and w_pos < input_padded.shape[3]
                                        ):
                                            grad_weight_data[c_out, c_in, kh, kw] += (
                                                input_padded[b, c_in, h_pos, w_pos]
                                                * grad_output.data[b, c_out, h, w]
                                            )

        # Compute gradient for input
        if input.requires_grad:
            grad_input_padded = np.zeros_like(input_padded)

            for b in range(batch_size):
                for c_out in range(out_channels):
                    for c_in in range(in_channels):
                        for h in range(out_height):
                            h_start = h * stride_h
                            for w in range(out_width):
                                w_start = w * stride_w

                                for kh in range(kernel_h):
                                    h_offset = kh * dilation_h
                                    for kw in range(kernel_w):
                                        w_offset = kw * dilation_w

                                        h_pos = h_start + h_offset
                                        w_pos = w_start + w_offset

                                        if (
                                            h_pos < input_padded.shape[2]
                                            and w_pos < input_padded.shape[3]
                                        ):
                                            grad_input_padded[
                                                b, c_in, h_pos, w_pos
                                            ] += (
                                                weight.data[c_out, c_in, kh, kw]
                                                * grad_output.data[b, c_out, h, w]
                                            )

            if padding_h > 0 or padding_w > 0:
                grad_input_data = grad_input_padded[
                    :, :, padding_h : padding_h + height, padding_w : padding_w + width
                ]
            else:
                grad_input_data = grad_input_padded

        grad_input = Tensor(grad_input_data) if input.requires_grad else None
        grad_weight = Tensor(grad_weight_data) if weight.requires_grad else None
        grad_bias = (
            Tensor(grad_bias_data) if bias is not None and bias.requires_grad else None
        )

        return grad_input, grad_weight, grad_bias, None, None, None


# Pooling functions for autograd engine
class MaxPool2DFunction(Function):
    @staticmethod
    def forward(ctx, x, kernel_size, stride=None, padding=0):
        """
        Forward pass for 2D Max Pooling
        Args:
            x: Input tensor of the shape (batch_size, channels, height, width)
            kernel_size: Size of the pooling window (int or tuple)
            stride: Stride of the pooling window (int or tuple)
            padding: Padding to be added on boht sides (int or tuple)
        """

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        if stride is None:
            stride = kernel_size
        elif isinstance(stride, int):
            stride = (stride, stride)

        if isinstance(padding, int):
            padding = (padding, padding)

        batch_size, channels, height, width = x.shape

        out_height = ((height + 2 * padding[0] - kernel_size[0]) // stride[0]) + 1
        out_width = ((width + 2 * padding[1] - kernel_size[1]) // stride[1]) + 1

        if padding[0] > 0 or padding[1] > 0:
            x_padded = np.pad(
                x.data,
                ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])),
                mode="constant",
                constant_values=float("-inf"),  # Use -inf for max pooling padding
            )
        else:
            x_padded = x.data

        # Initialize output
        output = np.zeros((batch_size, channels, out_height, out_width))

        # Store indices of max values for backward pass
        max_indices = np.zeros(
            (batch_size, channels, out_height, out_width, 2), dtype=int
        )

        # Perform max pooling
        for b in range(batch_size):
            for c in range(channels):
                for h_out in range(out_height):
                    for w_out in range(out_width):
                        h_start = h_out * stride[0]
                        h_end = min(h_start + kernel_size[0], x_padded.shape[2])
                        w_start = w_out * stride[1]
                        w_end = min(w_start + kernel_size[1], x_padded.shape[3])

                        # Extract pool region
                        pool_region = x_padded[b, c, h_start:h_end, w_start:w_end]

                        # Find max value and its index
                        output[b, c, h_out, w_out] = np.max(pool_region)

                        # Find index of max value (relative to top-left of pool window)
                        flat_idx = np.argmax(pool_region)
                        rel_h, rel_w = np.unravel_index(flat_idx, pool_region.shape)

                        # Store absolute indices (in padded input)
                        max_indices[b, c, h_out, w_out, 0] = h_start + rel_h
                        max_indices[b, c, h_out, w_out, 1] = w_start + rel_w

        # Save indices and parameters for backward pass
        ctx.save_data(
            max_indices=max_indices,
            input_shape=x.shape,
            padded_shape=x_padded.shape,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        return Tensor(output)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for 2D max pooling

        Args:
            grad_output: Gradient of loss with respect to output

        Returns:
            Gradient of loss with respect to input and None for parameters
        """
        # Get saved data
        max_indices = ctx.saved_data["max_indices"]
        input_shape = ctx.saved_data["input_shape"]
        padded_shape = ctx.saved_data["padded_shape"]
        padding = ctx.saved_data["padding"]

        # Initialize gradient for padded input
        grad_padded = np.zeros(padded_shape)

        # Get output dimensions
        batch_size, channels, out_height, out_width = grad_output.shape

        # Distribute gradients to max value locations
        for b in range(batch_size):
            for c in range(channels):
                for h_out in range(out_height):
                    for w_out in range(out_width):
                        # Get position of max value in padded input
                        h_max, w_max = max_indices[b, c, h_out, w_out]

                        # Add gradient to that position
                        grad_padded[b, c, h_max, w_max] += grad_output.data[
                            b, c, h_out, w_out
                        ]

        # Remove padding if necessary
        if padding[0] > 0 or padding[1] > 0:
            grad_input = grad_padded[
                :,
                :,
                padding[0] : padded_shape[2] - padding[0],
                padding[1] : padded_shape[3] - padding[1],
            ]
        else:
            grad_input = grad_padded

        # Return gradients for inputs and None for parameters
        return Tensor(grad_input), None, None, None


class AvgPool2DFunction(Function):
    @staticmethod
    def forward(ctx, x, kernel_size, stride=None, padding=0, count_include_pad=True):
        """
        Forward pass for 2D average pooling

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            kernel_size: Size of the pooling window (int or tuple)
            stride: Stride of the pooling window (int or tuple)
            padding: Padding to be added on both sides (int or tuple)
            count_include_pad: Whether to include padding in averaging calculation

        Returns:
            Output tensor of shape (batch_size, channels, out_height, out_width)
        """
        # Handle scalar and tuple inputs
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        if stride is None:
            stride = kernel_size
        elif isinstance(stride, int):
            stride = (stride, stride)

        if isinstance(padding, int):
            padding = (padding, padding)

        # Get input dimensions
        batch_size, channels, height, width = x.shape

        # Calculate output dimensions
        out_height = ((height + 2 * padding[0] - kernel_size[0]) // stride[0]) + 1
        out_width = ((width + 2 * padding[1] - kernel_size[1]) // stride[1]) + 1

        # Apply padding if needed
        if padding[0] > 0 or padding[1] > 0:
            x_padded = np.pad(
                x.data,
                ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])),
                mode="constant",
                constant_values=0,  # Use 0 for average pooling padding
            )
        else:
            x_padded = x.data

        # Initialize output
        output = np.zeros((batch_size, channels, out_height, out_width))

        # Store window shapes for backward pass
        window_shapes = np.zeros((out_height, out_width, 2), dtype=int)

        # Perform average pooling
        for h_out in range(out_height):
            for w_out in range(out_width):
                h_start = h_out * stride[0]
                h_end = min(h_start + kernel_size[0], x_padded.shape[2])
                w_start = w_out * stride[1]
                w_end = min(w_start + kernel_size[1], x_padded.shape[3])

                # Get actual window size (might be smaller at edges)
                window_h = h_end - h_start
                window_w = w_end - w_start
                window_shapes[h_out, w_out] = [window_h, window_w]

                # Calculate average
                if count_include_pad:
                    # Divide by the full kernel size
                    divisor = kernel_size[0] * kernel_size[1]
                else:
                    # Divide by the actual window size
                    divisor = window_h * window_w

                # Extract pool region and compute average
                for b in range(batch_size):
                    for c in range(channels):
                        pool_region = x_padded[b, c, h_start:h_end, w_start:w_end]
                        output[b, c, h_out, w_out] = np.sum(pool_region) / divisor

        # Save parameters for backward pass
        ctx.save_data(
            input_shape=x.shape,
            padded_shape=x_padded.shape,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            count_include_pad=count_include_pad,
            window_shapes=window_shapes,
        )

        return Tensor(output)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for 2D average pooling

        Args:
            grad_output: Gradient of loss with respect to output

        Returns:
            Gradient of loss with respect to input and None for parameters
        """
        # Get saved data
        input_shape = ctx.saved_data["input_shape"]
        padded_shape = ctx.saved_data["padded_shape"]
        kernel_size = ctx.saved_data["kernel_size"]
        stride = ctx.saved_data["stride"]
        padding = ctx.saved_data["padding"]
        count_include_pad = ctx.saved_data["count_include_pad"]
        window_shapes = ctx.saved_data["window_shapes"]

        # Initialize gradient for padded input
        grad_padded = np.zeros(padded_shape)

        # Get output dimensions
        batch_size, channels, out_height, out_width = grad_output.shape

        # Distribute gradients evenly to all positions in each window
        for h_out in range(out_height):
            for w_out in range(out_width):
                h_start = h_out * stride[0]
                h_end = min(h_start + kernel_size[0], padded_shape[2])
                w_start = w_out * stride[1]
                w_end = min(w_start + kernel_size[1], padded_shape[3])

                # Get window size for this output position
                window_h, window_w = window_shapes[h_out, w_out]

                # Calculate gradient scaling factor
                if count_include_pad:
                    # Division was by full kernel size
                    scale = 1.0 / (kernel_size[0] * kernel_size[1])
                else:
                    # Division was by actual window size
                    scale = 1.0 / (window_h * window_w)

                # Add scaled gradient to all positions in the window
                for b in range(batch_size):
                    for c in range(channels):
                        grad_value = grad_output.data[b, c, h_out, w_out] * scale
                        grad_padded[b, c, h_start:h_end, w_start:w_end] += grad_value

        # Remove padding if necessary
        if padding[0] > 0 or padding[1] > 0:
            grad_input = grad_padded[
                :,
                :,
                padding[0] : padded_shape[2] - padding[0],
                padding[1] : padded_shape[3] - padding[1],
            ]
        else:
            grad_input = grad_padded

        # Return gradients for inputs and None for parameters
        return Tensor(grad_input), None, None, None, None


class AdaptiveMaxPool2DFunction(Function):
    @staticmethod
    def forward(ctx, x, output_size):
        """
        Forward pass for 2D adaptive max pooling

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            output_size: Tuple of (output_height, output_width)

        Returns:
            Output tensor of shape (batch_size, channels, output_height, output_width)
        """
        # Get input dimensions
        batch_size, channels, in_height, in_width = x.shape
        out_height, out_width = output_size

        # Calculate stride for each output position
        stride_h = in_height / out_height
        stride_w = in_width / out_width

        # Initialize output
        output = np.zeros((batch_size, channels, out_height, out_width))

        # Store max indices for backward pass
        max_indices = np.zeros(
            (batch_size, channels, out_height, out_width, 2), dtype=int
        )

        # Perform adaptive max pooling
        for b in range(batch_size):
            for c in range(channels):
                for h_out in range(out_height):
                    # Calculate input window boundaries for this output position
                    h_start = int(np.floor(h_out * stride_h))
                    h_end = int(np.ceil((h_out + 1) * stride_h))
                    h_end = min(h_end, in_height)  # Ensure we don't go out of bounds

                    for w_out in range(out_width):
                        # Calculate input window boundaries for this output position
                        w_start = int(np.floor(w_out * stride_w))
                        w_end = int(np.ceil((w_out + 1) * stride_w))
                        w_end = min(w_end, in_width)  # Ensure we don't go out of bounds

                        # Extract pool region
                        pool_region = x.data[b, c, h_start:h_end, w_start:w_end]

                        # Find max value and its position
                        max_val = np.max(pool_region)
                        output[b, c, h_out, w_out] = max_val

                        # Find position of max value
                        max_pos = np.unravel_index(
                            np.argmax(pool_region), pool_region.shape
                        )

                        # Store absolute indices
                        max_indices[b, c, h_out, w_out, 0] = h_start + max_pos[0]
                        max_indices[b, c, h_out, w_out, 1] = w_start + max_pos[1]

        # Save for backward
        ctx.save_data(input_shape=x.shape, max_indices=max_indices)

        return Tensor(output)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for adaptive max pooling

        Args:
            grad_output: Gradient of loss with respect to output

        Returns:
            Gradient of loss with respect to input and None for output_size
        """
        # Get saved data
        input_shape = ctx.saved_data["input_shape"]
        max_indices = ctx.saved_data["max_indices"]

        # Initialize gradient for input
        grad_input = np.zeros(input_shape)

        # Get dimensions
        batch_size, channels = input_shape[:2]
        out_height, out_width = grad_output.shape[2:]

        # Distribute gradients to max value locations
        for b in range(batch_size):
            for c in range(channels):
                for h_out in range(out_height):
                    for w_out in range(out_width):
                        # Get max position
                        h_max, w_max = max_indices[b, c, h_out, w_out]

                        # Add gradient to that position
                        grad_input[b, c, h_max, w_max] += grad_output.data[
                            b, c, h_out, w_out
                        ]

        return Tensor(grad_input), None


class AdaptiveAvgPool2DFunction(Function):
    @staticmethod
    def forward(ctx, x, output_size):
        """
        Forward pass for 2D adaptive average pooling

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            output_size: Tuple of (output_height, output_width)

        Returns:
            Output tensor of shape (batch_size, channels, output_height, output_width)
        """
        # Get input dimensions
        print(f"DEBUG: output_size = {output_size}, type = {type(output_size)}")
        batch_size, channels, in_height, in_width = x.shape
        out_height, out_width = output_size

        # Calculate stride and kernel size for each output position
        stride_h = in_height / out_height
        stride_w = in_width / out_width

        # Initialize output
        output = np.zeros((batch_size, channels, out_height, out_width))

        # Store window indices for backward pass
        window_indices = np.zeros((out_height, out_width, 4), dtype=int)

        # Perform adaptive average pooling
        for h_out in range(out_height):
            h_start = int(np.floor(h_out * in_height / out_height))
            h_end = int(np.ceil((h_out + 1) * in_height / out_height))
            
            for w_out in range(out_width):
                w_start = int(np.floor(w_out * in_width / out_width))
                w_end = int(np.ceil((w_out + 1) * in_width / out_width))
                window_indices[h_out, w_out] = [h_start, h_end, w_start, w_end]

                # Calculate average for each batch and channel
                for b in range(batch_size):
                    for c in range(channels):
                        # Extract pool region
                        pool_region = x.data[b, c, h_start:h_end, w_start:w_end]
                        # Compute average
                        output[b, c, h_out, w_out] = np.mean(pool_region)

        # Save for backward
        ctx.save_data(input_shape=x.shape, window_indices=window_indices)

        return Tensor(output, requires_grad=x.requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for adaptive average pooling

        Args:
            grad_output: Gradient of loss with respect to output

        Returns:
            Gradient of loss with respect to input and None for output_size
        """
        # Get saved data
        input_shape = ctx.saved_data["input_shape"]
        window_indices = ctx.saved_data["window_indices"]

        # Initialize gradient for input
        grad_input = np.zeros(input_shape)

        # Get dimensions
        batch_size, channels = input_shape[:2]
        out_height, out_width = grad_output.shape[2:]

        # Distribute gradients to each position in the input
        for h_out in range(out_height):
            for w_out in range(out_width):
                # Get window indices
                h_start, h_end, w_start, w_end = window_indices[h_out, w_out]

                # Calculate number of elements in the window
                num_elements = (h_end - h_start) * (w_end - w_start)

                # Distribute gradient evenly across window
                scale = 1.0 / num_elements

                for b in range(batch_size):
                    for c in range(channels):
                        # Get gradient
                        grad_val = grad_output.data[b, c, h_out, w_out] * scale

                        # Add to all positions in the window
                        grad_input[b, c, h_start:h_end, w_start:w_end] += grad_val

        return Tensor(grad_input), None


class GlobalAvgPool2DFunction(Function):
    @staticmethod
    def forward(ctx, x):
        """
        Forward pass for 2D global average pooling

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Output tensor of shape (batch_size, channels, 1, 1)
        """
        # Get input dimensions
        batch_size, channels, height, width = x.shape

        # Compute global average
        output = np.mean(x.data, axis=(2, 3), keepdims=True)

        # Save dimensions for backward pass
        ctx.save_data(input_shape=x.shape)

        return Tensor(output)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for global average pooling

        Args:
            grad_output: Gradient of loss with respect to output (shape: batch_size, channels, 1, 1)

        Returns:
            Gradient of loss with respect to input
        """
        # Get saved data
        input_shape = ctx.saved_data["input_shape"]

        # Get height and width
        height, width = input_shape[2:]
        num_elements = height * width

        # Scale gradient by 1/num_elements
        scale = 1.0 / num_elements

        # Distribute gradient evenly to all spatial positions
        grad_input = np.ones(input_shape) * scale * grad_output.data

        return Tensor(grad_input)


class GlobalMaxPool2DFunction(Function):
    @staticmethod
    def forward(ctx, x):
        """
        Forward pass for 2D global max pooling

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Output tensor of shape (batch_size, channels, 1, 1)
        """
        # Get input dimensions
        batch_size, channels, height, width = x.shape

        # Compute global max
        output = np.max(x.data, axis=(2, 3), keepdims=True)

        # Find positions of max values for each (batch, channel)
        max_indices = np.zeros((batch_size, channels, 2), dtype=int)

        for b in range(batch_size):
            for c in range(channels):
                # Flatten spatial dimensions
                flattened = x.data[b, c].reshape(-1)
                # Find index of max value
                flat_idx = np.argmax(flattened)
                # Convert to 2D index
                h_idx, w_idx = np.unravel_index(flat_idx, (height, width))
                # Store indices
                max_indices[b, c] = [h_idx, w_idx]

        # Save dimensions and max indices for backward pass
        ctx.save_data(input_shape=x.shape, max_indices=max_indices)

        return Tensor(output)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for global max pooling

        Args:
            grad_output: Gradient of loss with respect to output (shape: batch_size, channels, 1, 1)

        Returns:
            Gradient of loss with respect to input
        """
        # Get saved data
        input_shape = ctx.saved_data["input_shape"]
        max_indices = ctx.saved_data["max_indices"]

        # Initialize gradient for input
        grad_input = np.zeros(input_shape)

        # Get dimensions
        batch_size, channels = input_shape[:2]

        # Distribute gradients to max value locations
        for b in range(batch_size):
            for c in range(channels):
                # Get position of max value
                h_max, w_max = max_indices[b, c]

                # Add gradient to that position
                grad_input[b, c, h_max, w_max] = grad_output.data[b, c, 0, 0]

        return Tensor(grad_input)
