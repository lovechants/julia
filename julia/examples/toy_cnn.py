# toy_cnn.py (Refactored CNN class)
import numpy as np
from julia.core.tensor import Tensor
from julia.core.nn.conv import Conv2D
from julia.core.nn.pooling import MaxPool2D
from julia.core.nn.layers import (
    Linear,
    Sequential,
    Flatten,
    Dropout,
    Layer,
)  # Import Layer
from julia.core.optim import SGD


class SimpleCNN(Sequential):
    def __init__(self, in_channels=1, num_classes=10):
        # Define layers
        conv1 = Conv2D(in_channels, 16, 3, 1, 1)
        relu1 = lambda t: t.relu()
        pool1 = MaxPool2D(2, 2)

        conv2 = Conv2D(16, 32, 3, 1, 1)
        relu2 = lambda t: t.relu()
        pool2 = MaxPool2D(2, 2)

        flatten = Flatten()
        # Calculate flattened size (assuming 28x28 input)
        # 28x28 -> Conv1(pad=1) -> 28x28 -> Pool1(k=2,s=2) -> 14x14
        # 14x14 -> Conv2(pad=1) -> 14x14 -> Pool2(k=2,s=2) -> 7x7
        # Flattened size = 32 * 7 * 7 = 1568
        fc1 = Linear(32 * 7 * 7, 128)
        relu3 = lambda t: t.relu()
        dropout = Dropout(0.5)
        fc2 = Linear(128, num_classes)

        super().__init__(
            conv1, relu1, pool1, conv2, relu2, pool2, flatten, fc1, relu3, dropout, fc2
        )


def create_toy_dataset(num_samples=100, num_classes=10):
    # Create random 28x28 grayscale images
    X = np.random.randn(num_samples, 1, 28, 28)
    # Create random labels
    y = np.random.randint(0, num_classes, size=num_samples)
    # Convert to one-hot encoding
    y_one_hot = np.zeros((num_samples, num_classes))
    y_one_hot[np.arange(num_samples), y] = 1
    return X, y_one_hot


def cross_entropy_loss(predictions: Tensor, targets_one_hot: np.ndarray):
    # Ensure targets is numpy array
    if isinstance(targets_one_hot, Tensor):
        targets_one_hot = targets_one_hot.data

    # Softmax calculation (numerically stable)
    logits = predictions.data
    max_logits = np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    # NLL Loss
    N = logits.shape[0]
    log_probs = np.log(probs + 1e-9)  # Add epsilon for stability
    loss_value = -np.sum(targets_one_hot * log_probs) / N

    # Gradient of Loss w.r.t Logits (predictions)
    # dLoss/dLogits = Probs - Targets
    grad_data = (probs - targets_one_hot) / N

    # Return loss value and gradient Tensor
    # Ideally, loss calculation itself should use Tensor ops for autograd
    # For now, we return the pre-computed gradient
    loss_tensor = Tensor(np.array(loss_value))  # Loss is scalar
    # We need to attach the gradient to the *predictions* tensor later
    # So, just return the gradient data for now.
    return loss_value, grad_data


def train(model: Layer, X: np.ndarray, y: np.ndarray, epochs=5, batch_size=32, lr=0.01):
    optimizer = SGD(model.parameters(), lr=lr)

    num_samples = X.shape[0]

    for epoch in range(epochs):
        total_loss = 0
        indices = np.random.permutation(num_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        num_batches = num_samples // batch_size

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            X_batch = X_shuffled[start_idx:end_idx]
            y_batch = y_shuffled[start_idx:end_idx]  # One-hot targets

            X_batch_tensor = Tensor(
                X_batch, requires_grad=True
            )  # Input needs grad usually

            # Zero gradients before forward/backward
            optimizer.zero_grad()  # Zeros gradients on parameters stored in optimizer

            # Set model to training mode (important for Dropout, BatchNorm)
            model.train(True)

            # Forward pass
            predictions = model(X_batch_tensor)  # Use model.__call__

            # Compute loss and gradient w.r.t predictions
            loss_value, grad_data = cross_entropy_loss(predictions, y_batch)
            total_loss += loss_value

            # Backward pass - Start backprop from predictions using the calculated gradient
            predictions.backward(Tensor(grad_data))

            # Update weights
            optimizer.step()

        avg_loss = total_loss / num_batches if num_batches > 0 else total_loss
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")


def main():
    print("Creating Model")
    model = SimpleCNN()

    print("Creating Toy Dataset")
    X, y_one_hot = create_toy_dataset(num_samples=320, num_classes=10)

    print("Starting Training")
    train(model, X, y_one_hot, epochs=5, batch_size=32, lr=0.01)

    print("\nTraining Finished. Testing on single example")
    test_input = Tensor(np.random.randn(1, 1, 28, 28))
    model.eval()
    output = model(test_input)  # Use model.__call__

    print("Test output shape:", output.shape)
    print("Predicted class:", np.argmax(output.data))


if __name__ == "__main__":
    main()
