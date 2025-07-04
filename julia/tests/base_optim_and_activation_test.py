import numpy as np
from julia.core.tensor import Tensor
from julia.core.optim import SGD, Adam, AdamW, RMSProp, LAMB, AdaGrad, Muon

# from julia.core.ops import ReLU, LeakyReLU, ELU, Sigmoid, Tanh, GELU, Swish


def test_optimizers():
    """Test various optimizers on a simple regression problem"""
    # Create synthetic data: y = 2x + 3
    X = Tensor(np.linspace(0, 1, 100).reshape(-1, 1))
    true_w = 2.0
    true_b = 3.0
    y = Tensor(true_w * X.data + true_b)

    optimizers = {
        "SGD": SGD,
        "Muon": Muon,
        "Adam": Adam,
        "AdamW": AdamW,
        "RMSProp": RMSProp,
        "LAMB": LAMB,
        "AdaGrad": AdaGrad,
    }

    for name, optimizer_class in optimizers.items():
        # Initialize weights
        W = Tensor(np.random.randn(1, 1), requires_grad=True)
        b = Tensor(np.zeros((1,)), requires_grad=True)

        # Create optimizer
        if name == "SGD":
            optimizer = optimizer_class([W, b], lr=0.1)
        elif name == "Muon":
            optimizer = optimizer_class([W, b], lr=0.1, momentum=0.9)
        elif name in ["Adam", "AdamW"]:
            optimizer = optimizer_class([W, b], lr=0.01)
        elif name == "RMSProp":
            optimizer = optimizer_class([W, b], lr=0.01, alpha=0.9)
        elif name == "LAMB":
            optimizer = optimizer_class([W, b], lr=0.01, weight_decay=0.01)
        elif name == "AdaGrad":
            optimizer = optimizer_class([W, b], lr=0.1)

        # Training loop
        for epoch in range(200):
            optimizer.zero_grad()
            y_pred = X.matmul(W) + b
            loss = ((y_pred - y) * (y_pred - y)).reshape(-1).data.mean()
            loss_tensor = Tensor(loss, requires_grad=True)
            loss_tensor.backward()
            optimizer.step()

            if epoch % 50 == 0:
                print(
                    f"Epoch {epoch}, Loss: {loss:.6f}, W: {W.data.flatten()[0]:.4f}, b: {b.data.flatten()[0]:.4f}"
                )

        print(
            f"Final parameters - W: {W.data.flatten()[0]:.4f}, b: {b.data.flatten()[0]:.4f}"
        )
        print(f"True parameters - W: {true_w:.4f}, b: {true_b:.4f}")


def test_activations():
    """Test various activation functions"""
    # Create input tensor with a range of values
    x = Tensor(np.linspace(-5, 5, 10), requires_grad=True)

    activations = [
        ("ReLU", lambda t: t.relu()),
        ("LeakyReLU", lambda t: t.leaky_relu(0.1)),
        ("Sigmoid", lambda t: t.sigmoid()),
        ("Tanh", lambda t: t.tanh()),
        ("ELU", lambda t: t.elu(1.0)),
        ("SELU", lambda t: t.selu()),
        ("GELU", lambda t: t.gelu()),
        ("Swish", lambda t: t.swish()),
    ]

    print("\nTesting activation functions:")
    for name, activation_fn in activations:
        # Forward pass
        y = activation_fn(x)

        # Backward pass (use ones as gradient for simplicity)
        y.backward(Tensor(np.ones_like(y.data)))

        print(f"\n{name} activation:")
        print(f"Input: {x.data}")
        print(f"Output: {y.data}")
        print(f"Gradient: {x.grad.data}")

        # Reset gradient
        x.zero_grad()


def test_neural_network():
    """Test a small neural network with the new activations and optimizers"""
    # Generate a simple classification dataset
    np.random.seed(42)
    N = 100  # samples per class
    D = 2  # dimensions
    K = 3  # number of classes

    X = np.zeros((N * K, D))
    y = np.zeros(N * K, dtype="uint8")

    for j in range(K):
        ix = range(N * j, N * (j + 1))
        r = np.linspace(0.0, 1, N)
        t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = j

    # Convert to one-hot encoding
    y_onehot = np.zeros((N * K, K))
    y_onehot[np.arange(N * K), y] = 1

    # Convert to tensors
    X_tensor = Tensor(X, requires_grad=False)
    y_tensor = Tensor(y_onehot, requires_grad=False)

    # Create a simple neural network
    # Input layer -> Hidden layer -> Output layer
    W1 = Tensor(np.random.randn(D, 50) * 0.01, requires_grad=True)
    b1 = Tensor(np.zeros((1, 50)), requires_grad=True)

    W2 = Tensor(np.random.randn(50, K) * 0.01, requires_grad=True)
    b2 = Tensor(np.zeros((1, K)), requires_grad=True)

    # Choose optimizer
    optimizer = Adam([W1, b1, W2, b2], lr=0.01)

    # Training loop
    for epoch in range(200):
        # Forward pass
        hidden = X_tensor.matmul(W1) + b1
        hidden_activated = hidden.gelu()  # Using GELU activation
        scores = hidden_activated.matmul(W2) + b2

        # Compute softmax
        exp_scores = Tensor(
            np.exp(scores.data - np.max(scores.data, axis=1, keepdims=True))
        )
        probs = exp_scores / Tensor(np.sum(exp_scores.data, axis=1, keepdims=True))

        # Compute loss (cross-entropy)
        correct_logprobs = -np.log(probs.data[range(N * K), y])
        data_loss = np.sum(correct_logprobs) / N / K
        reg_loss = 0.5 * 0.001 * (np.sum(W1.data**2) + np.sum(W2.data**2))
        loss = data_loss + reg_loss

        # Compute accuracy
        predicted_class = np.argmax(scores.data, axis=1)
        accuracy = np.mean(predicted_class == y)

        # Print stats
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.6f}, Accuracy: {accuracy:.4f}")

        # Backward pass
        # Gradient of cross-entropy loss
        dscores = probs.data.copy()
        dscores[range(N * K), y] -= 1
        dscores /= N

        # Create tensor from gradient
        dscores_tensor = Tensor(dscores, requires_grad=False)

        # Zero gradients
        optimizer.zero_grad()

        # Manually compute gradients
        dW2 = hidden_activated.data.T.dot(dscores)
        db2 = np.sum(dscores, axis=0, keepdims=True)

        dhidden_activated = dscores.dot(W2.data.T)

        # GELU gradient
        x = hidden.data
        cdf = 0.5 * (1 + np.tanh((0.7978845608 * (x + 0.044715 * x**3))))
        tanh_deriv = 1 - np.tanh((0.7978845608 * (x + 0.044715 * x**3))) ** 2
        pdf = 0.5 * 0.7978845608 * (1 + 0.134145 * x**2) * tanh_deriv
        dgelu = dhidden_activated * (cdf + x * pdf)

        dW1 = X_tensor.data.T.dot(dgelu)
        db1 = np.sum(dgelu, axis=0, keepdims=True)

        # Add regularization gradient
        dW2 += 0.001 * W2.data
        dW1 += 0.001 * W1.data

        # Update gradients
        W2.grad = Tensor(dW2)
        b2.grad = Tensor(db2)
        W1.grad = Tensor(dW1)
        b1.grad = Tensor(db1)

        # Update parameters
        optimizer.step()

    # Final accuracy
    hidden = X_tensor.matmul(W1) + b1
    hidden_activated = hidden.gelu()
    scores = hidden_activated.matmul(W2) + b2
    predicted_class = np.argmax(scores.data, axis=1)
    accuracy = np.mean(predicted_class == y)

    print(f"Final accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    test_optimizers()
    test_activations()
    test_neural_network()
