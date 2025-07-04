import pytest
import numpy as np
import math

from julia.core.tensor import Tensor, Function, _ensure_tensor
from julia.core.ops import Add, MatMul  # Import Ops
from julia.core.optim import (
    SGD,
    Muon,
    Adam,
    AdamW,
    RMSProp,
    LAMB,
    AdaGrad,
)


class MSELoss(Function):
    @staticmethod
    def forward(ctx, y_pred, y_true):
        ctx.save_for_backwards(y_pred, y_true)
        y_true_tensor = _ensure_tensor(y_true)
        loss = np.mean((y_pred.data - y_true_tensor.data) ** 2)
        return Tensor(loss, requires_grad=y_pred.requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output is scalar gradient w.r.t. the final loss (usually 1.0)
        y_pred, y_true = ctx.saved_tensors
        y_true_tensor = _ensure_tensor(y_true)  # Ensure it's a tensor here too
        n = y_pred.data.size  # Or np.prod(y_pred.shape)
        # Gradient of MSE w.r.t y_pred
        grad_y_pred_data = 2.0 * (y_pred.data - y_true_tensor.data) / n
        # Apply incoming gradient (chain rule)
        grad_y_pred = Tensor(grad_output.data * grad_y_pred_data)
        # No gradient w.r.t y_true (it's considered constant target)
        return grad_y_pred, None


@pytest.fixture(scope="module")
def linear_regression_data():
    """Generates simple data for y = Wx + b + noise."""
    np.random.seed(42)  # for reproducibility
    W_true = np.array([[2.0], [-1.5]])
    b_true = np.array([3.0])
    X = np.random.rand(100, 2)  # 100 samples, 2 features
    noise = np.random.randn(100, 1) * 0.1  # Gaussian noise
    y = X @ W_true + b_true + noise
    return {
        "X": Tensor(X),
        "y": Tensor(y),  # Keep y as Tensor for consistency
        "W_true": W_true,
        "b_true": b_true,
    }


optimizer_configs = [
    pytest.param(SGD, {"lr": 0.1}, True, id="SGD_basic"),
    pytest.param(Muon, {"lr": 0.1, "momentum": 0.9}, True, id="Muon_basic"),
    pytest.param(Adam, {"lr": 0.01}, True, id="Adam_basic"),
    pytest.param(AdamW, {"lr": 0.01, "weight_decay": 0.01}, True, id="AdamW_basic"),
    pytest.param(RMSProp, {"lr": 0.01}, True, id="RMSProp_basic"),
    pytest.param(AdaGrad, {"lr": 0.1}, True, id="AdaGrad_basic"),
    pytest.param(LAMB, {"lr": 0.01, "weight_decay": 0.01}, True, id="LAMB_basic"),
    pytest.param(SGD, {"lr": 0.1, "momentum": 0.9}, True, id="SGD_momentum"),
    pytest.param(
        SGD, {"lr": 0.1, "momentum": 0.9, "nesterov": True}, True, id="SGD_nesterov"
    ),
    pytest.param(SGD, {"lr": 0.1, "weight_decay": 0.01}, True, id="SGD_weight_decay"),
    pytest.param(
        AdamW,
        {"lr": 0.01, "weight_decay": 0.0, "amsgrad": True},
        True,
        id="AdamW_amsgrad",
    ),
    pytest.param(RMSProp, {"lr": 0.01, "momentum": 0.9}, True, id="RMSProp_momentum"),
    pytest.param(RMSProp, {"lr": 0.01, "centered": True}, True, id="RMSProp_centered"),
    # Example of a case that might need tuning or could fail with default LR
    # pytest.param(AdaGrad, {'lr': 1.0}, False, id="AdaGrad_high_lr"), # Might diverge
]


@pytest.mark.parametrize(
    "optimizer_class, optimizer_kwargs, expected_to_converge", optimizer_configs
)
def test_optimizer_convergence(
    linear_regression_data, optimizer_class, optimizer_kwargs, expected_to_converge
):
    """
    Tests if an optimizer can fit a simple linear regression model.
    Verifies loss decrease, parameter change, and convergence.
    """
    X = linear_regression_data["X"]
    y_true = linear_regression_data["y"]
    W_true = linear_regression_data["W_true"]
    b_true = linear_regression_data["b_true"]

    # Initialize parameters - IMPORTANT: requires_grad=True
    W = Tensor(np.random.randn(2, 1) * 0.1, requires_grad=True)
    # Ensure bias is treated as a (1,) tensor for broadcasting/matmul compatibility
    b = Tensor(np.zeros((1,)), requires_grad=True)

    initial_W_data = W.data.copy()
    initial_b_data = b.data.copy()

    optimizer = optimizer_class([W, b], **optimizer_kwargs)

    n_epochs = 10000  # Number of training steps -> this works for all cases right now its quite slow some optimizers just need adjusted learning rate to work with a smaller range of epochs
    initial_loss = float("inf")
    final_loss = float("inf")

    print(
        f"\nTesting {optimizer_class.__name__} with {optimizer_kwargs}"
    )  # For clarity during verbose runs

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # Forward pass: y_pred = X @ W + b
        # Use the imported MatMul and Add operations
        y_pred = MatMul.apply(X, W)
        y_pred = Add.apply(
            y_pred, b
        )  # Broadcasting should handle adding (100,1) and (1,)

        # Calculate loss using the MSELoss Function
        loss = MSELoss.apply(y_pred, y_true)

        if epoch == 0:
            initial_loss = loss.data.item()
            # Sanity check gradients before first step
            assert W.grad is None, "W.grad should be None before backward()"
            assert b.grad is None, "b.grad should be None before backward()"

        # Backward pass
        loss.backward()  # Computes gradients for W and b

        # Sanity check gradients after backward() before step()
        assert W.grad is not None, f"W.grad is None after backward() in epoch {epoch}"
        assert b.grad is not None, f"b.grad is None after backward() in epoch {epoch}"
        assert np.all(np.isfinite(W.grad.data)), f"W.grad has NaN/Inf in epoch {epoch}"
        assert np.all(np.isfinite(b.grad.data)), f"b.grad has NaN/Inf in epoch {epoch}"

        # Optimizer step
        optimizer.step()  # Updates W and b based on gradients

        # Check parameters after step
        assert np.all(
            np.isfinite(W.data)
        ), f"W has NaN/Inf after step() in epoch {epoch}"
        assert np.all(
            np.isfinite(b.data)
        ), f"b has NaN/Inf after step() in epoch {epoch}"

        if (epoch + 1) % 50 == 0:
            print(
                f"Epoch {epoch + 1}, Loss: {loss.data.item():.6f}, W: {W.data.flatten()}, b: {b.data.item():.4f}"
            )

    final_loss = loss.data.item()
    final_W_data = W.data
    final_b_data = b.data

    print(f"Initial Loss: {initial_loss:.6f}, Final Loss: {final_loss:.6f}")
    print(f"Final Params: W: {final_W_data.flatten()}, b: {final_b_data.item():.4f}")
    print(f"True Params:  W: {W_true.flatten()}, b: {b_true.item():.4f}")

    # 1. Loss should decrease (robust check for non-increase)
    assert final_loss < initial_loss or math.isclose(
        final_loss, initial_loss, rel_tol=1e-5
    ), f"{optimizer_class.__name__}: Loss did not decrease significantly ({initial_loss=}, {final_loss=})"

    # 2. Parameters W should have changed
    assert not np.allclose(
        final_W_data, initial_W_data, atol=1e-8
    ), f"{optimizer_class.__name__}: Parameter W did not change from initial value {initial_W_data}"

    # 3. Parameter b should have changed (This was the original problem!)
    # Use item() for comparison if b is scalar-like, otherwise compare arrays
    if final_b_data.size == 1:
        assert not math.isclose(
            final_b_data.item(), initial_b_data.item(), abs_tol=1e-8
        ), f"{optimizer_class.__name__}: Parameter b did not change from initial value {initial_b_data.item()}"
    else:
        assert not np.allclose(
            final_b_data, initial_b_data, atol=1e-8
        ), f"{optimizer_class.__name__}: Parameter b did not change from initial value {initial_b_data}"

    # 4. Check for NaNs or Infs
    assert np.isfinite(final_loss), f"{optimizer_class.__name__}: Loss is NaN or Inf"
    assert np.all(
        np.isfinite(final_W_data)
    ), f"{optimizer_class.__name__}: Parameter W contains NaN or Inf"
    assert np.all(
        np.isfinite(final_b_data)
    ), f"{optimizer_class.__name__}: Parameter b contains NaN or Inf"

    # 5. Check for convergence
    if expected_to_converge:
        # May need adjustment based on epochs/LR
        assert np.allclose(
            final_W_data, W_true, rtol=0.1, atol=0.2
        ), f"{optimizer_class.__name__}: W did not converge to true value ({final_W_data=}, {W_true=})"
        assert np.allclose(
            final_b_data, b_true, rtol=0.1, atol=0.2
        ), f"{optimizer_class.__name__}: b did not converge to true value ({final_b_data=}, {b_true=})"
