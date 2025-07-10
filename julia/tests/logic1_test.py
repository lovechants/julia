import numpy as np
import pytest

from julia.core.tensor import Tensor
from julia.core.nn.layers import Linear, Sequential
from julia.core.loss import mse_loss
from julia.core.optim import SGD

def test_tensor_autograd_square():
    x = Tensor(np.array([3.0]), requires_grad=True)
    y = x * x
    y.backward()
    assert np.allclose(x.grad.data, 6.0)

def test_numerical_gradient():
    x = Tensor(np.array([2.0]), requires_grad=True)
    y = x * x
    y.backward()
    autograd_grad = x.grad.data.copy()
    eps = 1e-5
    x_val = 2.0
    f_plus = (x_val + eps) ** 2
    f_minus = (x_val - eps) ** 2
    num_grad = (f_plus - f_minus) / (2 * eps)
    assert np.allclose(autograd_grad, num_grad, atol=1e-4)

def test_mse_loss_backward():
    y_pred = Tensor(np.array([2.0, 3.0]), requires_grad=True)
    y_true = Tensor(np.array([1.0, 2.0]), requires_grad=False)
    loss = mse_loss(y_pred, y_true)
    loss.backward()
    # d/dy_pred = 2*(y_pred - y_true)/N
    expected_grad = 2 * (np.array([2.0, 3.0]) - np.array([1.0, 2.0])) / 2
    assert np.allclose(y_pred.grad.data, expected_grad)

def test_linear_regression_learns():
    np.random.seed(0)
    X = np.random.randn(20, 1)
    y = 2 * X + 1 + 0.1 * np.random.randn(20, 1)
    X_tensor = Tensor(X, requires_grad=False)
    y_tensor = Tensor(y, requires_grad=False)
    model = Linear(1, 1)
    optimizer = SGD(model.parameters(), lr=0.1)
    for epoch in range(100):
        optimizer.zero_grad()
        y_pred = model(X_tensor)
        loss = mse_loss(y_pred, y_tensor)
        loss.backward()
        optimizer.step()
    # After training, weight should be close to 2, bias close to 1
    assert np.allclose(model.weight.data, 2, atol=0.2)
    assert np.allclose(model.bias.data, 1, atol=0.2)

def test_sequential_backward():
    X = np.array([[1.0, 2.0]])
    y = np.array([[1.0]])
    X_tensor = Tensor(X, requires_grad=False)
    y_tensor = Tensor(y, requires_grad=False)
    model = Sequential(
        Linear(2, 4),
        lambda x: x.relu(),
        Linear(4, 1)
    )
    y_pred = model(X_tensor)
    loss = mse_loss(y_pred, y_tensor)
    loss.backward()
    grads = [p.grad.data for p in model.parameters()]
    assert all(g is not None and np.any(np.abs(g) > 0) for g in grads)
