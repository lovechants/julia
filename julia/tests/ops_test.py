from ..core.tensor import Tensor
from ..core.optim import SGD
import numpy as np


def test_add_mul():
    x = Tensor([2.0], requires_grad=True)
    y = Tensor([3.0], requires_grad=True)
    z = x * y + y
    z.backward()
    print("z:", z.data)
    print("dz/dx (should be y):", x.grad.data)
    print("dz/dy (should be x + 1):", y.grad.data)


def test_linear_regression():
    # y = 2x + 3
    X = Tensor(np.linspace(0, 1, 100).reshape(-1, 1))
    true_w = 2.0
    true_b = 3.0
    y = Tensor(true_w * X.data + true_b)

    # Initialize weights
    W = Tensor(np.random.randn(1, 1), requires_grad=True)
    b = Tensor(np.zeros((1,)), requires_grad=True)

    optimizer = SGD([W, b], lr=0.1)

    for epoch in range(200):
        optimizer.zero_grad()
        y_pred = X.matmul(W) + b
        loss = ((y_pred - y) * (y_pred - y)).reshape(-1).data.mean()
        loss_tensor = Tensor(loss, requires_grad=True)
        loss_tensor.backward()
        optimizer.step()

    print("Learned W:", W.data.flatten())
    print("Learned b:", b.data.flatten())


def test_sigmoid():
    x = Tensor([-1.0, 0.0, 1.0], requires_grad=True)
    s = x.sigmoid()
    s.backward(Tensor(np.ones_like(s.data)))
    print("Sigmoid:", s.data)
    print("Sigmoid grad:", x.grad.data)


if __name__ == "__main__":
    test_add_mul()
    test_linear_regression()
    test_sigmoid()
