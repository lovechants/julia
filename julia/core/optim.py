import numpy as np 

"""
Optimizers 
"""

class SGD:
    def __init__(self, params, lr=0.01):
        self.params = params
        self.lr = lr 

    def step(self):
        for param in self.params:
            if param.grad is not None:
                param.data -= self.lr * param.grad.data 

    def zero_grad(self):
        for param in self.params:
            param.zero_grad()
