"""
Register all operations with the operation registry
"""
from julia.core.ops import Add, Sub, Mul, Div, MatMul, ReLU, Sigmoid, Reshape
from julia.core.utils.op_registry import registry

registry.register("Add", Add)
registry.register("Sub", Sub)
registry.register("Mul", Mul)
registry.register("Div", Div)
registry.register("MatMul", MatMul)
registry.register("ReLU", ReLU)
registry.register("Sigmoid", Sigmoid)
registry.register("Reshape", Reshape)
