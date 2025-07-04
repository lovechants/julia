"""
Register all operations with the operation registry
"""
from julia.core.ops import Add, Sub, Mul, Div, MatMul, ReLU, Sigmoid, Reshape
from julia.core.ops import (
    LeakyReLU,
    PReLU,
    ELU,
    Softmax,
    Tanh,
    LogSoftmax,
    GELU,
    Swish,
    SELU,
)
from julia.core.utils.op_registry import registry

registry.register("Add", Add)
registry.register("Sub", Sub)
registry.register("Mul", Mul)
registry.register("Div", Div)
registry.register("MatMul", MatMul)
registry.register("ReLU", ReLU)
registry.register("Sigmoid", Sigmoid)
registry.register("Reshape", Reshape)
registry.register("LeakyReLU", LeakyReLU)
registry.register("PReLU", PReLU)
registry.register("ELU", ELU)
registry.register("SELU", SELU)
registry.register("Tanh", Tanh)
registry.register("Softmax", Softmax)
registry.register("LogSoftmax", LogSoftmax)
registry.register("GELU", GELU)
registry.register("Swish", Swish)


# Softmax and LogSoftmax might change dimensions based on the dim parameter
@registry.register_shape_inference("Softmax")
def infer_softmax_shape(node, input_shapes):
    """Shape inference for Softmax: typically preserves input shape"""
    if len(input_shapes) < 1:
        return None
    return input_shapes[0]


registry.register_shape_inference("LogSoftmax")(infer_softmax_shape)
