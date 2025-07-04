import numpy as np
from ..core.ir import IRGraph, IRNode, DataType

# Import op_registry to ensure operations are registered


def test_shape_inference_basic():
    """Test basic shape inference for common operations"""

    # Create a graph with various operations
    graph = IRGraph(name="shape_test")

    # Add input placeholders
    a = graph.add_placeholder(shape=(3, 4), dtype=DataType.FLOAT32, name="a")
    b = graph.add_placeholder(shape=(3, 4), dtype=DataType.FLOAT32, name="b")
    c = graph.add_placeholder(shape=(4, 2), dtype=DataType.FLOAT32, name="c")

    # Add operations
    add_node = IRNode(op_type="Add", inputs=[a, b], name="add")
    graph.add_node(add_node)

    matmul_node = IRNode(op_type="MatMul", inputs=[add_node, c], name="matmul")
    graph.add_node(matmul_node)

    relu_node = IRNode(op_type="ReLU", inputs=[matmul_node], name="relu")
    graph.add_node(relu_node)

    reshape_node = IRNode(
        op_type="Reshape",
        inputs=[relu_node],
        attributes={"shape": (2, 3)},
        name="reshape",
    )
    graph.add_node(reshape_node)

    graph.set_outputs([reshape_node])

    # Infer shapes
    graph.infer_shapes()

    # Print the inferred shapes
    print("Input a shape:", a.shape)
    print("Input b shape:", b.shape)
    print("Input c shape:", c.shape)
    print("Add shape:", add_node.shape)
    print("MatMul shape:", matmul_node.shape)
    print("ReLU shape:", relu_node.shape)
    print("Reshape shape:", reshape_node.shape)

    # Check
    assert add_node.shape == (3, 4), f"Expected (3, 4), got {add_node.shape}"
    assert matmul_node.shape == (3, 2), f"Expected (3, 2), got {matmul_node.shape}"
    assert relu_node.shape == (3, 2), f"Expected (3, 2), got {relu_node.shape}"
    assert reshape_node.shape == (2, 3), f"Expected (2, 3), got {reshape_node.shape}"

    print("Basic shape inference correct")


def test_shape_inference_broadcasting():
    """Test shape inference with broadcasting"""

    # Create a graph with broadcasting operations
    graph = IRGraph(name="broadcast_test")

    # Add input placeholders with different shapes
    a = graph.add_placeholder(shape=(3, 4), dtype=DataType.FLOAT32, name="a")
    b = graph.add_placeholder(
        shape=(4,), dtype=DataType.FLOAT32, name="b"
    )  # Will broadcast
    c = graph.add_placeholder(
        shape=(1, 4), dtype=DataType.FLOAT32, name="c"
    )  # Will broadcast
    d = graph.add_constant(np.array(2.0, dtype=np.float32), name="d")  # Scalar

    # Add operations with broadcasting
    add_node = IRNode(op_type="Add", inputs=[a, b], name="add")
    graph.add_node(add_node)

    mul_node = IRNode(op_type="Mul", inputs=[add_node, c], name="mul")
    graph.add_node(mul_node)

    div_node = IRNode(op_type="Div", inputs=[mul_node, d], name="div")
    graph.add_node(div_node)

    graph.set_outputs([div_node])

    # Infer shapes
    graph.infer_shapes()

    print("Input a shape:", a.shape)
    print("Input b shape:", b.shape)
    print("Input c shape:", c.shape)
    print("Input d shape:", d.shape)
    print("Add shape:", add_node.shape)
    print("Mul shape:", mul_node.shape)
    print("Div shape:", div_node.shape)

    # Check correctness
    assert add_node.shape == (3, 4), f"Expected (3, 4), got {add_node.shape}"
    assert mul_node.shape == (3, 4), f"Expected (3, 4), got {mul_node.shape}"
    assert div_node.shape == (3, 4), f"Expected (3, 4), got {div_node.shape}"

    print("Broadcasting shape inference tests correct")


def test_shape_inference_matmul():  # TODO like actually fix this one or just test the operations more
    """Test shape inference for matrix multiplication with different dimensions"""

    # Create a graph with various matrix multiplications
    graph = IRGraph(name="matmul_test")

    # Add input placeholders with different shapes
    vector = graph.add_placeholder(shape=(3,), dtype=DataType.FLOAT32, name="vector")
    matrix = graph.add_placeholder(shape=(5, 3), dtype=DataType.FLOAT32, name="matrix")
    batch_matrix = graph.add_placeholder(
        shape=(2, 3, 4), dtype=DataType.FLOAT32, name="batch_matrix"
    )
    matrix2 = graph.add_placeholder(
        shape=(4, 6), dtype=DataType.FLOAT32, name="matrix2"
    )

    # Vector-vector dot product
    vv_node = IRNode(op_type="MatMul", inputs=[vector, vector], name="vv")
    graph.add_node(vv_node)

    # Matrix-vector multiplication
    mv_node = IRNode(op_type="MatMul", inputs=[matrix, vector], name="mv")
    graph.add_node(mv_node)

    # Vector-matrix multiplication
    vm_node = IRNode(op_type="MatMul", inputs=[vector, matrix], name="vm")
    graph.add_node(vm_node)

    # Create a transposed matrix
    matrix_t = matrix.transpose()
    graph.add_node(matrix_t)

    # Matrix-matrix multiplication
    mm_node = IRNode(op_type="MatMul", inputs=[matrix, matrix_t], name="mm")
    graph.add_node(mm_node)

    # Batched matrix multiplication
    bmm_node = IRNode(op_type="MatMul", inputs=[batch_matrix, matrix2], name="bmm")
    graph.add_node(bmm_node)

    graph.set_outputs([vv_node, mv_node, vm_node, mm_node, bmm_node])

    # Print input shapes before shape inference
    print("Before inference:")
    print("Matrix shape for mv_node input:", matrix.shape)
    print("Vector shape for mv_node input:", vector.shape)

    # Infer shapes
    graph.infer_shapes()

    # Print the inferred shapes
    print("Vector shape:", vector.shape)
    print("Matrix shape:", matrix.shape)
    print("Batch Matrix shape:", batch_matrix.shape)
    print("Matrix2 shape:", matrix2.shape)
    print("Vector-Vector shape:", vv_node.shape)
    print("Matrix-Vector shape:", mv_node.shape)
    print("Vector-Matrix shape:", vm_node.shape)
    print("Matrix-Matrix shape:", mm_node.shape)
    print("Batch Matrix-Matrix shape:", bmm_node.shape)

    # Check correctness
    assert vv_node.shape == (), f"Expected (), got {vv_node.shape}"  # Scalar result
    assert mv_node.shape == (5,), f"Expected (5,), got {mv_node.shape}"
    # assert vm_node.shape == (3,), f"Expected (3,), got {vm_node.shape}" # I think this assert is ??? -> it is
    assert mm_node.shape == (5, 5), f"Expected (5, 5), got {mm_node.shape}"
    assert bmm_node.shape == (2, 3, 6), f"Expected (2, 3, 6), got {bmm_node.shape}"

    print("MatMul shape inference passed")


def test_shape_inference_complex_graph():
    """Test shape inference on a complex graph like a neural network"""

    # Create a graph for a simple neural network
    graph = IRGraph(name="neural_network")
    print(graph)
    # Input placeholder
    x = graph.add_placeholder(shape=(32, 784), dtype=DataType.FLOAT32, name="input")

    # Layer 1: Dense + ReLU
    W1 = graph.add_variable(shape=(784, 128), dtype=DataType.FLOAT32, name="W1")
    b1 = graph.add_variable(shape=(1, 128), dtype=DataType.FLOAT32, name="b1")

    # Layer 1 operations
    matmul1 = IRNode(op_type="MatMul", inputs=[x, W1], name="matmul1")
    graph.add_node(matmul1)

    add1 = IRNode(op_type="Add", inputs=[matmul1, b1], name="add1")
    graph.add_node(add1)

    relu1 = IRNode(op_type="ReLU", inputs=[add1], name="relu1")
    graph.add_node(relu1)

    # Layer 2: Dense + Sigmoid
    W2 = graph.add_variable(shape=(128, 10), dtype=DataType.FLOAT32, name="W2")
    b2 = graph.add_variable(shape=(1, 10), dtype=DataType.FLOAT32, name="b2")

    # Layer 2 operations
    matmul2 = IRNode(op_type="MatMul", inputs=[relu1, W2], name="matmul2")
    graph.add_node(matmul2)

    add2 = IRNode(op_type="Add", inputs=[matmul2, b2], name="add2")
    graph.add_node(add2)

    sigmoid = IRNode(op_type="Sigmoid", inputs=[add2], name="sigmoid")
    graph.add_node(sigmoid)

    # Set as output
    graph.set_outputs([sigmoid])

    # Infer shapes
    graph.infer_shapes()

    print("Input shape:", x.shape)
    print("Layer 1:")
    print("  W1 shape:", W1.shape)
    print("  b1 shape:", b1.shape)
    print("  matmul1 shape:", matmul1.shape)
    print("  add1 shape:", add1.shape)
    print("  relu1 shape:", relu1.shape)
    print("Layer 2:")
    print("  W2 shape:", W2.shape)
    print("  b2 shape:", b2.shape)
    print("  matmul2 shape:", matmul2.shape)
    print("  add2 shape:", add2.shape)
    print("  sigmoid shape:", sigmoid.shape)

    # Check correctness
    assert matmul1.shape == (32, 128), f"Expected (32, 128), got {matmul1.shape}"
    assert add1.shape == (32, 128), f"Expected (32, 128), got {add1.shape}"
    assert relu1.shape == (32, 128), f"Expected (32, 128), got {relu1.shape}"
    assert matmul2.shape == (32, 10), f"Expected (32, 10), got {matmul2.shape}"
    assert add2.shape == (32, 10), f"Expected (32, 10), got {add2.shape}"
    assert sigmoid.shape == (32, 10), f"Expected (32, 10), got {sigmoid.shape}"

    print("Neural network shape inference passed")


def test_shape_inference_reshape():
    """Test shape inference for reshape operation with dimension inference"""

    # Create a graph with reshape operations
    graph = IRGraph(name="reshape_test")

    # Add input placeholder
    x = graph.add_placeholder(shape=(32, 784), dtype=DataType.FLOAT32, name="input")

    # Test basic reshape
    reshape1 = IRNode(
        op_type="Reshape",
        inputs=[x],
        attributes={"shape": (32, 28, 28)},
        name="reshape1",
    )
    graph.add_node(reshape1)

    # Test reshape with -1 (inferred dimension)
    reshape2 = IRNode(
        op_type="Reshape",
        inputs=[x],
        attributes={"shape": (-1, 28, 28)},
        name="reshape2",
    )
    graph.add_node(reshape2)

    # Test flatten (reshape to 2D)
    flatten = IRNode(
        op_type="Reshape",
        inputs=[reshape1],
        attributes={"shape": (32, -1)},
        name="flatten",
    )
    graph.add_node(flatten)

    graph.set_outputs([reshape1, reshape2, flatten])

    # Infer shapes
    graph.infer_shapes()

    # Print the inferred shapes
    print("Input shape:", x.shape)
    print("Reshape1 shape:", reshape1.shape)
    print("Reshape2 shape:", reshape2.shape)
    print("Flatten shape:", flatten.shape)

    # Check correctness
    assert reshape1.shape == (
        32,
        28,
        28,
    ), f"Expected (32, 28, 28), got {reshape1.shape}"
    assert reshape2.shape == (
        32,
        28,
        28,
    ), f"Expected (32, 28, 28), got {reshape2.shape}"
    assert flatten.shape == (32, 784), f"Expected (32, 784), got {flatten.shape}"

    print("Reshape shape inference tests passed")


if __name__ == "__main__":
    test_shape_inference_basic()
    test_shape_inference_broadcasting()
    test_shape_inference_matmul()
    test_shape_inference_complex_graph()
    test_shape_inference_reshape()
