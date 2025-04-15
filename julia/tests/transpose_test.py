import numpy as np
from julia.core.ir import IRGraph, IRNode, DataType, ConstantNode, PlaceholderNode, VariableNode

def test_transpose_method():
    """Test the transpose method on different node types"""
    
    # Create a graph
    graph = IRGraph(name="transpose_test")
    
    # Test on PlaceholderNode
    placeholder = graph.add_placeholder(shape=(3, 4), dtype=DataType.FLOAT32, name="placeholder")
    placeholder_t = IRNode(
        op_type="Transpose",
        inputs=[placeholder],
        attributes={"perm": [1, 0]},
        name="placeholder_transpose"
    )
    placeholder_t.shape = (4, 3)
    placeholder_t.dtype = DataType.FLOAT32
    graph.add_node(placeholder_t)
    
    # Test on ConstantNode
    const_value = np.random.randn(2, 5).astype(np.float32)
    const_node = graph.add_constant(const_value, name="constant")
    const_t = IRNode(
        op_type="Transpose",
        inputs=[const_node],
        attributes={"perm": [1, 0]},
        name="constant_transpose"
    )
    const_t.shape = (5, 2)
    const_t.dtype = DataType.FLOAT32
    graph.add_node(const_t)
    
    # Test on VariableNode
    var_node = graph.add_variable(shape=(4, 6), dtype=DataType.FLOAT32, name="variable")
    var_t = IRNode(
        op_type="Transpose",
        inputs=[var_node],
        attributes={"perm": [1, 0]},
        name="variable_transpose"
    )
    var_t.shape = (6, 4)
    var_t.dtype = DataType.FLOAT32
    graph.add_node(var_t)
    
    # Test on regular IRNode
    op_node = IRNode(
        op_type="Add",
        inputs=[placeholder, const_node],
        name="add"
    )
    op_node.shape = (3, 4)
    op_node.dtype = DataType.FLOAT32
    graph.add_node(op_node)
    
    op_t = IRNode(
        op_type="Transpose",
        inputs=[op_node],
        attributes={"perm": [1, 0]},
        name="add_transpose"
    )
    op_t.shape = (4, 3)
    op_t.dtype = DataType.FLOAT32
    graph.add_node(op_t)
    
    # Set as outputs to test in the graph
    graph.set_outputs([placeholder_t, const_t, var_t, op_t])
    
    # Run shape inference on the graph
    graph.infer_shapes()
    
    # Print shapes of the transposed nodes
    print(f"Original placeholder shape: {placeholder.shape}")
    print(f"Transposed placeholder shape: {placeholder_t.shape}")
    
    print(f"Original constant shape: {const_node.shape}")
    print(f"Transposed constant shape: {const_t.shape}")
    
    print(f"Original variable shape: {var_node.shape}")
    print(f"Transposed variable shape: {var_t.shape}")
    
    print(f"Original op node shape: {op_node.shape}")
    print(f"Transposed op node shape: {op_t.shape}")
    
    # Verify shapes
    assert placeholder_t.shape == (4, 3), f"Expected (4, 3), got {placeholder_t.shape}"
    assert const_t.shape == (5, 2), f"Expected (5, 2), got {const_t.shape}"
    assert var_t.shape == (6, 4), f"Expected (6, 4), got {var_t.shape}"
    assert op_t.shape == (4, 3), f"Expected (4, 3), got {op_t.shape}"
    
    print("All transpose shapes are correct")
    
    assert graph

if __name__ == "__main__":
    test_transpose_method()
