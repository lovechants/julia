import numpy as np
from julia.core.ir import IRGraph, IRNode, DataType, ConstantNode, PlaceholderNode, VariableNode
import julia.core.utils.ops_registry  # Import to register operations

def test_matrix_vector_matmul():
    """Test specifically for matrix-vector multiplication shape inference"""
    
    # Create a minimal graph
    graph = IRGraph(name="matmul_mv_test")
    
    # Create the inputs with explicit shapes
    matrix = PlaceholderNode(shape=(5, 3), dtype=DataType.FLOAT32, name="matrix")
    vector = PlaceholderNode(shape=(3,), dtype=DataType.FLOAT32, name="vector")  # Note: using 3, not 5 -> refer to infer_shapes
    
    # Add them to graph
    graph.nodes[matrix.id] = matrix
    graph.nodes[vector.id] = vector
    graph.inputs = [matrix, vector]
    
    # Create matrix-vector multiplication
    mv_node = IRNode(op_type="MatMul", inputs=[matrix, vector], name="mv")
    graph.add_node(mv_node)
    
    # Set output
    graph.set_outputs([mv_node])
    
    print(f"Matrix shape: {matrix.shape}")
    print(f"Vector shape: {vector.shape}")
    
    graph.infer_shapes()
    
    print(f"MatMul result shape: {mv_node.shape}")
    
    # Verify
    assert mv_node.shape == (5,), f"Expected (5,), got {mv_node.shape}"
    
    print("Matrix-Vector shape inference passed")
    
    assert graph

if __name__ == "__main__":
    test_matrix_vector_matmul()
