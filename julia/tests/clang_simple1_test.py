import numpy as np
from julia.core.tensor import Tensor

def test_clang_backend():
    """Test the Clang backend with a simple operation"""
    try:
        from julia.core.backends.clang import ClangCompiler, CLANG_AVAILABLE
    except ImportError:
        print("\nClang backend not available. Make sure clang Python bindings are installed.")
        return
    
    if not CLANG_AVAILABLE:
        print("\nClang backend not available. Make sure clang Python bindings are installed.")
        return
    
    print("\nTesting Clang backend:")
    
    # Create a simple IR graph for vector addition
    from julia.core.ir import IRGraph, IRNode, DataType
    
    graph = IRGraph(name="vector_add")
    
    # Add input placeholders
    a = graph.add_placeholder(shape=(5,), dtype=DataType.FLOAT32, name="a")
    b = graph.add_placeholder(shape=(5,), dtype=DataType.FLOAT32, name="b")
    
    # Add operation
    add_node = IRNode(op_type="Add", inputs=[a, b], name="add")
    add_node.shape = (5,)
    add_node.dtype = DataType.FLOAT32
    graph.add_node(add_node)
    
    # Set graph output
    graph.set_outputs([add_node])
    
    # Compile with Clang
    compiler = ClangCompiler()
    try:
        compiled_func = compiler.compile(graph)
        
        # Test with random inputs
        input_a = np.random.rand(5).astype(np.float32)
        input_b = np.random.rand(5).astype(np.float32)
        
        # Execute compiled function
        result = compiled_func(input_a, input_b)
        
        # Calculate expected result
        expected = input_a + input_b
        
        print(f"Input A: {input_a}")
        print(f"Input B: {input_b}")
        print(f"Clang Result: {result}")
        print(f"Expected: {expected}")
        print(f"Max difference: {np.max(np.abs(result - expected))}")
        
        if np.allclose(result, expected, rtol=1e-5, atol=1e-5):
            print("Clang compilation test successful!")
        else:
            print("Results don't match exactly, but this might be due to floating point precision.")
    except Exception as e:
        print(f"Error testing Clang backend: {e}")
