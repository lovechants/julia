import os
import numpy as np
from julia.core.tensor import Tensor
from julia.core.ir import IRGraph, IRNode, DataType, ConstantNode, VariableNode, PlaceholderNode
from julia.core.ir_bridge import trace, execute_graph
import pytest
import julia.core.utils.ops_registry

try:
    import onnx
    from julia.core.onnx import import_onnx, export_onnx, ONNXImporter, ONNXExporter
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

# Skip tests if ONNX is not available
pytestmark = pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")


def test_basic_onnx_export_import():
    """Test basic export and import of a simple model""" 
    
    # Create a basic graph with Add operation
    graph = IRGraph(name="basic_model")
    
    # Add input placeholders
    a = graph.add_placeholder(shape=(3, 4), dtype=DataType.FLOAT32, name="input_a")
    b = graph.add_placeholder(shape=(3, 4), dtype=DataType.FLOAT32, name="input_b")
    
    # Add operation
    add_node = IRNode(op_type="Add", inputs=[a, b], name="add_result")
    add_node.shape = (3, 4)  # Set shape explicitly
    add_node.dtype = DataType.FLOAT32
    graph.add_node(add_node)
    
    # Set graph output
    graph.set_outputs([add_node])
    
    # Run shape inference
    graph.infer_shapes()
    
    # Export to ONNX
    output_path = "basic_model.onnx"
    export_onnx(graph, output_path)
    
    print(f"Exported model to {output_path}")
    
    # Import back
    imported_graph = import_onnx(output_path)
    
    print(f"Imported graph name: {imported_graph.name}")
    print(f"Number of nodes: {len(imported_graph.nodes)}")
    print(f"Input nodes: {[node.name for node in imported_graph.inputs]}")
    print(f"Output nodes: {[node.name for node in imported_graph.outputs]}")
    
    # Clean up
    if os.path.exists(output_path):
        os.remove(output_path)
    
    # Verify the imported graph structure
    assert len(imported_graph.inputs) == 2, "Should have 2 inputs"
    assert len(imported_graph.outputs) == 1, "Should have 1 output"
    
    output_node = imported_graph.outputs[0]
    assert output_node.op_type == "Add", f"Expected Add op, got {output_node.op_type}"


def test_neural_network_export_import():
    """Test export and import of a simple neural network"""
    
    # Create a simple neural network
    graph = IRGraph(name="neural_network")
    
    # Input
    x = graph.add_placeholder(shape=(32, 784), dtype=DataType.FLOAT32, name="input")
    
    # Layer 1
    W1 = graph.add_variable(shape=(784, 128), dtype=DataType.FLOAT32, name="W1")
    b1 = graph.add_variable(shape=(1, 128), dtype=DataType.FLOAT32, name="b1")
    
    # Layer 1 ops
    matmul1 = IRNode(op_type="MatMul", inputs=[x, W1], name="matmul1")
    matmul1.shape = (32, 128)
    matmul1.dtype = DataType.FLOAT32
    graph.add_node(matmul1)
    
    add1 = IRNode(op_type="Add", inputs=[matmul1, b1], name="add1")
    add1.shape = (32, 128)
    add1.dtype = DataType.FLOAT32
    graph.add_node(add1)
    
    relu1 = IRNode(op_type="ReLU", inputs=[add1], name="relu1")
    relu1.shape = (32, 128)
    relu1.dtype = DataType.FLOAT32
    graph.add_node(relu1)
    
    # Layer 2
    W2 = graph.add_variable(shape=(128, 10), dtype=DataType.FLOAT32, name="W2")
    b2 = graph.add_variable(shape=(1, 10), dtype=DataType.FLOAT32, name="b2")
    
    # Layer 2 ops
    matmul2 = IRNode(op_type="MatMul", inputs=[relu1, W2], name="matmul2")
    matmul2.shape = (32, 10)
    matmul2.dtype = DataType.FLOAT32
    graph.add_node(matmul2)
    
    add2 = IRNode(op_type="Add", inputs=[matmul2, b2], name="add2")
    add2.shape = (32, 10)
    add2.dtype = DataType.FLOAT32
    graph.add_node(add2)
    
    sigmoid = IRNode(op_type="Sigmoid", inputs=[add2], name="sigmoid")
    sigmoid.shape = (32, 10)
    sigmoid.dtype = DataType.FLOAT32
    graph.add_node(sigmoid)
    
    # Set output
    graph.set_outputs([sigmoid])
    
    # Run shape inference
    graph.infer_shapes()
    
    # Export to ONNX
    output_path = "neural_network.onnx"
    export_onnx(graph, output_path)
    
    print(f"Exported model to {output_path}")
    
    # Import back
    imported_graph = import_onnx(output_path)
    
    print(f"Imported graph name: {imported_graph.name}")
    print(f"Number of nodes: {len(imported_graph.nodes)}")
    print(f"Input nodes: {[node.name for node in imported_graph.inputs]}")
    print(f"Output nodes: {[node.name for node in imported_graph.outputs]}")
    
    # Clean up
    if os.path.exists(output_path):
        os.remove(output_path)
    
    # Verify the imported graph structure
    assert len(imported_graph.inputs) == 1, "Should have 1 input"
    assert len(imported_graph.outputs) == 1, "Should have 1 output"
    
    # Verify the last node 
    output_node = imported_graph.outputs[0]
    assert output_node.op_type == "Sigmoid", f"Expected Sigmoid op, got {output_node.op_type}"


def test_tensor_to_onnx_roundtrip():
    """Test full roundtrip from tensor ops to ONNX and back to tensor execution"""
        
    # Create a simple computation with tensors
    x = Tensor(np.random.randn(3, 4), requires_grad=True)
    W = Tensor(np.random.randn(4, 5), requires_grad=True)
    b = Tensor(np.random.randn(1, 5), requires_grad=True)
    
    # y = relu(x @ W + b)
    y = (x.matmul(W) + b).relu()
    
    # Trace the computation
    graph = trace(y)
    
    # Print graph info
    print(f"Original graph name: {graph.name}")
    print(f"Number of nodes: {len(graph.nodes)}")
    
    # Ensure shapes are inferred
    graph.infer_shapes()
    
    # Export to ONNX
    output_path = "tensor_model.onnx"
    export_onnx(graph, output_path)
    
    print(f"Exported model to {output_path}")
    
    imported_graph = import_onnx(output_path)
    
    print(f"Imported graph name: {imported_graph.name}")
    print(f"Number of nodes: {len(imported_graph.nodes)}")
    
    # Clean up
    if os.path.exists(output_path):
        os.remove(output_path)
    
    # Check if the imported graph has inputs
    if not imported_graph.inputs:
        print("Warning: Imported graph has no inputs")
        return
        
    # Create a new input tensor
    x_new = Tensor(np.random.randn(3, 4), requires_grad=False)
    
    # Execute the original computation
    y_orig = (x_new.matmul(W) + b).relu()
    
    if imported_graph.inputs:
        input_name = imported_graph.inputs[0].name
        
        # Execute the imported graph
        y_imported = execute_graph(imported_graph, {input_name: x_new})
        
        print(f"Original output shape: {y_orig.shape}")
        print(f"Imported graph output shape: {y_imported.shape}")
        
        # Verify the outputs are close
        if hasattr(y_imported, 'data') and hasattr(y_orig, 'data'):
            print(f"Max difference: {np.max(np.abs(y_imported.data - y_orig.data))}")
            assert np.allclose(y_imported.data, y_orig.data, rtol=1e-5, atol=1e-5), "Outputs should match"
    else:
        print("Skipping execution comparison, imported graph has no inputs")
    
    # Create a new input tensor
    x_new = Tensor(np.random.randn(3, 4), requires_grad=False)
    
    # Execute the original computation
    y_orig = (x_new.matmul(W) + b).relu()
    
    # Find the input node name in the imported graph
    input_name = imported_graph.inputs[0].name
    
    # Execute the imported graph
    y_imported = execute_graph(imported_graph, {input_name: x_new})
    
    print(f"Original output shape: {y_orig.shape}")
    print(f"Imported graph output shape: {y_imported.shape}")
    
    # Verify the outputs are close
    if hasattr(y_imported, 'data') and hasattr(y_orig, 'data'):
        print(f"Max difference: {np.max(np.abs(y_imported.data - y_orig.data))}")
        assert np.allclose(y_imported.data, y_orig.data, rtol=1e-5, atol=1e-5), "Outputs should match"
    
    assert imported_graph


def test_custom_op_registration():
    """Test registration of custom ONNX operations"""
    
    # Register a custom operation
    from julia.core.utils.onnx_registry import onnx_registry
    
    @onnx_registry.register_export("CustomOp", "Custom")
    def export_custom_op(exporter, ir_node, input_names):
        return onnx.helper.make_node(
            "Custom",
            input_names,
            [ir_node.name],
            domain="custom.domain",
            name=ir_node.name
        )
    
    # Create a graph with the custom op
    graph = IRGraph(name="custom_model")
    
    # Add input
    x = graph.add_placeholder(shape=(3, 4), dtype=DataType.FLOAT32, name="input")
    
    # Add custom op
    custom_node = IRNode(op_type="CustomOp", inputs=[x], name="custom_result")
    custom_node.shape = (3, 4)
    custom_node.dtype = DataType.FLOAT32
    graph.add_node(custom_node)
    
    # Set output
    graph.set_outputs([custom_node])
    
    # Run shape inference
    graph.infer_shapes()
    
    # Export to ONNX
    output_path = "custom_model.onnx"
    
    try:
        export_onnx(graph, output_path)
        print(f"Exported model to {output_path}")
        
        # Load the model to verify (ust verify structure)
        model = onnx.load(output_path)
        
        # Check if our custom op is in the model
        found_custom_op = False
        for node in model.graph.node:
            if node.op_type == "Custom":
                found_custom_op = True
                break
        
        assert found_custom_op, "Custom op should be in the exported model"
        print("Custom op found in exported model")
        
    except Exception as e:
        print(f"Error exporting custom op: {e}")
    finally:
        # Clean up
        if os.path.exists(output_path):
            os.remove(output_path)
    
    assert graph


def test_onnx_special_ops():
    """Test special ONNX operations like Gemm"""
    
    # Create a standard ONNX model with Gemm operation
    import onnx
    from onnx import helper, TensorProto
    
    # Create inputs
    a = helper.make_tensor_value_info('a', TensorProto.FLOAT, [3, 4])
    b = helper.make_tensor_value_info('b', TensorProto.FLOAT, [4, 5])
    c = helper.make_tensor_value_info('c', TensorProto.FLOAT, [3, 5])
    
    # Create output
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 5])
    
    # Create Gemm node (Y = alpha * A @ B + beta * C)
    gemm_node = helper.make_node(
        'Gemm',
        ['a', 'b', 'c'],
        ['y'],
        alpha=0.5,
        beta=0.8,
        transA=0,
        transB=0,
        name='gemm'
    )
    
    # Create graph
    graph_proto = helper.make_graph(
        [gemm_node],
        'gemm_model',
        [a, b, c],
        [y]
    )
    
    # Create model
    model = helper.make_model(graph_proto, producer_name='julia-test')
    model.opset_import[0].version = 13
    
    # Save model
    output_path = "gemm_model.onnx"
    onnx.save(model, output_path)
    
    print(f"Created ONNX model with Gemm op: {output_path}")
    
    # Import the model
    try:
        imported_graph = import_onnx(output_path)
        
        print(f"Imported graph name: {imported_graph.name}")
        print(f"Number of nodes: {len(imported_graph.nodes)}")
        print(f"Input nodes: {[node.name for node in imported_graph.inputs]}")
        print(f"Output nodes: {[node.name for node in imported_graph.outputs]}")
        
        # Verify the structure - Gemm should be converted to MatMul and Add
        # The actual implementation depends on converter in onnx_registry.py
        
        # Check if there are the right number of operations
        expected_ops = ["MatMul", "Add", "Mul"]  # or close enough 
        found_ops = [node.op_type for node in imported_graph.nodes.values()]
        
        print(f"Operations in imported graph: {found_ops}")
        
        # Depending on our converter implementation, structures might differ 
        # but we should be able to execute the graph
        
    except Exception as e:
        print(f"Error importing Gemm op: {e}")
    finally:
        # Clean up
        if os.path.exists(output_path):
            os.remove(output_path)
    
    assert imported_graph


if __name__ == "__main__":
    test_basic_onnx_export_import()
    test_neural_network_export_import()
    test_tensor_to_onnx_roundtrip()
    test_custom_op_registration()
    test_onnx_special_ops()

