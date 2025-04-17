import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
from julia.core.ir import IRGraph, IRNode, DataType, ConstantNode, VariableNode, PlaceholderNode
from julia.core.tensor import Tensor, Function

class TensorToIR:
    """
    Converts a computational graph built with Tensor operations to an IR graph
    """
    def __init__(self):
        self.tensor_to_node_map = {}  # Maps tensor.id to IRNode
        self.visited = set()  # Keep track of visited tensor IDs
        
    def convert(self, output_tensor: Tensor) -> IRGraph:
        """
        Convert a tensor and its computation graph to an IR graph
        
        Args:
            output_tensor: The output tensor of the computation
            
        Returns:
            IRGraph: The converted IR graph
        """
        graph = IRGraph(name="tensor_graph")
        
        # Clear maps for a new conversion
        self.tensor_to_node_map = {}
        self.visited = set()
        
        # Convert the tensor graph to IR graph recursively
        output_node = self._process_tensor(output_tensor, graph)
        
        # Set graph outputs
        graph.set_outputs([output_node])
        
        return graph
        
    def _process_tensor(self, tensor: Tensor, graph: IRGraph) -> IRNode:
        """Process a tensor and add it to the IR graph"""
        # If already processed
        if tensor.id in self.tensor_to_node_map:
            return self.tensor_to_node_map[tensor.id]
            
        # Mark as visited
        self.visited.add(tensor.id)
        
        # If tensor has no backward node, it's an input or constant
        if tensor._backward_node is None:
            # Create a node for the tensor
            if tensor.requires_grad:
                # It's a trainable variable (like weights)
                node = graph.add_variable(
                    shape=tensor.shape,
                    dtype=DataType.from_numpy(tensor.data.dtype),
                    name=f"var_{tensor.id[:8]}"
                )
                # Initialize with the tensor data
                node.attributes["value"] = tensor.data.copy()
            else:
                # It's a constant
                node = graph.add_constant(tensor.data.copy(), name=f"const_{tensor.id[:8]}")
                
            self.tensor_to_node_map[tensor.id] = node
            return node
            
        # For tensors with backward nodes, we need to process the operation
        backward_node = tensor._backward_node
        fn_cls = backward_node.fn_cls
        
        # Process inputs recursively if not already processed
        input_nodes = []
        for input_tensor in backward_node.inputs:
            if isinstance(input_tensor, Tensor):
                if input_tensor.id not in self.visited:
                    input_node = self._process_tensor(input_tensor, graph)
                else:
                    input_node = self.tensor_to_node_map[input_tensor.id]
                input_nodes.append(input_node)
            else:
                # Handle non-tensor inputs (like shapes for reshape)
                node = graph.add_constant(
                    np.array(input_tensor), 
                    name=f"const_{hash(str(input_tensor))}"
                )
                input_nodes.append(node)
        
        # Map operation name based on function class
        op_type = self._map_op_type(fn_cls)
        
        # Create node for this operation
        node = IRNode(
            op_type=op_type,
            inputs=input_nodes,
            name=f"{op_type}_{tensor.id[:8]}"
        )
        
        # Set shape and type
        node.shape = tensor.shape
        node.dtype = DataType.from_numpy(tensor.data.dtype)
        
        # Add node to graph
        graph.add_node(node)
        
        # Add to map
        self.tensor_to_node_map[tensor.id] = node
        
        return node
        
    def _map_op_type(self, fn_cls) -> str:
        """Map function class to operation type"""
        from julia.core.utils.op_registry import registry
        return registry.get_op_type(fn_cls)


class IRToTensor:
    """
    Converts an IR graph to a computation graph with Tensor operations
    """
    def __init__(self):
        self.node_to_tensor_map = {}  # Maps node.id to Tensor
        
    def convert(self, graph: IRGraph) -> List[Tensor]:
        """
        Convert an IR graph to tensor operations
        
        Args:
            graph: The IR graph to convert
            
        Returns:
            List[Tensor]: The output tensors
        """
        # Clear maps for new conversion
        self.node_to_tensor_map = {}
        
        # Process all nodes in topological order
        for node in graph.topological_sort():
            self._process_node(node)
            
        # Return output tensors
        output_tensors = [self.node_to_tensor_map[node.id] for node in graph.outputs]
        return output_tensors[0] if len(output_tensors) == 1 else output_tensors
        
    def _process_node(self, node: IRNode) -> Tensor: #TODO move this to registry implementation instead of hardcoding the operations (especially with new ops up)
        """Process an IR node and convert to tensor operations"""
        # If already processed
        if node.id in self.node_to_tensor_map:
            return self.node_to_tensor_map[node.id]
            
        # Process node based on type
        if node.op_type == "Constant":
            tensor = Tensor(node.attributes["value"], requires_grad=False)
            
        elif node.op_type == "Variable":
            # Get the initial value if available
            value = node.attributes.get("value", np.zeros(node.shape, dtype=node.dtype.to_numpy()))
            tensor = Tensor(value, requires_grad=node.attributes.get("trainable", True))
            
        elif node.op_type == "Placeholder":
            # For placeholders, create a tensor with zeros (will be replaced at runtime)
            tensor = Tensor(np.zeros(node.shape, dtype=node.dtype.to_numpy()), requires_grad=False)
            
        else:
            # For operations, first process input tensors
            input_tensors = [self._process_node(input_node) for input_node in node.inputs]
            
            # Get the function class from registry
            from julia.core.utils.op_registry import registry
            fn_cls = registry.get_function_class(node.op_type)
            
            if fn_cls is not None:
                # Use the registered operation class's apply method
                tensor = fn_cls.apply(*input_tensors)
            else:
                # Fall back to manual mapping for operations not in registry
                if node.op_type == "Add":
                    tensor = input_tensors[0] + input_tensors[1]
                elif node.op_type == "Sub":
                    tensor = input_tensors[0] - input_tensors[1]
                elif node.op_type == "Mul":
                    tensor = input_tensors[0] * input_tensors[1]
                elif node.op_type == "Div":
                    tensor = input_tensors[0] / input_tensors[1]
                elif node.op_type == "MatMul":
                    tensor = input_tensors[0].matmul(input_tensors[1])
                elif node.op_type == "ReLU":
                    tensor = input_tensors[0].relu()
                elif node.op_type == "Sigmoid":
                    tensor = input_tensors[0].sigmoid()
                elif node.op_type == "Reshape":
                    if len(input_tensors) > 1:
                        # If shape is a separate node
                        shape = input_tensors[1].data.astype(int).tolist()
                        tensor = input_tensors[0].reshape(shape)
                    else:
                        # If shape is an attribute
                        shape = node.attributes.get("shape", input_tensors[0].shape)
                        tensor = input_tensors[0].reshape(shape)
                else:
                    raise ValueError(f"Unsupported operation type: {node.op_type}")
        
        # Store in map
        self.node_to_tensor_map[node.id] = tensor
        
        return tensor


# Utility function to trace a tensor computation and create an IR graph
def trace(output_tensor: Tensor) -> IRGraph:
    """
    Trace a tensor computation and create an IR graph
    
    Args:
        output_tensor: The output tensor of the computation
        
    Returns:
        IRGraph: The traced IR graph
    """
    converter = TensorToIR()
    return converter.convert(output_tensor)


# Utility function to execute an IR graph with input tensors
def execute_graph(graph: IRGraph, input_dict: Dict[str, Tensor]) -> Union[Tensor, List[Tensor]]:
    """
    Execute an IR graph with the given input tensors
    
    Args:
        graph: The IR graph to execute
        input_dict: Dictionary mapping input node names to input tensors
        
    Returns:
        The output tensor(s)
    """
    # Create a modified graph with input placeholders replaced by constants
    modified_graph = IRGraph(name=graph.name)
    
    # Map from original node IDs to new nodes
    node_map = {}
    
    # Process all nodes
    for node_id, node in graph.nodes.items():
        if node.op_type == "Placeholder" and node.name in input_dict:
            # Replace placeholder with constant
            input_tensor = input_dict[node.name]
            new_node = modified_graph.add_constant(input_tensor.data, name=node.name)
            node_map[node_id] = new_node
        elif node.op_type == "Variable" or node.op_type == "Constant":
            # Copy variables and constants as is
            if node.op_type == "Variable":
                new_node = modified_graph.add_variable(
                    shape=node.shape,
                    dtype=node.dtype,
                    name=node.name,
                    trainable=node.attributes.get("trainable", True)
                )
                # Copy value if available
                if "value" in node.attributes:
                    new_node.attributes["value"] = node.attributes["value"]
            else:
                new_node = modified_graph.add_constant(
                    node.attributes["value"],
                    name=node.name
                )
            node_map[node_id] = new_node
        else:
            # For operations, first ensure all inputs are in the map
            inputs = [node_map[input_node.id] for input_node in node.inputs]
            
            # Create new node
            new_node = IRNode(
                op_type=node.op_type,
                inputs=inputs,
                attributes=node.attributes.copy(),
                name=node.name
            )
            new_node.shape = node.shape
            new_node.dtype = node.dtype
            
            modified_graph.add_node(new_node)
            node_map[node_id] = new_node
    
    # Set outputs
    modified_graph.outputs = [node_map[node.id] for node in graph.outputs]
    
    # Convert back to tensor
    converter = IRToTensor()
    return converter.convert(modified_graph)
