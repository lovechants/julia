import numpy as np
from typing import Any

try:
    import onnx
    from onnx import numpy_helper
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: onnx not found. ONNX import/export will not be available.")

from julia.core.ir import IRGraph, IRNode, DataType, ConstantNode

class ONNXImporter:
    """
    Import an ONNX model into IR graph
    """
    def __init__(self):
        if not ONNX_AVAILABLE:
            raise ImportError("onnx package is required for ONNX import/export")
        
        # Use registry for automatically registered operations
        from julia.core.utils.op_registry import registry
        self.registry = registry
        
        # Additional manual mappings for ONNX-specific ops that might differ in naming -> Will have to deep dive into the onnx docs more
        self.onnx_to_julia_ops = {
            "Relu": "ReLU",  # ONNX uses "Relu"
            "Gemm": "MatMul",  # ONNX's Gemm can be mapped to our MatMul (with attrs)
            "Constant": "Constant",
            "Identity": "Identity",
        }
            
        # Mapping dtypes between us & onnx -> same as before ^^
        self.dtype_map = {
            onnx.TensorProto.FLOAT: DataType.FLOAT32,
            onnx.TensorProto.DOUBLE: DataType.FLOAT64,
            onnx.TensorProto.INT32: DataType.INT32,
            onnx.TensorProto.INT64: DataType.INT64,
            onnx.TensorProto.BOOL: DataType.BOOL,
        }
    
    def import_model(self, model_path: str) -> IRGraph:
        """Import an ONNX model from file"""
        model = onnx.load(model_path)
        return self.import_from_model(model)
    
    def map_onnx_op_type(self, onnx_op_type: str) -> str:
        """Map ONNX op type to Julia op type"""
        from julia.core.utils.onnx_registry import onnx_registry
        return onnx_registry.get_julia_op_type(onnx_op_type)
    
    def import_from_model(self, model: Any) -> IRGraph:
        """Import from an in-memory ONNX model"""
        graph = IRGraph(name=model.graph.name or "imported_model")
        
        # Build a map of value info
        value_info = {}
        for info in model.graph.input:
            value_info[info.name] = info
        for info in model.graph.output:
            value_info[info.name] = info
        for info in model.graph.value_info:
            value_info[info.name] = info
            
        # Process initializers (constants)
        initializers = {}
        for initializer in model.graph.initializer:
            array = numpy_helper.to_array(initializer)
            node = graph.add_constant(array, name=initializer.name)
            initializers[initializer.name] = node
            
        # Process inputs
        input_map = {}
        for input_info in model.graph.input:
            # Skip inputs that have initializers
            if input_info.name in initializers:
                input_map[input_info.name] = initializers[input_info.name]
                continue
                
            # Get shape
            shape = []
            for dim in input_info.type.tensor_type.shape.dim:
                if dim.dim_param:
                    # Dynamic dimension
                    shape.append(-1)
                else:
                    shape.append(dim.dim_value)
                    
            # Get data type
            onnx_dtype = input_info.type.tensor_type.elem_type
            dtype = self.dtype_map.get(onnx_dtype, DataType.FLOAT32)
            
            # Create placeholder node
            node = graph.add_placeholder(shape=tuple(shape), dtype=dtype, name=input_info.name)
            input_map[input_info.name] = node
            
        # Process nodes
        node_map = {}  
        node_map.update(input_map)  # Add inputs to the map
        
        for onnx_node in model.graph.node:
            inputs = []
            for input_name in onnx_node.input:
                if input_name in node_map:
                    inputs.append(node_map[input_name])
                elif input_name in initializers:
                    inputs.append(initializers[input_name])
                else:
                    raise ValueError(f"Unknown input: {input_name}")
            
            # Check if we have a custom converter for this op type
            from julia.core.utils.onnx_registry import onnx_registry
            custom_converter = onnx_registry.get_import_converter(onnx_node.op_type)
            
            if custom_converter:
                # Use onnx registry for operations  
                node = custom_converter(self, onnx_node, inputs)
                if node.id not in graph.nodes:
                    graph.add_node(node)
            else:
                # Map op type using registry
                op_type = self.map_onnx_op_type(onnx_node.op_type)
                
                # Get attributes
                attributes = {}
                for attr in onnx_node.attribute:
                    if attr.type == onnx.AttributeProto.FLOAT:
                        attributes[attr.name] = attr.f
                    elif attr.type == onnx.AttributeProto.INT:
                        attributes[attr.name] = attr.i
                    elif attr.type == onnx.AttributeProto.STRING:
                        attributes[attr.name] = attr.s.decode('utf-8')
                    elif attr.type == onnx.AttributeProto.TENSOR:
                        attributes[attr.name] = numpy_helper.to_array(attr.t)
                    elif attr.type == onnx.AttributeProto.FLOATS:
                        attributes[attr.name] = list(attr.floats)
                    elif attr.type == onnx.AttributeProto.INTS:
                        attributes[attr.name] = list(attr.ints)
                    elif attr.type == onnx.AttributeProto.STRINGS:
                        attributes[attr.name] = [s.decode('utf-8') for s in attr.strings]
                    # Add more attributes or just come up with a more adaptive solution 

                # Create IR node
                node = IRNode(op_type=op_type, inputs=inputs, attributes=attributes, 
                              name=onnx_node.name or None)
                graph.add_node(node)
                
            # Add to node map
            for output_name in onnx_node.output:
                node_map[output_name] = node
                
            # Handle shapes and types if available
            for output_name in onnx_node.output:
                if output_name in value_info:
                    info = value_info[output_name]
                    shape = []
                    for dim in info.type.tensor_type.shape.dim:
                        if dim.dim_param:
                            shape.append(-1)
                        else:
                            shape.append(dim.dim_value)
                    node.shape = tuple(shape)
                    
                    # Get data type
                    onnx_dtype = info.type.tensor_type.elem_type
                    node.dtype = self.dtype_map.get(onnx_dtype, DataType.FLOAT32)
                    
        # Set graph outputs
        graph.outputs = [node_map[output.name] for output in model.graph.output]
        
        # Infer shapes for nodes without explicit shape info
        graph.infer_shapes()          
        return graph

class ONNXExporter:
    """
    Export a Julia IR graph to ONNX model -> Import but reverse 
    """
    def __init__(self):
        if not ONNX_AVAILABLE:
            raise ImportError("onnx package is required for ONNX import/export")
            
        # Use registry 
        from julia.core.utils.op_registry import registry
        self.registry = registry
        
        self.julia_to_onnx_ops = {
            "ReLU": "Relu",      
        }
        
        self.dtype_map = {
            DataType.FLOAT32: onnx.TensorProto.FLOAT,
            DataType.FLOAT64: onnx.TensorProto.DOUBLE,
            DataType.INT32: onnx.TensorProto.INT32,
            DataType.INT64: onnx.TensorProto.INT64,
            DataType.BOOL: onnx.TensorProto.BOOL,
        }
    
    def map_julia_op_type(self, julia_op_type: str) -> str:
        """Map Julia op type to ONNX op type"""
        from julia.core.utils.onnx_registry import onnx_registry
        return onnx_registry.get_onnx_op_type(julia_op_type)
    
    def export_model(self, graph: IRGraph, file_path: str, opset_version: int = 13):
        """Export the graph to an ONNX model file"""
        model = self.export_to_model(graph, opset_version)
        onnx.save(model, file_path)

    def export_to_model(self, graph: IRGraph, opset_version: int = 13) -> Any:
        """Export to an in-memory ONNX model"""
        # Make sure all nodes have shapes
        nodes_without_shape = [node for node in graph.nodes.values() if node.shape is None]
        if nodes_without_shape:
            print(f"Warning: The following nodes don't have shapes inferred: {[node.name for node in nodes_without_shape]}")
            graph.infer_shapes()
            
            nodes_without_shape = [node for node in graph.nodes.values() if node.shape is None]
            if nodes_without_shape:
                nodes_str = ", ".join([f"{node.name} ({node.op_type})" for node in nodes_without_shape])
                raise ValueError(f"After shape inference, these nodes still have no shapes: {nodes_str}")
        
        # Make sure all nodes have types
        nodes_without_type = [node for node in graph.nodes.values() if node.dtype is None]
        if nodes_without_type:
            nodes_str = ", ".join([f"{node.name} ({node.op_type})" for node in nodes_without_type])
            raise ValueError(f"The following nodes don't have data types inferred: {nodes_str}")
            
        # Create ONNX graph
        onnx_nodes = []
        onnx_inputs = []
        onnx_outputs = []
        onnx_initializers = []
        onnx_value_info = []
        
        # Keep track of processed nodes and their outputs
        processed_outputs = {}
        
        # First, process inputs and constants
        for input_node in graph.inputs:
            # Create tensor type
            elem_type = self.dtype_map.get(input_node.dtype, onnx.TensorProto.FLOAT)
            
            # Create tensor shape
            shape = []
            for dim in input_node.shape:
                shape.append(onnx.TensorShapeProto.Dimension(dim_value=dim if dim > 0 else 1))
                
            tensor_type = onnx.TypeProto.Tensor(
                elem_type=elem_type,
                shape=onnx.TensorShapeProto(dim=shape)
            )
            type_proto = onnx.TypeProto(tensor_type=tensor_type)
            
            # Create value info
            value_info = onnx.ValueInfoProto(
                name=input_node.name,
                type=type_proto
            )
            
            onnx_inputs.append(value_info)
            processed_outputs[input_node.id] = input_node.name
        
        # Process constants
        for node_id, node in graph.nodes.items():
            if isinstance(node, ConstantNode):
                # Create tensor
                tensor = numpy_helper.from_array(node.attributes["value"], name=node.name)
                onnx_initializers.append(tensor)
                processed_outputs[node.id] = node.name
        
        # Topologically sort all nodes (including inputs and constants)
        all_nodes = graph.topological_sort()
        
        # Now process all operation nodes in topological order
        for node in all_nodes:
            # Skip already processed nodes (inputs and constants)
            if node.id in processed_outputs:
                continue
                
            # Make sure all input nodes are in processed_outputs
            missing_inputs = [input_node for input_node in node.inputs 
                             if input_node.id not in processed_outputs]
            
            if missing_inputs:
                raise ValueError(f"Node {node.name} ({node.op_type}) has inputs that have not been processed: "
                               f"{[input_node.name for input_node in missing_inputs]}")
            
            # Get input names
            input_names = [processed_outputs[input_node.id] for input_node in node.inputs]
            
            # Same deal as before its just reversed for exporting -> check `/utils/`
            from julia.core.utils.onnx_registry import onnx_registry
            try:
                custom_converter = onnx_registry.get_export_converter(node.op_type)
            except (ImportError, AttributeError):
                custom_converter = None
            
            if custom_converter:
                onnx_node = custom_converter(self, node, input_names)
                onnx_nodes.append(onnx_node)
                if hasattr(onnx_node, 'output') and onnx_node.output:
                    processed_outputs[node.id] = onnx_node.output[0]
                else:
                    processed_outputs[node.id] = node.name or f"{node.op_type}_{node.id[:8]}"
            else:
                try:
                    onnx_op_type = self.map_julia_op_type(node.op_type)
                except (ImportError, AttributeError):
                    onnx_op_type = node.op_type
                
                output_name = node.name or f"{onnx_op_type}_{node.id[:8]}"
                processed_outputs[node.id] = output_name
                
                # Create attributes
                attrs = []
                if hasattr(node, 'attributes') and node.attributes:
                    for attr_name, attr_value in node.attributes.items():
                        if isinstance(attr_value, float):
                            attr = onnx.helper.make_attribute(attr_name, attr_value)
                        elif isinstance(attr_value, int):
                            attr = onnx.helper.make_attribute(attr_name, attr_value)
                        elif isinstance(attr_value, str):
                            attr = onnx.helper.make_attribute(attr_name, attr_value)
                        elif isinstance(attr_value, (list, tuple)):
                            if all(isinstance(x, float) for x in attr_value):
                                attr = onnx.helper.make_attribute(attr_name, list(attr_value))
                            elif all(isinstance(x, int) for x in attr_value):
                                attr = onnx.helper.make_attribute(attr_name, list(attr_value))
                            elif all(isinstance(x, str) for x in attr_value):
                                attr = onnx.helper.make_attribute(attr_name, list(attr_value))
                            else:
                                continue  # Skip complex attributes
                        elif isinstance(attr_value, np.ndarray):
                            tensor = numpy_helper.from_array(attr_value)
                            attr = onnx.helper.make_attribute(attr_name, tensor)
                        else:
                            continue  # Skip unknown attribute types
                            
                        attrs.append(attr)
                
                # Create ONNX node
                onnx_node = onnx.NodeProto(
                    op_type=onnx_op_type,
                    input=input_names,
                    output=[output_name],
                    name=node.name or "",
                    attribute=attrs
                )
                
                onnx_nodes.append(onnx_node)
            
            # Create value info for output
            if node.shape is not None and node.dtype is not None:
                elem_type = self.dtype_map.get(node.dtype, onnx.TensorProto.FLOAT)
                shape = []
                for dim in node.shape:
                    shape.append(onnx.TensorShapeProto.Dimension(dim_value=dim if dim > 0 else 1))
                    
                tensor_type = onnx.TypeProto.Tensor(
                    elem_type=elem_type,
                    shape=onnx.TensorShapeProto(dim=shape)
                )
                type_proto = onnx.TypeProto(tensor_type=tensor_type)
                
                value_info = onnx.ValueInfoProto(
                    name=processed_outputs[node.id],
                    type=type_proto
                )
                
                onnx_value_info.append(value_info)
        
        # Process outputs
        for output_node in graph.outputs:
            if output_node.id not in processed_outputs:
                raise ValueError(f"Output node {output_node.name} was not processed during graph traversal")
                
            output_name = processed_outputs[output_node.id]
            
            # Create tensor type
            elem_type = self.dtype_map.get(output_node.dtype, onnx.TensorProto.FLOAT)
            
            # Create tensor shape
            shape = []
            for dim in output_node.shape:
                shape.append(onnx.TensorShapeProto.Dimension(dim_value=dim if dim > 0 else 1))
                
            tensor_type = onnx.TypeProto.Tensor(
                elem_type=elem_type,
                shape=onnx.TensorShapeProto(dim=shape)
            )
            type_proto = onnx.TypeProto(tensor_type=tensor_type)
            
            # Create value info
            value_info = onnx.ValueInfoProto(
                name=output_name,
                type=type_proto
            )
            
            onnx_outputs.append(value_info)
        
        # Create ONNX graph
        onnx_graph = onnx.GraphProto(
            node=onnx_nodes,
            name=graph.name,
            input=onnx_inputs,
            output=onnx_outputs,
            initializer=onnx_initializers,
            value_info=onnx_value_info
        )
        
        # Create ONNX model
        opset_imports = [onnx.helper.make_operatorsetid("", opset_version)]
        model = onnx.ModelProto(
            ir_version=7,  # Use latest IR version
            producer_name="Julia",
            producer_version="0.1",
            domain="",
            model_version=1,
            doc_string="Exported from Julia IR",
            graph=onnx_graph,
            opset_import=opset_imports
        )
        
        return model

def import_onnx(model_path: str) -> IRGraph:
    """Helper function to import an ONNX model"""
    if not ONNX_AVAILABLE:
        raise ImportError("onnx package is required for ONNX import/export")
    importer = ONNXImporter()
    return importer.import_model(model_path)


def export_onnx(graph: IRGraph, file_path: str, opset_version: int = 13):
    """Helper function to export a graph to ONNX"""
    if not ONNX_AVAILABLE:
        raise ImportError("onnx package is required for ONNX import/export")
    exporter = ONNXExporter()
    exporter.export_model(graph, file_path, opset_version)
