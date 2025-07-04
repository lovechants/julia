import numpy as np
import uuid
import json
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Any, Set
from julia.core.tensor import Tensor


# TODO More elegant way to do do data types
class DataType(Enum):
    # Core types
    FLOAT16 = "float16"
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    BFLOAT16 = "bfloat16"

    # Integer types
    INT8 = "int8"
    INT16 = "int16"
    INT32 = "int32"
    INT64 = "int64"
    UINT8 = "uint8"
    UINT16 = "uint16"
    UINT32 = "uint32"
    UINT64 = "uint64"

    # Special types
    BOOL = "bool"
    COMPLEX64 = "complex64"
    COMPLEX128 = "complex128"

    @staticmethod
    def from_numpy(dtype) -> "DataType":
        """Enhanced numpy dtype mapping"""
        mapping = {
            np.float16: DataType.FLOAT16,
            np.float32: DataType.FLOAT32,
            np.float64: DataType.FLOAT64,
            np.int8: DataType.INT8,
            np.int16: DataType.INT16,
            np.int32: DataType.INT32,
            np.int64: DataType.INT64,
            np.uint8: DataType.UINT8,
            np.uint16: DataType.UINT16,
            np.uint32: DataType.UINT32,
            np.uint64: DataType.UINT64,
            np.bool_: DataType.BOOL,
            np.complex64: DataType.COMPLEX64,
            np.complex128: DataType.COMPLEX128,
        }

        # Handle string representations
        if isinstance(dtype, str):
            try:
                return DataType(dtype)
            except ValueError:
                pass

        # Handle numpy dtype objects
        dtype_type = dtype.type if hasattr(dtype, "type") else type(dtype)
        return mapping.get(dtype_type, DataType.FLOAT32)

    def to_numpy(self):
        """Convert to numpy dtype"""
        mapping = {
            DataType.FLOAT16: np.float16,
            DataType.FLOAT32: np.float32,
            DataType.FLOAT64: np.float64,
            DataType.BFLOAT16: np.float32,  # Fallback for bfloat16
            DataType.INT8: np.int8,
            DataType.INT16: np.int16,
            DataType.INT32: np.int32,
            DataType.INT64: np.int64,
            DataType.UINT8: np.uint8,
            DataType.UINT16: np.uint16,
            DataType.UINT32: np.uint32,
            DataType.UINT64: np.uint64,
            DataType.BOOL: np.bool_,
            DataType.COMPLEX64: np.complex64,
            DataType.COMPLEX128: np.complex128,
        }
        return mapping[self]

    def size_bytes(self) -> int:
        """Get size in bytes"""
        sizes = {
            DataType.FLOAT16: 2,
            DataType.BFLOAT16: 2,
            DataType.FLOAT32: 4,
            DataType.FLOAT64: 8,
            DataType.INT8: 1,
            DataType.UINT8: 1,
            DataType.INT16: 2,
            DataType.UINT16: 2,
            DataType.INT32: 4,
            DataType.UINT32: 4,
            DataType.INT64: 8,
            DataType.UINT64: 8,
            DataType.BOOL: 1,
            DataType.COMPLEX64: 8,
            DataType.COMPLEX128: 16,
        }
        return sizes[self]


class IRNode:
    """
    Node in the computation graph
    """

    def __init__(
        self,
        op_type: str,
        inputs: List["IRNode"] = None,
        attributes: Dict[str, Any] = None,
        name: str = None,
    ):
        self.id = str(uuid.uuid4())
        self.op_type = op_type
        self.inputs = inputs or []
        self.attributes = attributes or {}
        self.outputs = []  # Nodes that use this node as input
        self.name = name or f"{op_type}_{self.id[:8]}"
        self.shape = None  # Will be set during shape inference
        self.dtype = None  # Will be set during type inference

        # Connect this node to its inputs
        for input_node in self.inputs:
            input_node.outputs.append(self)

    def __repr__(self):
        return f"IRNode(type={self.op_type}, name={self.name}, shape={self.shape})"

    def to_dict(self):
        """Convert node to dictionary for serialization"""
        return {
            "id": self.id,
            "op_type": self.op_type,
            "name": self.name,
            "inputs": [node.id for node in self.inputs],
            "attributes": self._serialize_attributes(),
            "shape": self.shape,
            "dtype": self.dtype.value if self.dtype else None,
        }

    def _serialize_attributes(self):
        """Helper to serialize attribute values"""
        result = {}
        for k, v in self.attributes.items():
            if isinstance(v, np.ndarray):
                result[k] = v.tolist()
            elif isinstance(v, Tensor):
                result[k] = v.data.tolist()
            else:
                result[k] = v
        return result

    def transpose(self):
        """
        Create a transpose node for the given node

        This is a helper function to create a transpose operation in the IR graph
        """
        if hasattr(self, "shape") and self.shape:
            # Get the shape and reverse the last two dimensions
            if len(self.shape) < 2:
                raise ValueError(
                    f"Cannot transpose node with shape {self.shape} - needs at least 2 dimensions"
                )

            # Create a permutation that swaps the last two dimensions
            perm = list(range(len(self.shape)))
            perm[-1], perm[-2] = perm[-2], perm[-1]

            # Create a transpose node
            transpose_node = IRNode(
                op_type="Transpose",
                inputs=[self],
                attributes={"perm": perm},
                name=f"{self.name}_transpose" if self.name else None,
            )

            # Set the shape - swap the last two dimensions
            if self.shape:
                new_shape = list(self.shape)
                new_shape[-1], new_shape[-2] = new_shape[-2], new_shape[-1]
                transpose_node.shape = tuple(new_shape)

            # Set the dtype to match the input
            if hasattr(self, "dtype"):
                transpose_node.dtype = self.dtype
            return transpose_node
        else:
            raise ValueError("Cannot transpose node without shape information")


class ConstantNode(IRNode):
    """Node representing a constant value"""

    def __init__(self, value: Union[np.ndarray, Tensor], name: str = None):
        if isinstance(value, Tensor):
            data = value.data
        else:
            data = value

        super().__init__(
            op_type="Constant",
            inputs=[],
            attributes={"value": data},
            name=name or f"const_{uuid.uuid4()[:8]}",
        )
        self.shape = data.shape
        self.dtype = DataType.from_numpy(data.dtype)


class VariableNode(IRNode):
    """Node representing a variable (parameter or input)"""

    def __init__(
        self,
        shape: Tuple[int, ...],
        dtype: DataType = DataType.FLOAT32,
        name: str = None,
        trainable: bool = True,
    ):
        super().__init__(
            op_type="Variable",
            inputs=[],
            attributes={"trainable": trainable},
            name=name or f"var_{uuid.uuid4()[:8]}",
        )
        self.shape = shape
        self.dtype = dtype


class PlaceholderNode(IRNode):
    """Node representing an input placeholder"""

    def __init__(
        self,
        shape: Tuple[int, ...],
        dtype: DataType = DataType.FLOAT32,
        name: str = None,
    ):
        super().__init__(
            op_type="Placeholder",
            inputs=[],
            attributes={},
            name=name or f"input_{uuid.uuid4()[:8]}",
        )
        self.shape = shape
        self.dtype = dtype


class IRGraph:
    """
    Computation graph representation
    """

    def __init__(self, name: str = "graph"):
        self.name = name
        self.nodes: Dict[str, IRNode] = {}
        self.inputs: List[IRNode] = []
        self.outputs: List[IRNode] = []
        self.parameters: List[IRNode] = []

    def add_node(self, node: IRNode) -> IRNode:
        """Add a node to the graph"""
        self.nodes[node.id] = node
        return node

    def add_placeholder(
        self,
        shape: Tuple[int, ...],
        dtype: DataType = DataType.FLOAT32,
        name: str = None,
    ) -> IRNode:
        """Add an input placeholder node"""
        node = PlaceholderNode(shape, dtype, name)
        self.nodes[node.id] = node
        self.inputs.append(node)
        return node

    def add_variable(
        self,
        shape: Tuple[int, ...],
        dtype: DataType = DataType.FLOAT32,
        name: str = None,
        trainable: bool = True,
    ) -> IRNode:
        """Add a variable node"""
        node = VariableNode(shape, dtype, name, trainable)
        self.nodes[node.id] = node
        if trainable:
            self.parameters.append(node)
        return node

    def add_constant(
        self, value: Union[np.ndarray, Tensor], name: str = None
    ) -> IRNode:
        """Add a constant node"""
        node = ConstantNode(value, name)
        self.nodes[node.id] = node
        return node

    def set_outputs(self, outputs: List[IRNode]):
        """Set the output nodes of the graph"""
        self.outputs = outputs

    def serialize(self) -> str:
        """Serialize the graph to JSON"""
        graph_dict = {
            "name": self.name,
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "inputs": [node.id for node in self.inputs],
            "outputs": [node.id for node in self.outputs],
            "parameters": [node.id for node in self.parameters],
        }
        return json.dumps(graph_dict, indent=2)

    @classmethod
    def deserialize(cls, json_str: str) -> "IRGraph":
        """Deserialize a graph from JSON"""
        graph_dict = json.loads(json_str)
        graph = cls(name=graph_dict["name"])

        # First pass: create all nodes
        node_map = {}
        for node_dict in graph_dict["nodes"]:
            if node_dict["op_type"] == "Constant":
                value = np.array(node_dict["attributes"]["value"])
                node = ConstantNode(value, name=node_dict["name"])
            elif node_dict["op_type"] == "Variable":
                node = VariableNode(
                    shape=node_dict["shape"],
                    dtype=DataType(node_dict["dtype"]),
                    name=node_dict["name"],
                    trainable=node_dict["attributes"].get("trainable", True),
                )
            elif node_dict["op_type"] == "Placeholder":
                node = PlaceholderNode(
                    shape=node_dict["shape"],
                    dtype=DataType(node_dict["dtype"]),
                    name=node_dict["name"],
                )
            else:
                node = IRNode(
                    op_type=node_dict["op_type"],
                    inputs=[],  # Will connect in second pass
                    attributes=node_dict["attributes"],
                    name=node_dict["name"],
                )
                node.shape = node_dict["shape"]
                if node_dict["dtype"]:
                    node.dtype = DataType(node_dict["dtype"])

            # Override auto-generated id with the saved one
            node.id = node_dict["id"]
            node_map[node.id] = node
            graph.nodes[node.id] = node

        # Second pass: connect nodes
        for node_dict in graph_dict["nodes"]:
            node = node_map[node_dict["id"]]
            for input_id in node_dict["inputs"]:
                input_node = node_map[input_id]
                node.inputs.append(input_node)
                input_node.outputs.append(node)

        # Set inputs, outputs, and parameters
        graph.inputs = [node_map[node_id] for node_id in graph_dict["inputs"]]
        graph.outputs = [node_map[node_id] for node_id in graph_dict["outputs"]]
        graph.parameters = [node_map[node_id] for node_id in graph_dict["parameters"]]

        return graph

    def topological_sort(self) -> List[IRNode]:
        """Sort nodes in topological order"""
        visited: Set[str] = set()
        topo_order: List[IRNode] = []

        def visit(node: IRNode):
            if node.id in visited:
                return
            visited.add(node.id)
            for input_node in node.inputs:
                visit(input_node)
            topo_order.append(node)

        # Start from output nodes
        for output_node in self.outputs:
            visit(output_node)

        return topo_order

    def infer_shapes(self):
        """Infer shapes of all nodes in the graph"""
        from julia.core.utils.op_registry import registry

        # Process nodes in topological order
        topo_order = self.topological_sort()
        print(
            f"Topological order for shape inference: {[node.name for node in topo_order]}"
        )

        for node in topo_order:
            # Skip nodes that already have shapes
            if node.shape is not None:
                print(f"Node {node.name} already has shape {node.shape}")
                continue

            # Skip nodes that don't need shape inference
            if node.op_type in ["Variable", "Constant", "Placeholder"]:
                print(f"Node {node.name} is {node.op_type}, skipping shape inference")
                continue

            # Get input shapes
            input_shapes = []
            all_shapes_available = True

            for input_node in node.inputs:
                if input_node.shape is None:
                    # If any input doesn't have a shape, we can't infer this node's shape
                    print(f"Input node {input_node.name} for {node.name} has no shape")
                    all_shapes_available = False
                    break
                input_shapes.append(input_node.shape)

            if not all_shapes_available:
                # Some inputs don't have shapes yet
                print(f"Not all inputs for {node.name} have shapes")
                continue

            # If we reach here, all inputs have shapes
            print(f"All inputs for {node.name} have shapes: {input_shapes}")

            # Get shape inference function
            infer_func = registry.get_shape_inference(node.op_type)
            if infer_func:
                print(f"Found shape inference function for {node.op_type}")
                try:
                    shape = infer_func(node, input_shapes)
                    node.shape = shape
                    print(f"Inferred shape for {node.name}: {shape}")
                except Exception as e:
                    print(f"Error inferring shape for {node.name}: {e}")
            else:
                print(f"No shape inference function found for {node.op_type}")

        # Check if all nodes have shapes
        nodes_without_shape = [
            node for node in self.nodes.values() if node.shape is None
        ]
        if nodes_without_shape:
            node_names = [node.name for node in nodes_without_shape]
            print(f"Warning: Could not infer shapes for nodes: {node_names}")
            return False

        return True

    def optimize(self):
        """Apply optimization passes to the graph"""
        # TODO: Implement common optimization passes
        # - Constant folding
        # - Common subexpression elimination
        # - Dead code elimination
        pass


# TODO enhance the shape representation with more dynamic dimensions
class Shape:
    """Enhanced shape representation with dynamic dimensions"""

    def __init__(self, dims: Union[Tuple[int, ...], list]):
        self.dims = tuple(dims) if dims else ()

    def __getitem__(self, idx):
        return self.dims[idx]

    def __len__(self):
        return len(self.dims)

    def __iter__(self):
        return iter(self.dims)

    def __eq__(self, other):
        if isinstance(other, (tuple, list)):
            return self.dims == tuple(other)
        return self.dims == other.dims

    def __repr__(self):
        return f"Shape{self.dims}"

    @property
    def numel(self) -> int:
        """Number of elements"""
        if not self.dims:
            return 1
        result = 1
        for dim in self.dims:
            if dim < 0:  # Dynamic dimension
                return -1
            result *= dim
        return result

    def is_compatible_with(self, other: "Shape") -> bool:
        """Check if shapes are compatible for broadcasting"""
        try:
            a = np.empty(self.dims)
            b = np.empty(other.dims)
            np.broadcast_arrays(a, b)
            return True
        except ValueError:
            return False

    def broadcast_with(self, other: "Shape") -> "Shape":
        """Get broadcast result shape"""
        try:
            a = np.empty(self.dims)
            b = np.empty(other.dims)
            result = np.broadcast_arrays(a, b)[0]
            return Shape(result.shape)
        except ValueError:
            raise ValueError(f"Cannot broadcast shapes {self} and {other}")


def infer_broadcast_shape(shapes: list) -> Optional[Shape]:
    """Infer shape from multiple input shapes with broadcasting"""
    if not shapes:
        return None

    try:
        # Create dummy arrays and use numpy's broadcasting
        arrays = [
            np.empty(shape.dims if hasattr(shape, "dims") else shape)
            for shape in shapes
        ]
        result = np.broadcast_arrays(*arrays)[0]
        return Shape(result.shape)
    except ValueError:
        return None
