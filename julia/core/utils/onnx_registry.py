"""
ONNX Conversion Registry for Operation Mapping -> just going to have to continue reading the ONNX documentation and messing with it for a better solution. 
This is just a more focused registry for ONNX its really similar to `op_registry.py`
"""
from typing import Dict, Callable, Optional
import numpy as np

try:
    import onnx
    from onnx import numpy_helper
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False



class ONNXConversionRegistry:
    """
    Registry for ONNX conversion functions
    """
    _instance = None  # Singleton instance
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        # Only initialize if this is the singleton instance
        if self.__class__._instance is not None:
            return
        
        # Map from Julia op_type to ONNX op_type
        self.julia_to_onnx: Dict[str, str] = {
            "ReLU": "Relu",      
            "Identity": "Identity",
        }
        
        # Map from ONNX op_type to Julia op_type
        self.onnx_to_julia: Dict[str, str] = {
            "Relu": "ReLU",      
            "Gemm": "MatMul",    # ONNX's Gemm can be mapped to our MatMul (with attrs)
            "Identity": "Identity",
        }
        
        # Specialized import converters: ONNX op_type -> converter function
        # Function signature: converter(importer, onnx_node, inputs) -> IRNode
        self.import_converters: Dict[str, Callable] = {}
        
        # Specialized export converters: Julia op_type -> converter function
        # Function signature: converter(exporter, ir_node, input_names) -> onnx.NodeProto
        self.export_converters: Dict[str, Callable] = {}
    
    def register_import(self, onnx_op_type: str, julia_op_type: Optional[str] = None):
        """
        Register an ONNX import mapping
        
        Args:
            onnx_op_type: The ONNX operation type
            julia_op_type: The Julia operation type (if None, uses onnx_op_type)
        
        Can be used as a decorator for import converter:
        @registry.register_import("Conv")
        def import_conv(importer, onnx_node, inputs):
            ...
        """
        def decorator(func=None):
            if julia_op_type is not None:
                self.onnx_to_julia[onnx_op_type] = julia_op_type
            
            if func is not None:
                self.import_converters[onnx_op_type] = func
            
            return func
        
        return decorator
    
    def register_export(self, julia_op_type: str, onnx_op_type: Optional[str] = None):
        """
        Register an ONNX export mapping
        
        Args:
            julia_op_type: The Julia operation type
            onnx_op_type: The ONNX operation type (if None, uses julia_op_type)
        
        Can be used as a decorator for export converter:
        @registry.register_export("Conv")
        def export_conv(exporter, ir_node, input_names):
            ...
        """
        def decorator(func=None):
            if onnx_op_type is not None:
                self.julia_to_onnx[julia_op_type] = onnx_op_type
            
            if func is not None:
                self.export_converters[julia_op_type] = func
            
            return func
        
        return decorator
    
    def get_julia_op_type(self, onnx_op_type: str) -> str:
        """Get the <lib> operation type for an ONNX operation type"""
        return self.onnx_to_julia.get(onnx_op_type, onnx_op_type)
    
    def get_onnx_op_type(self, julia_op_type: str) -> str:
        """Get the ONNX operation type for a <lib> operation type"""
        return self.julia_to_onnx.get(julia_op_type, julia_op_type)
    
    def get_import_converter(self, onnx_op_type: str) -> Optional[Callable]:
        """Get the import converter for an ONNX operation type"""
        return self.import_converters.get(onnx_op_type)
    
    def get_export_converter(self, julia_op_type: str) -> Optional[Callable]:
        """Get the export converter for a <lib> operation type"""
        return self.export_converters.get(julia_op_type)


# Create singleton instance
onnx_registry = ONNXConversionRegistry.get_instance()


# Register standard converters

if ONNX_AVAILABLE:
    @onnx_registry.register_import("Gemm")
    def import_gemm(importer, onnx_node, inputs):
        """Import ONNX Gemm (Generalized Matrix Multiplication) to <lib> MatMul and Add"""
        from julia.core.ir import IRNode
        
        # Get attributes with defaults
        alpha = 1.0
        beta = 1.0
        transA = 0
        transB = 0
        
        for attr in onnx_node.attribute:
            if attr.name == 'alpha':
                alpha = attr.f
            elif attr.name == 'beta':
                beta = attr.f
            elif attr.name == 'transA':
                transA = attr.i
            elif attr.name == 'transB':
                transB = attr.i
        
        # Get inputs
        A, B = inputs[:2]
        C = inputs[2] if len(inputs) > 2 else None
        
        # Handle transpositions
        if transA:
            # Create transpose node for A
            transposeA = IRNode(
                op_type="Transpose",
                inputs=[A],
                name=f"{onnx_node.name}_transA" if onnx_node.name else None
            )
            A = transposeA
        
        if transB:
            # Create transpose node for B
            transposeB = IRNode(
                op_type="Transpose",
                inputs=[B],
                name=f"{onnx_node.name}_transB" if onnx_node.name else None
            )
            B = transposeB
        
        # Create MatMul node
        matmul = IRNode(
            op_type="MatMul",
            inputs=[A, B],
            name=f"{onnx_node.name}_matmul" if onnx_node.name else None
        )
        
        # If alpha != 1.0, create a Mul node
        if alpha != 1.0:
            # Create constant for alpha
            from julia.core.ir import ConstantNode
            alpha_const = ConstantNode(
                np.array(alpha, dtype=np.float32),
                name=f"{onnx_node.name}_alpha" if onnx_node.name else None
            )
            
            # Create Mul node
            mul = IRNode(
                op_type="Mul",
                inputs=[matmul, alpha_const],
                name=f"{onnx_node.name}_mul" if onnx_node.name else None
            )
            matmul = mul
        
        # If C is present, add it
        if C is not None:
            # If beta != 1.0, create a Mul node for C
            if beta != 1.0:
                # Create constant for beta
                from julia.core.ir import ConstantNode
                beta_const = ConstantNode(
                    np.array(beta, dtype=np.float32),
                    name=f"{onnx_node.name}_beta" if onnx_node.name else None
                )
                
                # Create Mul node
                mul_c = IRNode(
                    op_type="Mul",
                    inputs=[C, beta_const],
                    name=f"{onnx_node.name}_mulC" if onnx_node.name else None
                )
                C = mul_c
            
            # Create Add node
            add = IRNode(
                op_type="Add",
                inputs=[matmul, C],
                name=onnx_node.name if onnx_node.name else None
            )
            return add
        
        # If no C, just return matmul (possibly with alpha)
        return matmul


    @onnx_registry.register_export("MatMul")
    def export_matmul(exporter, ir_node, input_names):
        """Export Julia MatMul to ONNX MatMul"""
        if not ONNX_AVAILABLE:
            raise ImportError("onnx package is required for ONNX export")
            
        return onnx.helper.make_node(
            "MatMul",
            input_names,
            [ir_node.name],
            name=ir_node.name
        )


    @onnx_registry.register_export("Add", "Add")
    def export_add(exporter, ir_node, input_names):
        """Export Julia Add to ONNX Add"""
        if not ONNX_AVAILABLE:
            raise ImportError("onnx package is required for ONNX export")
            
        return onnx.helper.make_node(
            "Add",
            input_names,
            [ir_node.name],
            name=ir_node.name
        )
