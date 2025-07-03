import ctypes
from typing import Tuple, Any
import numpy as np

try:
    import llvmlite.binding as llvm
    import llvmlite.ir as ir
    LLVM_AVAILABLE = True
except ImportError:
    LLVM_AVAILABLE = False
    print("Warning: llvmlite not found. LLVM compilation will not be available.")

from julia.core.ir import IRGraph, DataType

class LLVMCompiler:
    """
    Compiles Julia IR graph to LLVM IR for efficient execution
    """
    def __init__(self):
        if not LLVM_AVAILABLE:
            raise ImportError("llvmlite is required for LLVM compilation")
        
        # Initialize LLVM
        llvm.initialize()
        llvm.initialize_native_target()
        llvm.initialize_native_asmprinter()
        
        # Set up target information
        self.target = llvm.Target.from_default_triple()
        self.target_machine = self.target.create_target_machine()
        self.data_layout = self.target_machine.target_data
        
        # Create module passes - use only passes that are available in llvmlite -> read the llvmlite docs 
        self.pass_manager = llvm.create_module_pass_manager()
        self.pass_manager.add_instruction_combining_pass()
        self.pass_manager.add_cfg_simplification_pass()
        
    def _convert_type(self, dtype: DataType, shape: Tuple[int, ...]) -> ir.Type: #TODO This needs a cleaner solution I just want it started (like most of it atp)
        """Convert Julia DataType to LLVM IR type"""
        if dtype == DataType.FLOAT32:
            elem_type = ir.FloatType()
        elif dtype == DataType.FLOAT64:
            elem_type = ir.DoubleType()
        elif dtype == DataType.INT32:
            elem_type = ir.IntType(32)
        elif dtype == DataType.INT64:
            elem_type = ir.IntType(64)
        elif dtype == DataType.BOOL:
            elem_type = ir.IntType(1)
        else:
            raise ValueError(f"Unsupported data type: {dtype}")
            
        # For scalar 
        if not shape or len(shape) == 0:
            # Return pointer to the element type for scalar parameters
            return ir.PointerType(elem_type)
            
        # For arrays (1D or more), use pointers to element type
        return ir.PointerType(elem_type)
        
    def _create_module(self, graph: IRGraph) -> ir.Module:
        """Create LLVM IR module from graph"""
        module = ir.Module(name=f"julia_module_{graph.name}")
        module.triple = self.target.triple
        module.data_layout = str(self.data_layout)
        
        # Create function type
        # Inputs: all graph inputs as pointers
        input_types = [self._convert_type(node.dtype, node.shape) for node in graph.inputs]
        # Output: pointer to output buffer
        output_types = [self._convert_type(node.dtype, node.shape) for node in graph.outputs]
        
        # Create function
        func_type = ir.FunctionType(ir.VoidType(), input_types + output_types)
        func = ir.Function(module, func_type, name=f"execute_{graph.name}")
        
        # Create entry block
        block = func.append_basic_block(name="entry")
        builder = ir.IRBuilder(block)
        
        # Map from IR nodes to LLVM values
        value_map = {}
        
        # Process inputs
        for i, input_node in enumerate(graph.inputs):
            value_map[input_node.id] = func.args[i]
        
        # Process nodes in topological order
        for node in graph.topological_sort():
            if node.id in value_map:  # Skip inputs which are already processed
                continue
                
            if node.op_type == "Add":
                lhs = value_map[node.inputs[0].id]
                rhs = value_map[node.inputs[1].id]
                
                # For scalar or vector addition
                if not node.shape or len(node.shape) <= 1:
                    # Extract the actual elements via load if they're pointers
                    if isinstance(lhs.type, ir.PointerType):
                        lhs = builder.load(lhs)
                    if isinstance(rhs.type, ir.PointerType):
                        rhs = builder.load(rhs)
                        
                    if node.dtype in [DataType.FLOAT32, DataType.FLOAT64]:
                        result = builder.fadd(lhs, rhs, name=f"add_{node.id[:8]}")
                    else:
                        result = builder.add(lhs, rhs, name=f"add_{node.id[:8]}")
                    
                    value_map[node.id] = result
                else:
                    # For multi-dimensional arrays, will have to loop #TODO soon LOL 
                    raise NotImplementedError("Multi-dimensional array addition not yet implemented")
                    
            elif node.op_type == "Mul": # Might need a lot more op type util than anticipated 
                lhs = value_map[node.inputs[0].id]
                rhs = value_map[node.inputs[1].id]
                
                # Extract the actual elements via load if they're pointers
                if isinstance(lhs.type, ir.PointerType):
                    lhs = builder.load(lhs)
                if isinstance(rhs.type, ir.PointerType):
                    rhs = builder.load(rhs)
                    
                if node.dtype in [DataType.FLOAT32, DataType.FLOAT64]:
                    result = builder.fmul(lhs, rhs, name=f"mul_{node.id[:8]}")
                else:
                    result = builder.mul(lhs, rhs, name=f"mul_{node.id[:8]}")
                
                value_map[node.id] = result
                    
                
        # Copy results to output pointers
        for i, output_node in enumerate(graph.outputs):
            result = value_map[output_node.id]
            output_ptr = func.args[len(graph.inputs) + i]
            
            # If result is not a pointer but output_ptr is a pointer, store
            if not isinstance(result.type, ir.PointerType) and isinstance(output_ptr.type, ir.PointerType):
                builder.store(result, output_ptr)
            # If both are pointers, load from result first
            elif isinstance(result.type, ir.PointerType) and isinstance(output_ptr.type, ir.PointerType):
                loaded_result = builder.load(result)
                builder.store(loaded_result, output_ptr)
            # If result is a pointer but output_ptr is not, load from result
            elif isinstance(result.type, ir.PointerType) and not isinstance(output_ptr.type, ir.PointerType):
                loaded_result = builder.load(result)
                # Output should be a pointer
                raise TypeError(f"Output is not a pointer but result is: {result.type} vs {output_ptr.type}")
            else:
                # Both are not pointers??? 
                raise TypeError(f"Both result and output are not pointers: {result.type} vs {output_ptr.type}")
            

        builder.ret_void()
        
        return module
        
    def compile(self, graph: IRGraph) -> Any:
        """
        Compile graph to executable function
        
        Returns a callable function that takes numpy arrays as inputs and returns numpy array outputs
        """
        # Make sure shapes are inferred
        graph.infer_shapes()
        
        # Create LLVM IR module
        ir_module = self._create_module(graph)
        
        # Verify module
        llvm_module = llvm.parse_assembly(str(ir_module))
        llvm_module.verify()
        
        # Optimize module
        self.pass_manager.run(llvm_module)
        
        # Create execution engine
        engine = llvm.create_mcjit_compiler(llvm_module, self.target_machine)
        
        # Finalize the execution engine
        engine.finalize_object()
        
        # Get function pointer
        func_ptr = engine.get_function_address(f"execute_{graph.name}")
        
        # Create a Python callable
        def execute(*args):
            """Execute the compiled function with numpy array inputs"""
            if len(args) != len(graph.inputs):
                raise ValueError(f"Expected {len(graph.inputs)} inputs, got {len(args)}")
                
            # Prepare inputs
            input_ptrs = []
            for i, arg in enumerate(args):
                if not isinstance(arg, np.ndarray):
                    arg = np.array(arg, dtype=graph.inputs[i].dtype.to_numpy())
                if arg.dtype != graph.inputs[i].dtype.to_numpy():
                    arg = arg.astype(graph.inputs[i].dtype.to_numpy())
                    
                # Make sure array is contiguous in memory
                if not arg.flags.c_contiguous:
                    arg = np.ascontiguousarray(arg)
                    
                input_ptrs.append(arg.ctypes.data_as(ctypes.c_void_p))
                
            # Prepare outputs
            outputs = []
            output_ptrs = []
            for output_node in graph.outputs:
                output = np.zeros(output_node.shape, dtype=output_node.dtype.to_numpy())
                
                # Make sure array is contiguous in memory -> the function expects to recieve a pointer in std C order
                if not output.flags.c_contiguous:
                    output = np.ascontiguousarray(output)
                    
                outputs.append(output)
                output_ptrs.append(output.ctypes.data_as(ctypes.c_void_p))
                
            # Create function type for ctypes
            argtypes = [ctypes.c_void_p] * (len(input_ptrs) + len(output_ptrs))
            cfunc = ctypes.CFUNCTYPE(None, *argtypes)(func_ptr)
            
            # Call function
            cfunc(*(input_ptrs + output_ptrs))
            
            # Return outputs
            if len(outputs) == 1:
                return outputs[0]
            return tuple(outputs)
            
        # Save a reference to the execution engine to prevent garbage collection
        execute._engine = engine
        execute._module = llvm_module
        
        return execute


class JITCompileOp:
    """
    A wrapper for JIT-compiled operations
    
    This class can be used to wrap a subgraph and compile it for efficient execution
    """
    def __init__(self, graph: IRGraph):
        self.graph = graph
        self.compiled_func = None
        
    def compile(self):
        """Compile the graph if LLVM is available"""
        if LLVM_AVAILABLE == True:  
            compiler = LLVMCompiler()
            self.compiled_func = compiler.compile(self.graph)
            
    def __call__(self, *args):
        """Execute the compiled function or fall back to interpreter"""
        if self.compiled_func is not None:
            return self.compiled_func(*args)
            
        # Fall back to interpreter if not compiled
        raise NotImplementedError("Graph interpreter not implemented yet")
