import os 
import ctypes
import numpy as np 
from typing import Any

try:
    import clang.cindex as cl 
    CLANG_AVAILABLE = True
    print(CLANG_AVAILABLE)
except ImportError:
    CLANG_AVAILABLE = False
    print("clang not avaibable")

from julia.core.ir import IRGraph, DataType

# TODO the actual C conversion.. actually make it compile 

class ClangCompiler:
    """
    Julia IR Graph -> C code
    """

    def __init__(self):
        if not CLANG_AVAILABLE:
            raise ImportError("clang bindings required")
        try:
            cl.Config.set_library_file('/usr/lib/llvm-10/lib/libclang.so') # fix this 
            self.index = cl.Index.create()
        except Exception as e:
            raise RuntimeError(f'Failed to init clang: {e}')

        self.temp_dir = os.path.join(os.getcwd(), "julia_clang_temp")
        os.makedirs(self.temp_dir, exist_ok=True)
    
    def _convert_type(self, dtype: DataType) -> str:
        if dtype == DataType.FLOAT32:
            return "float"
        elif dtype == DataType.FLOAT64:
            return "double"
        elif dtype == DataType.INT32:
            return "int32_t"
        elif dtype == DataType.INT64:
            return "int64_t"
        elif dtype == DataType.BOOL:
            return "bool"
        else:
            raise ValueError(f"Unsupported data type: {dtype}")
    
    def _generate_c_code(self, graph: IRGraph) -> str:
        """Generate C code from IR graph"""
        # Include necessary headers
        c_code = """
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>

"""
        func_signature = f"void execute_{graph.name}("
        
        for i, node in enumerate(graph.inputs):
            c_type = self._convert_type(node.dtype)
            if not node.shape or len(node.shape) == 0:
                func_signature += f"{c_type}* input{i}, "
            else:
                func_signature += f"{c_type}* input{i}, "  # Pointer to data
                
        for i, node in enumerate(graph.outputs):
            c_type = self._convert_type(node.dtype)
            if not node.shape or len(node.shape) == 0:
                func_signature += f"{c_type}* output{i}"
            else:
                func_signature += f"{c_type}* output{i}"  # Pointer to data
            
            if i < len(graph.outputs) - 1:
                func_signature += ", "
        
        func_signature += ")"
        c_code += func_signature + " {\n"
        
        variable_map = {}  # Maps node ID to C variable name
        
        # Map input nodes to C variables
        for i, node in enumerate(graph.inputs):
            variable_map[node.id] = f"input{i}"
        
        # Process nodes in topological order
        for node in graph.topological_sort():
            if node.id in variable_map:  # Skip inputs
                continue
            
            # Generate unique variable name
            var_name = f"var_{node.name.replace('-', '_')}"
            
            # Generate code based on operation type
            if node.op_type == "Add":
                input1 = variable_map[node.inputs[0].id]
                input2 = variable_map[node.inputs[1].id]
                
                if not node.shape or len(node.shape) <= 1:
                    # Scalar or vector addition
                    size = 1 if not node.shape else node.shape[0]
                    c_type = self._convert_type(node.dtype)
                    
                    c_code += f"    {c_type}* {var_name} = ({c_type}*)malloc(sizeof({c_type}) * {size});\n"
                    c_code += f"    for (int i = 0; i < {size}; i++) {{\n"
                    c_code += f"        {var_name}[i] = {input1}[i] + {input2}[i];\n"
                    c_code += "    }\n"
                else:
                    # Multi-dimensional array addition
                    total_size = np.prod(node.shape)
                    c_type = self._convert_type(node.dtype)
                    
                    c_code += f"    {c_type}* {var_name} = ({c_type}*)malloc(sizeof({c_type}) * {total_size});\n"
                    c_code += f"    for (int i = 0; i < {total_size}; i++) {{\n"
                    c_code += f"        {var_name}[i] = {input1}[i] + {input2}[i];\n"
                    c_code += "    }\n"
            
            elif node.op_type == "Mul":
                input1 = variable_map[node.inputs[0].id]
                input2 = variable_map[node.inputs[1].id]
                
                if not node.shape or len(node.shape) <= 1:
                    # Scalar or vector multiplication
                    size = 1 if not node.shape else node.shape[0]
                    c_type = self._convert_type(node.dtype)
                    
                    c_code += f"    {c_type}* {var_name} = ({c_type}*)malloc(sizeof({c_type}) * {size});\n"
                    c_code += f"    for (int i = 0; i < {size}; i++) {{\n"
                    c_code += f"        {var_name}[i] = {input1}[i] * {input2}[i];\n"
                    c_code += "    }\n"
                else:
                    # Multi-dimensional array multiplication
                    total_size = np.prod(node.shape)
                    c_type = self._convert_type(node.dtype)
                    
                    c_code += f"    {c_type}* {var_name} = ({c_type}*)malloc(sizeof({c_type}) * {total_size});\n"
                    c_code += f"    for (int i = 0; i < {total_size}; i++) {{\n"
                    c_code += f"        {var_name}[i] = {input1}[i] * {input2}[i];\n"
                    c_code += "    }\n"
            
            elif node.op_type == "MatMul":
                input1 = variable_map[node.inputs[0].id]
                input2 = variable_map[node.inputs[1].id]
                
                if len(node.inputs[0].shape) == 2 and len(node.inputs[1].shape) == 2:
                    # Matrix-matrix multiplication
                    m, k = node.inputs[0].shape
                    k2, n = node.inputs[1].shape
                    
                    if k != k2:
                        raise ValueError(f"Incompatible dimensions for MatMul: {node.inputs[0].shape} and {node.inputs[1].shape}")
                    
                    c_type = self._convert_type(node.dtype)
                    c_code += f"    {c_type}* {var_name} = ({c_type}*)malloc(sizeof({c_type}) * {m * n});\n"
                    c_code += f"    memset({var_name}, 0, sizeof({c_type}) * {m * n});\n"
                    c_code += f"    for (int i = 0; i < {m}; i++) {{\n"
                    c_code += f"        for (int j = 0; j < {n}; j++) {{\n"
                    c_code += f"            for (int k = 0; k < {k}; k++) {{\n"
                    c_code += f"                {var_name}[i * {n} + j] += {input1}[i * {k} + k] * {input2}[k * {n} + j];\n"
                    c_code += "            }\n"
                    c_code += "        }\n"
                    c_code += "    }\n"
                elif len(node.inputs[0].shape) == 1 and len(node.inputs[1].shape) == 1:
                    # Vector-vector dot product
                    k = node.inputs[0].shape[0]
                    c_type = self._convert_type(node.dtype)
                    
                    c_code += f"    {c_type}* {var_name} = ({c_type}*)malloc(sizeof({c_type}));\n"
                    c_code += f"    {var_name}[0] = 0;\n"
                    c_code += f"    for (int i = 0; i < {k}; i++) {{\n"
                    c_code += f"        {var_name}[0] += {input1}[i] * {input2}[i];\n"
                    c_code += "    }\n"
                else:
                    # Add other matrix multiplication cases as needed
                    raise NotImplementedError(f"Unsupported shapes for MatMul: {node.inputs[0].shape} and {node.inputs[1].shape}")
            
            elif node.op_type == "ReLU":
                input1 = variable_map[node.inputs[0].id]
                
                if not node.shape or len(node.shape) <= 1:
                    # Scalar or vector ReLU
                    size = 1 if not node.shape else node.shape[0]
                    c_type = self._convert_type(node.dtype)
                    
                    c_code += f"    {c_type}* {var_name} = ({c_type}*)malloc(sizeof({c_type}) * {size});\n"
                    c_code += f"    for (int i = 0; i < {size}; i++) {{\n"
                    c_code += f"        {var_name}[i] = {input1}[i] > 0 ? {input1}[i] : 0;\n"
                    c_code += "    }\n"
                else:
                    # Multi-dimensional array ReLU
                    total_size = np.prod(node.shape)
                    c_type = self._convert_type(node.dtype)
                    
                    c_code += f"    {c_type}* {var_name} = ({c_type}*)malloc(sizeof({c_type}) * {total_size});\n"
                    c_code += f"    for (int i = 0; i < {total_size}; i++) {{\n"
                    c_code += f"        {var_name}[i] = {input1}[i] > 0 ? {input1}[i] : 0;\n"
                    c_code += "    }\n"
            
            elif node.op_type == "Sigmoid":
                input1 = variable_map[node.inputs[0].id]
                
                if not node.shape or len(node.shape) <= 1:
                    # Scalar or vector Sigmoid
                    size = 1 if not node.shape else node.shape[0]
                    c_type = self._convert_type(node.dtype)
                    
                    c_code += f"    {c_type}* {var_name} = ({c_type}*)malloc(sizeof({c_type}) * {size});\n"
                    c_code += f"    for (int i = 0; i < {size}; i++) {{\n"
                    c_code += f"        {var_name}[i] = 1.0 / (1.0 + exp(-{input1}[i]));\n"
                    c_code += "    }\n"
                else:
                    # Multi-dimensional array Sigmoid
                    total_size = np.prod(node.shape)
                    c_type = self._convert_type(node.dtype)
                    
                    c_code += f"    {c_type}* {var_name} = ({c_type}*)malloc(sizeof({c_type}) * {total_size});\n"
                    c_code += f"    for (int i = 0; i < {total_size}; i++) {{\n"
                    c_code += f"        {var_name}[i] = 1.0 / (1.0 + exp(-{input1}[i]));\n"
                    c_code += "    }\n"
                    
            
            variable_map[node.id] = var_name
        
        for i, output_node in enumerate(graph.outputs):
            result_var = variable_map[output_node.id]
            
            if not output_node.shape or len(output_node.shape) == 0:
                c_code += f"    *output{i} = {result_var}[0];\n"
            else:
                total_size = np.prod(output_node.shape)
                c_code += f"    memcpy(output{i}, {result_var}, sizeof({self._convert_type(output_node.dtype)}) * {total_size});\n"
        
        # Free allocated memory
        for node_id, var_name in variable_map.items():
            if node_id not in [n.id for n in graph.inputs] and var_name not in [f"output{i}" for i in range(len(graph.outputs))]:
                c_code += f"    free({var_name});\n"
        
        c_code += "}\n"
        return c_code
    
    def compile(self, graph: IRGraph) -> Any:
        """
        Compile graph to executable function
        
        Returns a callable function that takes numpy arrays as inputs and returns numpy array outputs
        """
        graph.infer_shapes()
        
        c_code = self._generate_c_code(graph)
        
        c_file_path = os.path.join(self.temp_dir, f"{graph.name}.c")
        with open(c_file_path, "w") as f:
            f.write(c_code)
        
        so_file_path = os.path.join(self.temp_dir, f"lib{graph.name}.so")
        compile_cmd = f"gcc -shared -fPIC -O3 -o {so_file_path} {c_file_path}"
        
        import subprocess
        result = subprocess.run(compile_cmd, shell=True, check=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Failed to compile C code: {result.stderr}")
        
        lib = ctypes.CDLL(so_file_path)
        
        func = getattr(lib, f"execute_{graph.name}")
        
        argtypes = []
        for node in graph.inputs:
            if node.dtype == DataType.FLOAT32:
                argtypes.append(ctypes.POINTER(ctypes.c_float))
            elif node.dtype == DataType.FLOAT64:
                argtypes.append(ctypes.POINTER(ctypes.c_double))
            elif node.dtype == DataType.INT32:
                argtypes.append(ctypes.POINTER(ctypes.c_int32))
            elif node.dtype == DataType.INT64:
                argtypes.append(ctypes.POINTER(ctypes.c_int64))
            elif node.dtype == DataType.BOOL:
                argtypes.append(ctypes.POINTER(ctypes.c_bool))
        
        for node in graph.outputs:
            if node.dtype == DataType.FLOAT32:
                argtypes.append(ctypes.POINTER(ctypes.c_float))
            elif node.dtype == DataType.FLOAT64:
                argtypes.append(ctypes.POINTER(ctypes.c_double))
            elif node.dtype == DataType.INT32:
                argtypes.append(ctypes.POINTER(ctypes.c_int32))
            elif node.dtype == DataType.INT64:
                argtypes.append(ctypes.POINTER(ctypes.c_int64))
            elif node.dtype == DataType.BOOL:
                argtypes.append(ctypes.POINTER(ctypes.c_bool))
        
        func.argtypes = argtypes
        func.restype = None  # void return type
        
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
                    
                input_ptrs.append(arg.ctypes.data_as(argtypes[i]))
                
            # Prepare outputs
            outputs = []
            output_ptrs = []
            for i, output_node in enumerate(graph.outputs):
                output = np.zeros(output_node.shape, dtype=output_node.dtype.to_numpy())
                
                if not output.flags.c_contiguous:
                    output = np.ascontiguousarray(output)
                    
                outputs.append(output)
                output_ptrs.append(output.ctypes.data_as(argtypes[len(graph.inputs) + i]))
                
            func(*(input_ptrs + output_ptrs))
            
            if len(outputs) == 1:
                return outputs[0]
            return tuple(outputs)
            
        # Save a reference to the library to prevent garbage collection
        execute._lib = lib
        
        return execute

class JitCompileOpClang:
    """
    Wrapper for JIT-compiled ops in CLANG 
    Pretty much the same as the LLVM
    """

    def __init__(self, graph: IRGraph):
        self.graph = graph
        self.compiled_func = None

    def compile(self):
        """Compile if CLANG"""
        if CLANG_AVAILABLE:
            compiler = ClangCompiler()
            self.compiled_func = compiler.compile(self.graph)

    def __call__(self, *args):
        """Exectute the compiled function or use the interpreter"""
        if self.compiled_func is not None:
            return self.compiled_func(*args)

        raise NotImplementedError("Graph interpreter not done")

