try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


class TritonDeviceManager:
    pass


class TritonTensorManager:
    pass


class TritonOperations:
    pass


class TritonCompiler:
    """Triton -> Julia IR graphs for native tensor acceleration"""

    pass


class JITCompileOpTriton:
    pass
