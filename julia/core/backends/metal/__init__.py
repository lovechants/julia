try:
    from julia.core.backends.metal.compiler import MetalCompiler, MetalKernelRunner, METAL_AVAILABLE
    from julia.core.backends.metal.device_management import MetalDeviceManager
except ImportError:
    METAL_AVAILABLE = False
