import platform

try:
    if platform.system() == "Darwin":
        import objc
        from Foundation import NSBundle
        from Metal import (
            MTLCreateSystemDefaultDevice,
            MTLCommandQueue,
            MTLBuffer,
            MTLComputePipelineState,
            MTLLibrary,
            MTLFunction,
            MTLCommandBuffer,
            MTLComputeCommandEncoder,
            MTLResourceStorageModeShared,
            MTLResourceStorageModePrivate,
        )

        METAL_AVAILABLE = True
    else:
        METAL_AVAILABLE = False
except ImportError:
    METAL_AVAILABLE = False


class MetalBuffer:
    pass


class MetalDeviceManager:
    pass
