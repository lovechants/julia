import platform
import sys

print(f"Platform: {platform.system()}")
print(f"macOS version: {platform.mac_ver()[0]}")

try:
    from Metal import MTLCreateSystemDefaultDevice
    device = MTLCreateSystemDefaultDevice()
    if device:
        print(f"Metal device: {device.name()}")
        print("Metal is available")
    else:
        print("No Metal device found")
except ImportError as e:
    print(f"Metal not available: {e}")
