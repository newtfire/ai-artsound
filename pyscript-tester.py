import pyscript
from pyscript.media import list_devices

async def show_available_devices():
    devices = await list_devices()
    for device in devices:
        print(f"Device: {device.label}, Type: {device.kind}, ID: {device.id}")

# List all available devices
show_available_devices()