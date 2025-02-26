import iio

# List all contexts (connected devices)
contexts = iio.scan_contexts()
print("Available PlutoSDR Devices:")
if not contexts:
    print("No devices found.")
else:
    for uri, description in contexts.items():
        print(f"URI: {uri} - Description: {description}")
