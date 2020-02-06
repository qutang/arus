import os


def is_large_file(filepath, threshold=20):
    size_in_bytes = os.path.getsize(filepath)
    size_in_mb = size_in_bytes / 1024 / 1024
    if filepath.endswith('gz'):
        threshold = int(threshold / 7)
    if size_in_mb > threshold:
        return True
    else:
        return False
