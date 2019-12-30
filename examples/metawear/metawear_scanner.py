"""
Demonstration of the usage of MetaWearScanner
====================================================
"""
from arus.plugins.metawear.scanner import MetaWearScanner


if __name__ == "__main__":
    scanner = MetaWearScanner()
    print(scanner.get_nearby_devices(max_devices=2))