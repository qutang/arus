import logging
from pymetawear.discover import discover_devices


class MetaWearScanner():
    def get_nearby_devices(self, max_devices=None):
        metawears = set()
        retries = 0
        max_devices = 100 if max_devices is None else max_devices
        while retries < 3 or len(metawears) < max_devices:
            logging.info('Scanning metawear devices nearby...')
            try:
                retries += 1
                candidates = discover_devices(timeout=1)
                metawears |= set(
                    map(lambda d: d[0],
                        filter(lambda d: d[1] == 'MetaWear', candidates)))
            except ValueError as e:
                logging.error(str(e))
                continue
        return list(metawears)
