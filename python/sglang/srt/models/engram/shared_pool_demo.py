#!/usr/bin/env python3
"""Demo: two servers share one distributed memory pool via Mooncake Store.

Run on two machines (or two processes with different hostnames/ports):
- Secondary server writes a key into the shared pool.
- Primary server waits and reads the key from the same pool.
"""

import argparse
import time
from typing import Optional

from mooncake.store import MooncakeDistributedStore


def create_store(
    local_hostname: str,
    metadata_server: str,
    global_segment_size: int,
    local_buffer_size: int,
    protocol: str,
    device_name: str,
    master_server_address: str,
) -> MooncakeDistributedStore:
    store = MooncakeDistributedStore()
    ret = store.setup(
        local_hostname,
        metadata_server,
        global_segment_size,
        local_buffer_size,
        protocol,
        device_name,
        master_server_address,
    )
    if ret != 0:
        raise RuntimeError(f"Store setup failed with code {ret}")
    return store


def wait_for_key(store: MooncakeDistributedStore, key: str, timeout_s: float) -> bool:
    start = time.time()
    while time.time() - start < timeout_s:
        if store.is_exist(key):
            return True
        time.sleep(0.2)
    return False


def run_writer(store: MooncakeDistributedStore, key: str, value: bytes) -> None:
    ret = store.put(key, value)
    if ret != 0:
        raise RuntimeError(f"PUT failed with code {ret}")
    print(f"[writer] put key={key} size={len(value)}")


def run_reader(store: MooncakeDistributedStore, key: str, timeout_s: float) -> None:
    ok = wait_for_key(store, key, timeout_s)
    if not ok:
        raise TimeoutError(f"Timeout waiting for key={key}")
    data = store.get(key)
    print(f"[reader] got key={key} size={len(data)} value={data!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Shared pool demo with Mooncake Store")
    parser.add_argument("--role", choices=["primary", "secondary"], required=True)
    parser.add_argument("--local-hostname", default="localhost")
    parser.add_argument("--metadata-server", default="http://127.0.0.1:8080/metadata")
    parser.add_argument("--master-server", default="127.0.0.1:50051")
    parser.add_argument("--global-segment-size", type=int, default=512 * 1024 * 1024)
    parser.add_argument("--local-buffer-size", type=int, default=128 * 1024 * 1024)
    parser.add_argument("--protocol", default="tcp")
    # parser.add_argument("--protocol", default="rdma")
    parser.add_argument("--device-name", default="")
    parser.add_argument("--timeout-s", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    store = create_store(
        local_hostname=args.local_hostname,
        metadata_server=args.metadata_server,
        global_segment_size=args.global_segment_size,
        local_buffer_size=args.local_buffer_size,
        protocol=args.protocol,
        device_name=args.device_name,
        master_server_address=args.master_server,
    )

    key = "demo_key"
    value = "hello from secondary"

    try:
        if args.role == "secondary":
            run_writer(store, key, value.encode())
        else:
            run_reader(store, key, args.timeout_s)
    finally:
        store.close()


if __name__ == "__main__":
    main()
