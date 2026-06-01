import shutil
import socket
import subprocess
import time

import pytest


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture()
def surreal_url(tmp_path):
    if shutil.which("surreal") is None:
        pytest.skip("surreal binary not installed")
    port = _free_port()
    # SurrealDB 3.x dropped the `file://` storage scheme; use `rocksdb:`.
    proc = subprocess.Popen(
        ["surreal", "start", "--user", "root", "--pass", "root",
         f"rocksdb:{tmp_path}/db", "--bind", f"127.0.0.1:{port}"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    for _ in range(50):
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
            break
        except OSError:
            time.sleep(0.2)
    else:
        proc.terminate()
        pytest.fail("SurrealDB did not start")
    yield f"ws://127.0.0.1:{port}/rpc"
    proc.terminate()
    proc.wait(timeout=10)
