import os

# torch (open_clip) and faiss-cpu each bundle their own OpenMP runtime; importing
# both in one process aborts on macOS with "OMP: Error #15 ... libomp already
# initialized". Allowing the duplicate runtime is the standard, safe-in-practice
# workaround for a test process that loads both. Set before any test module imports
# torch or faiss.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import shutil
import socket
import subprocess
import time
import urllib.request

import pytest


def _ollama_up(base: str = "http://localhost:11434") -> bool:
    """True when a local Ollama HTTP endpoint answers within ~1s."""
    try:
        urllib.request.urlopen(base, timeout=1)
        return True
    except Exception:
        return False


def pytest_collection_modifyitems(config, items):
    """Skip @pytest.mark.live items unless a local Ollama is reachable."""
    if _ollama_up():
        return
    skip_live = pytest.mark.skip(reason="local Ollama not reachable (live tier)")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)


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
