# The maintained suite must not contact remote model providers or download weights.
import socket
import pytest


@pytest.fixture(autouse=True)
def offline_tests(monkeypatch):
    def blocked(*args, **kwargs):
        raise RuntimeError("Network disabled in offline tests")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
