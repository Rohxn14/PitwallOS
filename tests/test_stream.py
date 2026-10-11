"""
The replay's stream server sends fixed tables (lap times) once per client, not in every message.

Before, every message re-sent a ~600 KB lap-time table, which held the replay to ~8 updates a
second. These tests use a real server on a free local port.
"""

import json
import socket
import time

import pytest

stream = pytest.importorskip("src.services.stream")


def read_messages(sock, n, timeout=5.0):
    sock.settimeout(timeout)
    buf, out = b"", []
    while len(out) < n:
        buf += sock.recv(1 << 16)
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            out.append(json.loads(line))
    return out


def connect(server):
    sock = socket.create_connection(("localhost", server.server_socket.getsockname()[1]))
    deadline = time.time() + 5
    while len(server.clients) < 1 + connect.count and time.time() < deadline:
        time.sleep(0.01)
    connect.count += 1
    return sock


@pytest.fixture
def server():
    connect.count = 0
    srv = stream.TelemetryStreamServer(port=0)
    srv.start()
    yield srv
    srv.stop()


def test_static_fields_arrive_once_then_messages_stay_small(server):
    client = connect(server)
    server.set_static({"lap_times": {"HAM": [1, 2, 3]}})
    server.broadcast({"frame_index": 1})
    server.broadcast({"frame_index": 2})
    first, second = read_messages(client, 2)
    assert first == {"frame_index": 1, "lap_times": {"HAM": [1, 2, 3]}}
    assert second == {"frame_index": 2}
    client.close()


def test_a_client_connecting_later_still_gets_the_static_fields(server):
    early = connect(server)
    server.set_static({"lap_times": {"VER": [9]}})
    server.broadcast({"frame_index": 1})
    read_messages(early, 1)

    late = connect(server)
    server.broadcast({"frame_index": 2})
    assert read_messages(early, 1) == [{"frame_index": 2}]
    assert read_messages(late, 1) == [{"frame_index": 2, "lap_times": {"VER": [9]}}]
    early.close()
    late.close()
