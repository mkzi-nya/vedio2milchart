"""Run the local upload flow against a supplied gameplay recording."""
import argparse
import http.client
import json
import time
from pathlib import Path
from urllib.parse import urlparse


def request(connection, method, path, body=None, headers=None):
    connection.request(method, path, body=body, headers=headers or {})
    response = connection.getresponse()
    data = response.read()
    if response.status >= 400:
        raise RuntimeError(f"{method} {path}: HTTP {response.status}: {data[:500]!r}")
    return response.status, data


def main():
    parser = argparse.ArgumentParser(description="Test the running upload server with a gameplay recording.")
    parser.add_argument("video", type=Path, help="complete gameplay recording")
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--timeout", type=int, default=1200, help="conversion timeout in seconds")
    args = parser.parse_args()
    video = args.video.expanduser().resolve()
    if not video.is_file():
        parser.error(f"video not found: {video}")

    url = urlparse(args.url)
    connection_type = http.client.HTTPSConnection if url.scheme == "https" else http.client.HTTPConnection
    connection = connection_type(url.hostname, url.port, timeout=60)
    size = video.stat().st_size
    connection.putrequest("POST", "/api/upload")
    connection.putheader("Content-Length", str(size))
    connection.putheader("Content-Type", "video/mp4")
    connection.putheader("X-Title", video.stem[:120])
    connection.endheaders()
    with video.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            connection.send(chunk)
    response = connection.getresponse()
    payload = response.read()
    if response.status >= 400:
        raise RuntimeError(f"Upload failed: HTTP {response.status}: {payload[:500]!r}")
    job = json.loads(payload)
    job_id = job["id"]

    status, ranged = request(connection, "GET", f"/jobs/{job_id}/source.mp4", headers={"Range": "bytes=0-99"})
    assert status == 206 and len(ranged) == min(100, size), "video range response is invalid"
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        _, data = request(connection, "GET", f"/api/job/{job_id}")
        state = json.loads(data)
        if state["state"] == "done":
            break
        if state["state"] == "failed":
            raise AssertionError(state)
        time.sleep(1)
    else:
        raise TimeoutError(f"conversion exceeded {args.timeout} seconds")

    _, data = request(connection, "GET", f"{state['base']}/events.json")
    events = json.loads(data)
    assert state["report"].get("automatic") is True
    assert state["report"].get("nominalPixelsPerSecondAt1280", 0) > 0
    assert len(events) > 100
    _, chart = request(
        connection,
        "POST",
        "/api/export",
        body=json.dumps({"events": events[:5]}),
        headers={"Content-Type": "application/json"},
    )
    assert chart.count(b"let n=m.note") == 5
    print(f"PASS upload, range streaming, automatic conversion and chart export: {len(events)} events, job {job_id}")
    connection.close()


if __name__ == "__main__":
    main()
