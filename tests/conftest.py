#!/usr/bin/env python3
"""
Pytest configuration for Driver Fatigue Monitor tests.
"""

import os
import pytest
import requests
import time
from typing import Generator


@pytest.fixture(scope="session")
def api_url() -> str:
    """Base URL for API tests."""
    return os.getenv("API_URL", "http://localhost:8000")


@pytest.fixture(scope="session")
def ws_url() -> str:
    """WebSocket URL for integration tests."""
    return os.getenv("WS_URL", "ws://localhost:8000")


@pytest.fixture(scope="session")
def api_client(api_url: str) -> requests.Session:
    """HTTP session for API tests."""
    session = requests.Session()
    session.timeout = 10
    return session


@pytest.fixture(scope="session", autouse=True)
def ensure_backend_ready(api_url: str, api_client: requests.Session) -> Generator[None, None, None]:
    """Ensure backend is ready before running tests."""
    max_retries = 30
    for i in range(max_retries):
        try:
            r = api_client.get(f"{api_url}/health", timeout=3)
            if r.status_code == 200 and r.json().get("status") == "healthy":
                print(f"\n✅ Backend ready at {api_url}")
                yield
                return
        except Exception:
            pass
        time.sleep(1)
    pytest.fail(f"Backend not ready at {api_url} after {max_retries}s")


@pytest.fixture
def test_frame_b64() -> str:
    """Create a base64 test frame."""
    import cv2
    import numpy as np
    import base64

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame, (200, 150), (440, 350), (200, 200, 200), -1)
    cv2.circle(frame, (280, 220), 30, (100, 100, 100), -1)
    cv2.circle(frame, (360, 220), 30, (100, 100, 100), -1)
    cv2.ellipse(frame, (320, 300), (40, 20), 0, 0, 180, (100, 100, 100), -1)
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf).decode("utf-8")


@pytest.fixture
def session_id(api_url: str, api_client: requests.Session) -> Generator[str, None, None]:
    """Create and cleanup a test session."""
    r = api_client.post(f"{api_url}/session/start", json={"source": "pytest"})
    assert r.status_code == 200
    sid = r.json()["session_id"]
    yield sid
    # Cleanup
    api_client.post(f"{api_url}/session/{sid}/stop")


# Pytest markers
def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "api: marks tests as API tests")
    config.addinivalue_line("markers", "websocket: marks tests as WebSocket tests")