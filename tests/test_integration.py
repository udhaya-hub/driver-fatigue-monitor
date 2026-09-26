#!/usr/bin/env python3
"""
Integration Tests for Driver Fatigue Monitor
Tests full stack: FastAPI + WebSocket + Streamlit-compatible flows.

Usage:
    python tests/test_integration.py
"""

import asyncio
import base64
import json
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import requests
import websockets


class IntegrationTester:
    """Full-stack integration tests."""

    def __init__(self, api_url: str = "http://localhost:8000", ws_url: str = "ws://localhost:8000"):
        self.api_url = api_url.rstrip("/")
        self.ws_url = ws_url.rstrip("/")
        self.session = requests.Session()
        self.session_id: Optional[str] = None

    def log(self, msg: str, level: str = "INFO"):
        prefix = {"INFO": "ℹ️", "PASS": "✅", "FAIL": "❌", "WARN": "⚠️"}.get(level, "•")
        print(f"  {prefix} {msg}")

    def create_test_frame_b64(self) -> str:
        """Create a test frame with face-like features."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(frame, (200, 150), (440, 350), (200, 200, 200), -1)
        cv2.circle(frame, (280, 220), 30, (100, 100, 100), -1)
        cv2.circle(frame, (360, 220), 30, (100, 100, 100), -1)
        cv2.ellipse(frame, (320, 300), (40, 20), 0, 0, 180, (100, 100, 100), -1)
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return base64.b64encode(buf).decode("utf-8")

    def test_rest_api_flow(self) -> bool:
        """Test complete REST API flow."""
        self.log("Testing REST API flow...")

        # 1. Health
        r = self.session.get(f"{self.api_url}/health", timeout=5)
        if r.status_code != 200 or r.json().get("status") != "healthy":
            self.log("Health check failed", "FAIL")
            return False
        self.log("Backend healthy")

        # 2. Start session
        r = self.session.post(f"{self.api_url}/session/start", json={"source": "integration_test"}, timeout=5)
        if r.status_code != 200:
            self.log("Session start failed", "FAIL")
            return False
        self.session_id = r.json()["session_id"]
        self.log(f"Session started: {self.session_id[:8]}...")

        # 3. Send frames via REST
        frame_b64 = self.create_test_frame_b64()
        results = []
        for i in range(5):
            r = self.session.post(
                f"{self.api_url}/inference/frame",
                json={"frame_data": frame_b64},
                timeout=5
            )
            if r.status_code == 200:
                data = r.json()
                results.append(data)
                self.log(f"Frame {i+1}: state={data['fatigue_state']}, conf={data['confidence']:.2f}")
            else:
                self.log(f"Frame {i+1} failed: {r.status_code}", "FAIL")
                return False
            time.sleep(0.1)

        # 4. Verify results
        states = [r["fatigue_state"] for r in results]
        self.log(f"States observed: {set(states)}")

        # 5. Stop session
        r = self.session.post(f"{self.api_url}/session/{self.session_id}/stop", timeout=5)
        if r.status_code != 200:
            self.log("Session stop failed", "FAIL")
            return False
        self.log(f"Session stopped, frames: {r.json().get('frames_processed')}")

        return True

    async def test_websocket_flow(self) -> bool:
        """Test WebSocket real-time flow."""
        if not self.session_id:
            self.log("No session for WebSocket test", "WARN")
            return False

        self.log("Testing WebSocket flow...")

        frame_b64 = self.create_test_frame_b64()
        ws_uri = f"{self.ws_url}/ws/{self.session_id}"

        try:
            async with websockets.connect(ws_uri, timeout=10) as ws:
                # Send a few frames
                for i in range(3):
                    msg = {
                        "type": "frame",
                        "data": frame_b64,
                        "timestamp": time.time(),
                        "physio": [0.5] * 32  # Dummy physio
                    }
                    await ws.send(json.dumps(msg))
                    await asyncio.sleep(0.1)

                # Wait for results
                results_received = 0
                start_time = time.time()
                while results_received < 3 and time.time() - start_time < 10:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=2)
                        data = json.loads(msg)
                        if data.get("type") == "result":
                            results_received += 1
                            self.log(f"WS Result {results_received}: {data.get('fatigue_state')}")
                    except asyncio.TimeoutError:
                        break

                self.log(f"Received {results_received} WebSocket results")
                return results_received > 0

        except Exception as e:
            self.log(f"WebSocket error: {e}", "FAIL")
            return False

    def test_error_resilience(self) -> bool:
        """Test error handling."""
        self.log("Testing error resilience...")

        # Invalid frame
        r = self.session.post(
            f"{self.api_url}/inference/frame",
            json={"frame_data": "not_base64"},
            timeout=5
        )
        if r.status_code not in (400, 500):
            self.log(f"Invalid frame should error, got {r.status_code}", "FAIL")
            return False
        self.log("Invalid frame rejected correctly")

        # Invalid session
        r = self.session.post(f"{self.api_url}/session/bad/stop", timeout=5)
        if r.status_code != 404:
            self.log(f"Invalid session should 404, got {r.status_code}", "FAIL")
            return False
        self.log("Invalid session rejected correctly")

        return True


async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--ws", default="ws://localhost:8000", help="WebSocket URL")
    args = parser.parse_args()

    print(f"\n{'='*50}")
    print("  Integration Test Suite")
    print(f"  API: {args.api}")
    print(f"  WS:  {args.ws}")
    print(f"{'='*50}\n")

    tester = IntegrationTester(args.api, args.ws)

    # Run REST tests
    rest_ok = tester.test_rest_api_flow()
    print()

    # Run WebSocket tests
    ws_ok = await tester.test_websocket_flow()
    print()

    # Run error tests
    err_ok = tester.test_error_resilience()
    print()

    # Summary
    print(f"{'='*50}")
    all_ok = rest_ok and ws_ok and err_ok
    if all_ok:
        print("  🎉 All integration tests passed!")
    else:
        print("  ⚠️  Some tests failed:")
        if not rest_ok: print("    - REST API flow")
        if not ws_ok: print("    - WebSocket flow")
        if not err_ok: print("    - Error resilience")
    print(f"{'='*50}\n")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    asyncio.run(main())