#!/usr/bin/env python3
"""
Automated API Test Suite for Driver Fatigue Monitor Backend
Tests all FastAPI endpoints, model inference, and session lifecycle.

Usage:
    python tests/test_api.py                    # Run all tests
    python tests/test_api.py --url http://localhost:8000  # Custom URL
    python tests/test_api.py --verbose          # Verbose output
"""

import argparse
import base64
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

import cv2
import numpy as np
import requests


class APITester:
    """Comprehensive API test client."""

    def __init__(self, base_url: str, timeout: int = 10):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session_id: Optional[str] = None
        self.test_results: Dict[str, Any] = {}

    def log(self, message: str, level: str = "INFO"):
        """Print formatted log message."""
        prefix = {"INFO": "[+]", "WARN": "[!]", "ERROR": "[x]", "PASS": "[PASS]", "FAIL": "[FAIL]"}.get(level, "[*]")
        print(f"  {prefix} {message}")

    def record(self, test_name: str, success: bool, details: Any = None):
        """Record test result."""
        self.test_results[test_name] = {"success": success, "details": details}
        self.log(f"{test_name}: {'PASS' if success else 'FAIL'}", "PASS" if success else "FAIL")

    def check_health(self) -> bool:
        """Test GET /health endpoint."""
        try:
            resp = self.session.get(f"{self.base_url}/health", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                healthy = data.get("status") == "healthy"
                model_loaded = data.get("model_loaded", False)
                device = data.get("device", "unknown")
                self.record("Health Check", healthy, 
                    f"status={data.get('status')}, model_loaded={model_loaded}, device={device}")
                return healthy
            self.record("Health Check", False, f"HTTP {resp.status_code}")
            return False
        except Exception as e:
            self.record("Health Check", False, str(e))
            return False

    def check_config(self) -> bool:
        """Test GET /config endpoint."""
        try:
            resp = self.session.get(f"{self.base_url}/config", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                has_model = "model" in data
                has_vision = "vision" in data
                has_inference = "inference" in data
                success = has_model and has_vision and has_inference
                self.record("Config Endpoint", success,
                    f"model={has_model}, vision={has_vision}, inference={has_inference}")
                return success
            self.record("Config Endpoint", False, f"HTTP {resp.status_code}")
            return False
        except Exception as e:
            self.record("Config Endpoint", False, str(e))
            return False

    def create_test_frame(self) -> str:
        """Create a dummy test frame as base64 JPEG."""
        # Create a simple test image with a face-like pattern
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Add some visual features
        cv2.rectangle(frame, (200, 150), (440, 350), (200, 200, 200), -1)  # Face area
        cv2.circle(frame, (280, 220), 30, (100, 100, 100), -1)  # Left eye
        cv2.circle(frame, (360, 220), 30, (100, 100, 100), -1)  # Right eye
        cv2.ellipse(frame, (320, 300), (40, 20), 0, 0, 180, (100, 100, 100), -1)  # Mouth
        
        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return base64.b64encode(buffer).decode("utf-8")

    def test_single_frame_inference(self) -> bool:
        """Test POST /inference/frame with a single frame."""
        if not self.session_id:
            self.log("Skipping frame inference - no active session", "WARN")
            return False

        try:
            frame_b64 = self.create_test_frame()
            resp = self.session.post(
                f"{self.base_url}/inference/frame",
                json={"frame_data": frame_b64},
                timeout=self.timeout
            )
            if resp.status_code == 200:
                data = resp.json()
                required_fields = ["frame_id", "fatigue_state", "confidence", 
                                 "eye_metrics", "mouth_metrics", "head_pose"]
                has_all = all(field in data for field in required_fields)
                fatigue_state = data.get("fatigue_state", "unknown")
                confidence = data.get("confidence", 0.0)
                self.record("Single Frame Inference", has_all,
                    f"state={fatigue_state}, confidence={confidence:.2f}, fields={len(data)}")
                return has_all
            self.record("Single Frame Inference", False, f"HTTP {resp.status_code}: {resp.text[:200]}")
            return False
        except Exception as e:
            self.record("Single Frame Inference", False, str(e))
            return False

    def test_sequence_inference(self) -> bool:
        """Test POST /inference/sequence with multiple frames."""
        if not self.session_id:
            self.log("Skipping sequence inference - no active session", "WARN")
            return False

        try:
            frames = [self.create_test_frame() for _ in range(5)]
            resp = self.session.post(
                f"{self.base_url}/inference/sequence",
                json={"frames_data": frames},
                timeout=self.timeout
            )
            if resp.status_code == 200:
                data = resp.json()
                is_list = isinstance(data, list) and len(data) > 0
                if is_list and isinstance(data[0], dict):
                    self.record("Sequence Inference", True,
                        f"returned {len(data)} results")
                    return True
                self.record("Sequence Inference", False, "Invalid response format")
                return False
            self.record("Sequence Inference", False, f"HTTP {resp.status_code}: {resp.text[:200]}")
            return False
        except Exception as e:
            self.record("Sequence Inference", False, str(e))
            return False

    def test_session_lifecycle(self) -> bool:
        """Test complete session start -> inference -> stop lifecycle."""
        # Start session
        try:
            resp = self.session.post(
                f"{self.base_url}/session/start",
                json={"source": "test"},
                timeout=self.timeout
            )
            if resp.status_code != 200:
                self.record("Session Start", False, f"HTTP {resp.status_code}")
                return False
            
            data = resp.json()
            self.session_id = data.get("session_id")
            if not self.session_id:
                self.record("Session Start", False, "No session_id returned")
                return False
            self.record("Session Start", True, f"session_id={self.session_id[:8]}...")
        except Exception as e:
            self.record("Session Start", False, str(e))
            return False

        # Test session status
        try:
            resp = self.session.get(
                f"{self.base_url}/session/{self.session_id}/status",
                timeout=self.timeout
            )
            if resp.status_code == 200:
                data = resp.json()
                self.record("Session Status", True, f"frames={data.get('frame_count', 0)}")
            else:
                self.record("Session Status", False, f"HTTP {resp.status_code}")
        except Exception as e:
            self.record("Session Status", False, str(e))

        # Run a few inferences
        for i in range(3):
            self.test_single_frame_inference()
            time.sleep(0.1)

        # Stop session
        try:
            resp = self.session.post(
                f"{self.base_url}/session/{self.session_id}/stop",
                timeout=self.timeout
            )
            if resp.status_code == 200:
                data = resp.json()
                frames = data.get("frames_processed", 0)
                self.record("Session Stop", True, f"frames_processed={frames}")
                return True
            self.record("Session Stop", False, f"HTTP {resp.status_code}")
            return False
        except Exception as e:
            self.record("Session Stop", False, str(e))
            return False

    def test_invalid_requests(self) -> bool:
        """Test error handling for invalid requests."""
        all_passed = True
        
        # Invalid frame data
        try:
            resp = self.session.post(
                f"{self.base_url}/inference/frame",
                json={"frame_data": "invalid_base64"},
                timeout=self.timeout
            )
            # Should return 400 or 500
            passed = resp.status_code in (400, 500)
            self.record("Invalid Frame Handling", passed, f"HTTP {resp.status_code}")
            all_passed &= passed
        except Exception as e:
            self.record("Invalid Frame Handling", False, str(e))
            all_passed = False

        # Invalid session stop
        try:
            resp = self.session.post(
                f"{self.base_url}/session/invalid-id/stop",
                timeout=self.timeout
            )
            passed = resp.status_code == 404
            self.record("Invalid Session Handling", passed, f"HTTP {resp.status_code}")
            all_passed &= passed
        except Exception as e:
            self.record("Invalid Session Handling", False, str(e))
            all_passed = False

        return all_passed

    def test_metrics_endpoint(self) -> bool:
        """Test Prometheus metrics endpoint."""
        try:
            resp = self.session.get(f"{self.base_url}/metrics", timeout=5)
            if resp.status_code == 200:
                content = resp.text
                has_metrics = any(key in content for key in [
                    "fatigue_inference_requests_total",
                    "fatigue_inference_latency_seconds",
                    "fatigue_active_websocket_connections"
                ])
                self.record("Metrics Endpoint", has_metrics, 
                    "Prometheus metrics exposed" if has_metrics else "No custom metrics found")
                return has_metrics
            self.record("Metrics Endpoint", False, f"HTTP {resp.status_code}")
            return False
        except Exception as e:
            self.record("Metrics Endpoint", False, str(e))
            return False

    def run_all_tests(self) -> Dict[str, Any]:
        """Run complete test suite."""
        print(f"\n{'='*60}")
        print(f"  Driver Fatigue Monitor - API Test Suite")
        print(f"  Target: {self.base_url}")
        print(f"{'='*60}\n")

        # Core health & config
        self.log("Testing Core Endpoints...", "INFO")
        self.check_health()
        self.check_config()
        self.test_metrics_endpoint()

        # Session lifecycle (includes inference tests)
        self.log("\nTesting Session Lifecycle...", "INFO")
        self.test_session_lifecycle()

        # Error handling
        self.log("\nTesting Error Handling...", "INFO")
        self.test_invalid_requests()

        # Summary
        print(f"\n{'='*60}")
        passed = sum(1 for r in self.test_results.values() if r["success"])
        total = len(self.test_results)
        print(f"  Results: {passed}/{total} tests passed")
        
        if passed == total:
            print(f"  [OK] All tests passed!")
        else:
            print(f"  [WARN] {total - passed} test(s) failed:")
            for name, result in self.test_results.items():
                if not result["success"]:
                    print(f"     - {name}: {result['details']}")
        print(f"{'='*60}\n")

        return self.test_results


def main():
    parser = argparse.ArgumentParser(description="Driver Fatigue Monitor API Tests")
    parser.add_argument("--url", default="http://localhost:8000", 
                        help="Base URL of FastAPI server")
    parser.add_argument("--timeout", type=int, default=10, help="Request timeout")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    tester = APITester(args.url, timeout=args.timeout)
    results = tester.run_all_tests()

    # Exit with error code if any tests failed
    failed = sum(1 for r in results.values() if not r["success"])
    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()