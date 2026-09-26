#!/usr/bin/env python3
"""
Pytest test suite for Driver Fatigue Monitor API.
Run with: pytest tests/test_pytest_api.py -v
"""

import base64
import pytest
import requests


class TestHealthAndConfig:
    """Core endpoint tests."""

    def test_health_endpoint(self, api_client: requests.Session, api_url: str):
        """GET /health returns healthy status."""
        r = api_client.get(f"{api_url}/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert "device" in data

    def test_config_endpoint(self, api_client: requests.Session, api_url: str):
        """GET /config returns full configuration."""
        r = api_client.get(f"{api_url}/config")
        assert r.status_code == 200
        data = r.json()
        assert "model" in data
        assert "vision" in data
        assert "inference" in data
        assert "api" in data

    def test_metrics_endpoint(self, api_client: requests.Session, api_url: str):
        """GET /metrics returns Prometheus metrics."""
        r = api_client.get(f"{api_url}/metrics")
        assert r.status_code == 200
        assert "fatigue_inference_requests_total" in r.text
        assert "fatigue_inference_latency_seconds" in r.text


class TestSessionLifecycle:
    """Session management tests."""

    def test_start_session(self, api_client: requests.Session, api_url: str):
        """POST /session/start creates a session."""
        r = api_client.post(f"{api_url}/session/start", json={"source": "pytest"})
        assert r.status_code == 200
        data = r.json()
        assert "session_id" in data
        assert len(data["session_id"]) > 0

    def test_session_status(self, api_client: requests.Session, api_url: str, session_id: str):
        """GET /session/{id}/status returns session info."""
        r = api_client.get(f"{api_url}/session/{session_id}/status")
        assert r.status_code == 200
        data = r.json()
        assert data["session_id"] == session_id
        assert "frame_count" in data
        assert "duration_seconds" in data

    def test_stop_session(self, api_client: requests.Session, api_url: str, session_id: str):
        """POST /session/{id}/stop ends session."""
        r = api_client.post(f"{api_url}/session/{session_id}/stop")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "stopped"
        assert "frames_processed" in data

    def test_double_stop_returns_404(self, api_client: requests.Session, api_url: str, session_id: str):
        """Stopping already-stopped session returns 404."""
        # First stop (fixture will stop, but we test the endpoint)
        api_client.post(f"{api_url}/session/{session_id}/stop")
        # Second stop
        r = api_client.post(f"{api_url}/session/{session_id}/stop")
        assert r.status_code == 404


class TestInferenceEndpoints:
    """Model inference tests."""

    @pytest.fixture
    def frame_b64(self, test_frame_b64: str) -> str:
        return test_frame_b64

    def test_single_frame_inference(self, api_client: requests.Session, api_url: str, session_id: str, frame_b64: str):
        """POST /inference/frame returns fatigue analysis."""
        r = api_client.post(
            f"{api_url}/inference/frame",
            json={"frame_data": frame_b64}
        )
        assert r.status_code == 200
        data = r.json()
        # Required fields
        assert "frame_id" in data
        assert "fatigue_state" in data
        assert "confidence" in data
        assert "eye_metrics" in data
        assert "mouth_metrics" in data
        assert "head_pose" in data
        assert "processing_time_ms" in data
        # Value ranges
        assert data["fatigue_state"] in ["alert", "drowsy", "fatigued", "unknown"]
        assert 0.0 <= data["confidence"] <= 1.0
        assert "avg_ear" in data["eye_metrics"]
        assert "perclos" in data["eye_metrics"]

    def test_sequence_inference(self, api_client: requests.Session, api_url: str, session_id: str, frame_b64: str):
        """POST /inference/sequence processes multiple frames."""
        r = api_client.post(
            f"{api_url}/inference/sequence",
            json={"frames_data": [frame_b64, frame_b64, frame_b64]}
        )
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, list)
        assert len(data) == 3
        for frame_result in data:
            assert "fatigue_state" in frame_result
            assert "confidence" in frame_result

    def test_inference_without_session_fails(self, api_client: requests.Session, api_url: str, frame_b64: str):
        """Inference without active session should still work (stateless)."""
        # The API is stateless per-frame, session is for tracking
        r = api_client.post(
            f"{api_url}/inference/frame",
            json={"frame_data": frame_b64}
        )
        assert r.status_code == 200


class TestErrorHandling:
    """Error handling tests."""

    def test_invalid_frame_data(self, api_client: requests.Session, api_url: str):
        """Invalid base64 returns 400 or 500."""
        r = api_client.post(
            f"{api_url}/inference/frame",
            json={"frame_data": "not_valid_base64!"}
        )
        assert r.status_code in (400, 500)

    def test_missing_frame_field(self, api_client: requests.Session, api_url: str):
        """Missing frame_data field returns 422."""
        r = api_client.post(f"{api_url}/inference/frame", json={})
        assert r.status_code == 422

    def test_invalid_session_id(self, api_client: requests.Session, api_url: str):
        """Invalid session ID returns 404."""
        r = api_client.post(f"{api_url}/session/invalid-id/stop")
        assert r.status_code == 404


@pytest.mark.slow
class TestPerformance:
    """Performance benchmarks."""

    def test_inference_latency_p95(self, api_client: requests.Session, api_url: str, session_id: str, frame_b64: str):
        """P95 latency should be under 500ms."""
        import time

        latencies = []
        for _ in range(20):
            start = time.perf_counter()
            r = api_client.post(
                f"{api_url}/inference/frame",
                json={"frame_data": frame_b64}
            )
            latencies.append((time.perf_counter() - start) * 1000)
            assert r.status_code == 200

        latencies.sort()
        p95 = latencies[int(0.95 * len(latencies))]
        print(f"\n  Latencies (ms): min={min(latencies):.1f}, p50={latencies[len(latencies)//2]:.1f}, p95={p95:.1f}, max={max(latencies):.1f}")
        assert p95 < 500, f"P95 latency {p95:.1f}ms exceeds 500ms threshold"


@pytest.mark.integration
class TestIntegration:
    """Full integration tests."""

    def test_full_session_flow(self, api_client: requests.Session, api_url: str, frame_b64: str):
        """Complete session: start → inference → stop."""
        # Start
        r = api_client.post(f"{api_url}/session/start", json={"source": "integration"})
        assert r.status_code == 200
        sid = r.json()["session_id"]

        # Multiple inferences
        for i in range(5):
            r = api_client.post(f"{api_url}/inference/frame", json={"frame_data": frame_b64})
            assert r.status_code == 200
            assert r.json()["fatigue_state"] in ["alert", "drowsy", "fatigued", "unknown"]

        # Status check
        r = api_client.get(f"{api_url}/session/{sid}/status")
        assert r.status_code == 200
        assert r.json()["frame_count"] >= 5

        # Stop
        r = api_client.post(f"{api_url}/session/{sid}/stop")
        assert r.status_code == 200
        assert r.json()["frames_processed"] >= 5