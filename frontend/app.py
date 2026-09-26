"""
Real-Time Cognitive Workload & Fatigue Monitoring Dashboard
Professional Streamlit Frontend for FastAPI Backend

Run: streamlit run frontend/app.py --server.headless true
"""

import os
import sys
import time
import base64
import json
import threading
import queue
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import cv2
import numpy as np
import requests
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


# ============================================================================
# Configuration & Constants
# ============================================================================

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# Multiple STUN servers for better connectivity
RTC_CONFIG = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
        {"urls": ["stun:stun3.l.google.com:19302"]},
        {"urls": ["stun:stun4.l.google.com:19302"]},
        {"urls": ["stun:stun.cloudflare.com:3478"]},
        {"urls": ["stun:stun.stunprotocol.org:3478"]},
    ],
    "iceTransportPolicy": "all",
    "bundlePolicy": "max-bundle",
    "rtcpMuxPolicy": "require",
})

# Professional color palette
COLORS = {
    "primary": "#2563eb",
    "primary_light": "#3b82f6",
    "success": "#22c55e",
    "warning": "#f59e0b",
    "danger": "#ef4444",
    "neutral": "#6b7280",
    "background": "#0f172a",
    "surface": "#1e293b",
    "surface_hover": "#334155",
    "text_primary": "#f8fafc",
    "text_secondary": "#94a3b8",
    "border": "#334155",
}

STATE_CONFIG = {
    "alert": {"color": COLORS["success"], "icon": "🟢", "label": "ALERT"},
    "drowsy": {"color": COLORS["warning"], "icon": "🟡", "label": "DROWSY"},
    "fatigued": {"color": COLORS["danger"], "icon": "🔴", "label": "FATIGUED"},
    "unknown": {"color": COLORS["neutral"], "icon": "⚪", "label": "UNKNOWN"},
}


# ============================================================================
# Custom CSS Injection
# ============================================================================

def inject_custom_css():
    """Inject professional custom CSS for the dashboard."""
    st.markdown(f"""
    <style>
        /* ===== Global Styles ===== */
        .stApp {{
            background: linear-gradient(135deg, {COLORS["background"]} 0%, #1e1b4b 100%);
        }}
        
        /* Hide Streamlit branding */
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        header {{visibility: hidden;}}
        
        /* ===== Typography ===== */
        .main-title {{
            font-size: 2.25rem;
            font-weight: 700;
            background: linear-gradient(135deg, {COLORS["text_primary"]} 0%, {COLORS["primary_light"]} 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 0.25rem;
            letter-spacing: -0.02em;
        }}
        
        .main-subtitle {{
            font-size: 1rem;
            color: {COLORS["text_secondary"]};
            margin-bottom: 2rem;
            font-weight: 400;
        }}
        
        .section-title {{
            font-size: 1.125rem;
            font-weight: 600;
            color: {COLORS["text_primary"]};
            margin: 1.5rem 0 1rem 0;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid {COLORS["border"]};
        }}
        
        /* ===== Metric Cards ===== */
        .metric-card {{
            background: {COLORS["surface"]};
            border: 1px solid {COLORS["border"]};
            border-radius: 12px;
            padding: 1.25rem;
            transition: all 0.2s ease;
        }}
        
        .metric-card:hover {{
            border-color: {COLORS["primary"]};
            box-shadow: 0 4px 20px rgba(37, 99, 235, 0.15);
        }}
        
        .metric-label {{
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: {COLORS["text_secondary"]};
            margin-bottom: 0.5rem;
            font-weight: 500;
        }}
        
        .metric-value {{
            font-size: 2rem;
            font-weight: 700;
            color: {COLORS["text_primary"]};
            line-height: 1.2;
        }}
        
        .metric-delta {{
            font-size: 0.75rem;
            margin-top: 0.25rem;
        }}
        
        /* ===== State Badge ===== */
        .state-badge {{
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem 1rem;
            border-radius: 9999px;
            font-weight: 600;
            font-size: 0.875rem;
        }}
        
        /* ===== Video Container ===== */
        .video-container {{
            background: {COLORS["surface"]};
            border: 1px solid {COLORS["border"]};
            border-radius: 16px;
            overflow: hidden;
            position: relative;
        }}
        
        .video-container video {{
            width: 100%;
            height: auto;
            display: block;
        }}
        
        /* ===== Sidebar ===== */
        .css-1d391kg {{
            background: {COLORS["surface"]};
            border-right: 1px solid {COLORS["border"]};
        }}
        
        .sidebar-section {{
            margin-bottom: 1.5rem;
        }}
        
        .sidebar-label {{
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: {COLORS["text_secondary"]};
            margin-bottom: 0.5rem;
            font-weight: 600;
        }}
        
        /* ===== Status Indicators ===== */
        .status-dot {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 0.5rem;
        }}
        
        .status-online {{ background: {COLORS["success"]}; }}
        .status-offline {{ background: {COLORS["danger"]}; }}
        
        /* ===== Alert Banner ===== */
        .alert-banner {{
            padding: 1rem 1.25rem;
            border-radius: 12px;
            margin-bottom: 1.5rem;
            display: flex;
            align-items: center;
            gap: 1rem;
            animation: slideIn 0.3s ease;
        }}
        
        @keyframes slideIn {{
            from {{ opacity: 0; transform: translateY(-10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        
        .alert-warning {{
            background: rgba(245, 158, 11, 0.15);
            border: 1px solid {COLORS["warning"]};
            color: {COLORS["warning"]};
        }}
        
        .alert-danger {{
            background: rgba(239, 68, 68, 0.15);
            border: 1px solid {COLORS["danger"]};
            color: {COLORS["danger"]};
        }}
        
        /* ===== Session Stats ===== */
        .stat-item {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.75rem 0;
            border-bottom: 1px solid {COLORS["border"]};
        }}
        
        .stat-item:last-child {{ border-bottom: none; }}
        
        .stat-label {{ color: {COLORS["text_secondary"]}; font-size: 0.875rem; }}
        .stat-value {{ font-weight: 600; color: {COLORS["text_primary"]}; }}
        
        /* ===== Button Styling ===== */
        .stButton > button {{
            border-radius: 8px;
            font-weight: 600;
            padding: 0.625rem 1.25rem;
            transition: all 0.2s ease;
            border: none;
        }}
        
        .stButton > button[kind="primary"] {{
            background: {COLORS["primary"]};
            color: white;
        }}
        
        .stButton > button[kind="primary"]:hover {{
            background: {COLORS["primary_light"]};
            box-shadow: 0 4px 16px rgba(37, 99, 235, 0.4);
        }}
        
        .stButton > button[kind="secondary"] {{
            background: {COLORS["surface_hover"]};
            color: {COLORS["text_primary"]};
            border: 1px solid {COLORS["border"]};
        }}
        
        .stButton > button[kind="secondary"]:hover {{
            background: {COLORS["border"]};
        }}
        
        /* ===== Slider ===== */
        .stSlider > div > div > div > div {{
            background: {COLORS["primary"]} !important;
        }}
        
        /* ===== Radio ===== */
        .stRadio > div {{
            gap: 0.75rem;
        }}
        
        .stRadio label {{
            background: {COLORS["surface"]};
            border: 1px solid {COLORS["border"]};
            border-radius: 8px;
            padding: 0.75rem 1rem;
            cursor: pointer;
            transition: all 0.2s ease;
        }}
        
        .stRadio label:hover {{
            border-color: {COLORS["primary"]};
        }}
        
        .stRadio input:checked + div {{
            border-color: {COLORS["primary"]} !important;
            background: rgba(37, 99, 235, 0.1);
        }}
        
        /* ===== Chart Container ===== */
        .chart-container {{
            background: {COLORS["surface"]};
            border: 1px solid {COLORS["border"]};
            border-radius: 12px;
            padding: 1rem;
            margin-bottom: 1rem;
        }}
        
        /* ===== Connection Status ===== */
        .connection-status {{
            display: flex;
            align-items: center;
            gap: 0.75rem;
            padding: 0.75rem 1rem;
            background: {COLORS["surface"]};
            border: 1px solid {COLORS["border"]};
            border-radius: 8px;
            margin-bottom: 1rem;
        }}
        
        /* ===== Info Box ===== */
        .info-box {{
            background: rgba(37, 99, 235, 0.1);
            border: 1px solid {COLORS["primary"]};
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1.5rem;
            color: {COLORS["primary_light"]};
            font-size: 0.875rem;
        }}
    </style>
    """, unsafe_allow_html=True)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class FrameMetrics:
    """Container for per-frame inference results."""
    timestamp: float
    frame_id: int
    fatigue_state: str
    confidence: float
    ear: float
    perclos: float
    blink_detected: bool
    mar: float
    yawn_detected: bool
    pitch: float
    yaw: float
    roll: float
    processing_time_ms: float


@dataclass
class SessionState:
    """Manages the monitoring session state."""
    active: bool = False
    session_id: Optional[str] = None
    frame_count: int = 0
    total_blinks: int = 0
    total_yawns: int = 0
    fatigue_counts: Dict[str, int] = field(default_factory=lambda: {"alert": 0, "drowsy": 0, "fatigued": 0})
    metrics_history: List[FrameMetrics] = field(default_factory=list)
    last_error: Optional[str] = None


# ============================================================================
# API Client
# ============================================================================

class APIClient:
    """Handles communication with FastAPI backend."""

    def __init__(self, base_url: str, timeout: int = 10):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()

    def health_check(self) -> bool:
        try:
            resp = self._session.get(f"{self.base_url}/health", timeout=3)
            return resp.status_code == 200 and resp.json().get("status") == "healthy"
        except Exception:
            return False

    def get_config(self) -> Optional[Dict]:
        try:
            resp = self._session.get(f"{self.base_url}/config", timeout=self.timeout)
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass
        return None

    def start_session(self) -> Optional[str]:
        try:
            resp = self._session.post(f"{self.base_url}/session/start", json={"source": "webcam"}, timeout=self.timeout)
            if resp.status_code == 200:
                return resp.json().get("session_id")
        except Exception as e:
            st.session_state.session.last_error = str(e)
        return None

    def stop_session(self, session_id: str) -> Optional[Dict]:
        try:
            resp = self._session.post(f"{self.base_url}/session/{session_id}/stop", timeout=self.timeout)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            st.session_state.session.last_error = str(e)
        return None

    def infer_frame(self, frame_b64: str) -> Optional[FrameMetrics]:
        try:
            resp = self._session.post(
                f"{self.base_url}/inference/frame",
                json={"frame_data": frame_b64},
                timeout=self.timeout
            )
            if resp.status_code == 200:
                data = resp.json()
                return FrameMetrics(
                    timestamp=data.get("timestamp", time.time()),
                    frame_id=data.get("frame_id", 0),
                    fatigue_state=data.get("fatigue_state", "unknown"),
                    confidence=data.get("confidence", 0.0),
                    ear=data.get("eye_metrics", {}).get("avg_ear", 0.0),
                    perclos=data.get("eye_metrics", {}).get("perclos", 0.0),
                    blink_detected=data.get("eye_metrics", {}).get("blink_detected", False),
                    mar=data.get("mouth_metrics", {}).get("mar", 0.0),
                    yawn_detected=data.get("mouth_metrics", {}).get("yawn_detected", False),
                    pitch=data.get("head_pose", {}).get("pitch", 0.0),
                    yaw=data.get("head_pose", {}).get("yaw", 0.0),
                    roll=data.get("head_pose", {}).get("roll", 0.0),
                    processing_time_ms=data.get("processing_time_ms", 0.0),
                )
        except Exception as e:
            st.session_state.session.last_error = str(e)
        return None


# ============================================================================
# Video Processor (WebRTC)
# ============================================================================

class VideoProcessor:
    """Processes webcam frames and sends to API asynchronously (WebRTC mode).
    
    This processor:
    1. Receives frames from the WebRTC stream
    2. Sends frames asynchronously to FastAPI /inference/frame
    3. Maintains latest metrics and overlays them on the returned video frame
    4. Handles connection errors gracefully
    """

    def __init__(
        self, 
        api_client: APIClient, 
        result_queue: queue.Queue, 
        confidence_threshold: float = 0.5,
        inference_interval: int = 3  # Run inference every N frames
    ):
        self.api_client = api_client
        self.result_queue = result_queue
        self.confidence_threshold = confidence_threshold
        self.inference_interval = inference_interval
        
        self.frame_count = 0
        self.running = False
        self.latest_metrics: Optional[FrameMetrics] = None
        self.inference_thread: Optional[threading.Thread] = None
        self._frame_lock = threading.Lock()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """Process incoming WebRTC frame and return annotated frame."""
        if not self.running:
            return frame

        self.frame_count += 1
        
        # Convert WebRTC frame to OpenCV format (BGR)
        img = frame.to_ndarray(format="bgr24")
        
        # Trigger inference every N frames
        if self.frame_count % self.inference_interval == 0:
            self._trigger_inference(img)
        
        # Overlay latest metrics on frame
        with self._frame_lock:
            if self.latest_metrics is not None:
                img = draw_overlays(img, self.latest_metrics)
        
        # Return annotated frame as new VideoFrame
        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def _trigger_inference(self, img: np.ndarray):
        """Send frame to API for inference in background thread."""
        # Encode frame as JPEG base64
        _, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame_b64 = base64.b64encode(buffer).decode("utf-8")
        
        # Start async inference
        self.inference_thread = threading.Thread(
            target=self._run_inference, 
            args=(frame_b64,), 
            daemon=True
        )
        self.inference_thread.start()

    def _run_inference(self, frame_b64: str):
        """Run inference and store results."""
        try:
            metrics = self.api_client.infer_frame(frame_b64)
            if metrics and metrics.confidence >= self.confidence_threshold:
                # Update latest metrics (thread-safe)
                with self._frame_lock:
                    self.latest_metrics = metrics
                # Also put in queue for session state tracking
                self.result_queue.put(metrics)
        except Exception as e:
            # Silently handle inference errors to avoid crashing video stream
            pass


# ============================================================================
# OpenCV Capture (Local Camera Fallback)
# ============================================================================

class OpenCVCapture:
    """OpenCV-based video capture for local camera (fallback when WebRTC fails)."""
    
    def __init__(self, api_client, result_queue, confidence_threshold, camera_index=0):
        self.api_client = api_client
        self.result_queue = result_queue
        self.confidence_threshold = confidence_threshold
        self.camera_index = camera_index
        self.cap = None
        self.running = False
        self.thread = None
        self.frame_placeholder = None
        self.latest_metrics = None
        self.last_frame_time = 0
        self.target_fps = 30
        self.frame_interval = 1.0 / self.target_fps

    def start(self, frame_placeholder):
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.camera_index}")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        
        self.frame_placeholder = frame_placeholder
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        if self.cap:
            self.cap.release()
            self.cap = None

    def _capture_loop(self):
        while self.running:
            current_time = time.time()
            if current_time - self.last_frame_time < self.frame_interval:
                time.sleep(0.001)
                continue
            
            self.last_frame_time = current_time
            
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame_b64 = base64.b64encode(buffer).decode("utf-8")
            
            threading.Thread(
                target=self._run_inference, 
                args=(frame_b64, frame.shape), 
                daemon=True
            ).start()
            
            if self.latest_metrics:
                frame = draw_overlays(frame, self.latest_metrics)
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            if self.frame_placeholder:
                try:
                    self.frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                except Exception:
                    pass

    def _run_inference(self, frame_b64, frame_shape):
        metrics = self.api_client.infer_frame(frame_b64)
        if metrics and metrics.confidence >= self.confidence_threshold:
            self.result_queue.put(metrics)
            self.latest_metrics = metrics


# ============================================================================
# Visualization Helpers
# ============================================================================

def draw_overlays(frame: np.ndarray, metrics: Optional[FrameMetrics]) -> np.ndarray:
    """Draw fatigue annotations on frame with professional styling."""
    if frame is None or metrics is None:
        return frame

    annotated = frame.copy()
    h, w = frame.shape[:2]
    
    cfg = STATE_CONFIG.get(metrics.fatigue_state, STATE_CONFIG["unknown"])
    color_hex = cfg["color"]
    color = tuple(int(color_hex.lstrip("#")[i:i+2], 16) for i in (4, 2, 0))

    # PERCLOS bar with gradient
    bar_width = int(200 * metrics.perclos)
    cv2.rectangle(annotated, (12, 32), (12 + bar_width, 52), color, -1)
    cv2.rectangle(annotated, (12, 32), (212, 52), (255, 255, 255), 1)
    cv2.putText(annotated, f"PERCLOS: {metrics.perclos:.1%}", (12, 28),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # EAR
    cv2.putText(annotated, f"EAR: {metrics.ear:.3f}", (12, 66),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Blink indicator
    if metrics.blink_detected:
        cv2.putText(annotated, "BLINK!", (w - 110, 36),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    # Yawn indicator
    if metrics.yawn_detected:
        cv2.putText(annotated, "YAWN!", (w - 110, 68),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2, cv2.LINE_AA)

    # State badge
    badge_text = f"{cfg['icon']} {cfg['label']} ({metrics.confidence:.0%})"
    (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(annotated, (12, 80), (12 + tw + 16, 110), (0, 0, 0), -1)
    cv2.rectangle(annotated, (12, 80), (12 + tw + 16, 110), color, 2)
    cv2.putText(annotated, badge_text, (20, 102),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    # Head pose
    cv2.putText(annotated, f"Pitch: {metrics.pitch:.1f}°  Yaw: {metrics.yaw:.1f}°  Roll: {metrics.roll:.1f}°",
               (12, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    # FPS
    if metrics.processing_time_ms > 0:
        fps = 1000 / metrics.processing_time_ms
        cv2.putText(annotated, f"FPS: {fps:.1f}", (w - 110, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

    return annotated


def create_state_badge(state: str, confidence: float) -> str:
    """Create HTML badge for fatigue state."""
    cfg = STATE_CONFIG.get(state, STATE_CONFIG["unknown"])
    return f"""
    <span class="state-badge" style="background: {cfg['color']}20; color: {cfg['color']}; border: 1px solid {cfg['color']};">
        {cfg['icon']} {cfg['label']} ({confidence:.0%})
    </span>
    """


def create_metric_card(label: str, value: str, delta: str = "", delta_color: str = "normal") -> str:
    """Create a professional metric card."""
    delta_html = ""
    if delta:
        delta_clr = COLORS["success"] if delta_color == "normal" else COLORS["danger"]
        delta_html = f'<div class="metric-delta" style="color: {delta_clr};">{delta}</div>'
    
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """


def create_alert_banner(state: str, message: str) -> str:
    """Create alert banner for drowsy/fatigued states."""
    if state == "drowsy":
        alert_class = "alert-warning"
        icon = "⚠️"
    elif state == "fatigued":
        alert_class = "alert-danger"
        icon = "🚨"
    else:
        return ""
    
    return f"""
    <div class="alert-banner {alert_class}">
        <span style="font-size: 1.5rem;">{icon}</span>
        <div>
            <strong>{message}</strong>
            <div style="font-size: 0.875rem; opacity: 0.8; margin-top: 0.25rem;">
                Consider taking a break immediately.
            </div>
        </div>
    </div>
    """


def create_charts(metrics_history: List[FrameMetrics]) -> go.Figure:
    """Create professional Plotly charts for telemetry."""
    if len(metrics_history) < 2:
        return None

    timestamps = [m.timestamp for m in metrics_history]
    t0 = timestamps[0]
    time_axis = [t - t0 for t in timestamps]

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Eye Metrics (EAR & PERCLOS)", "Head Pose (degrees)", "Processing Performance"),
        row_heights=[0.4, 0.35, 0.25],
    )

    # Color scheme
    colors = {
        "ear": COLORS["primary_light"],
        "perclos": COLORS["danger"],
        "pitch": "#22c55e",
        "yaw": COLORS["primary_light"],
        "roll": COLORS["warning"],
        "fps": COLORS["neutral"],
        "latency": COLORS["primary"],
    }

    # EAR & PERCLOS
    fig.add_trace(go.Scatter(
        x=time_axis, y=[m.ear for m in metrics_history],
        name="EAR", line=dict(color=colors["ear"], width=2),
        hovertemplate="EAR: %{y:.3f}<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=time_axis, y=[m.perclos for m in metrics_history],
        name="PERCLOS", line=dict(color=colors["perclos"], width=2),
        hovertemplate="PERCLOS: %{y:.1%}<extra></extra>",
    ), row=1, col=1)

    # Blink markers
    blink_times = [t for t, m in zip(time_axis, metrics_history) if m.blink_detected]
    blink_ears = [m.ear for m in metrics_history if m.blink_detected]
    if blink_times:
        fig.add_trace(go.Scatter(
            x=blink_times, y=blink_ears,
            mode="markers", name="Blink",
            marker=dict(color=COLORS["warning"], size=8, symbol="star"),
            hovertemplate="Blink detected<extra></extra>",
        ), row=1, col=1)

    # Head Pose
    for name, color, data_key in [("Pitch", colors["pitch"], "pitch"), 
                                   ("Yaw", colors["yaw"], "yaw"), 
                                   ("Roll", colors["roll"], "roll")]:
        fig.add_trace(go.Scatter(
            x=time_axis, y=[getattr(m, data_key) for m in metrics_history],
            name=name, line=dict(color=color, width=1.5),
            hovertemplate=f"{name}: %{{y:.1f}}°<extra></extra>",
        ), row=2, col=1)

    # Performance
    fig.add_trace(go.Scatter(
        x=time_axis, y=[1000/m.processing_time_ms if m.processing_time_ms > 0 else 0 for m in metrics_history],
        name="FPS", line=dict(color=colors["fps"], width=1.5, dash="dot"),
        hovertemplate="FPS: %{y:.1f}<extra></extra>",
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=time_axis, y=[m.processing_time_ms for m in metrics_history],
        name="Latency (ms)", line=dict(color=colors["latency"], width=1.5),
        yaxis="y2", hovertemplate="Latency: %{y:.1f}ms<extra></extra>",
    ), row=3, col=1)

    # Layout
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLORS["surface"],
        plot_bgcolor=COLORS["background"],
        font=dict(color=COLORS["text_primary"], size=11),
        margin=dict(l=50, r=30, t=50, b=40),
        height=550,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="right", x=1,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=10),
        ),
        hovermode="x unified",
    )

    fig.update_xaxes(
        gridcolor=COLORS["border"],
        zerolinecolor=COLORS["border"],
        showgrid=True,
    )
    fig.update_yaxes(
        gridcolor=COLORS["border"],
        zerolinecolor=COLORS["border"],
        showgrid=True,
    )

    # Y-axis ranges
    fig.update_yaxes(range=[0, 0.5], row=1, col=1)
    fig.update_yaxes(range=[-45, 45], row=2, col=1)

    return fig


def create_fatigue_distribution(fatigue_counts: Dict[str, int]) -> go.Figure:
    """Create donut chart for fatigue state distribution."""
    labels = list(fatigue_counts.keys())
    values = list(fatigue_counts.values())
    colors = [STATE_CONFIG[k]["color"] for k in labels]

    fig = go.Figure(data=[go.Pie(
        labels=[STATE_CONFIG[k]["label"] for k in labels],
        values=values,
        marker=dict(colors=colors, line=dict(color=COLORS["background"], width=2)),
        hole=0.55,
        textinfo="label+percent",
        textfont=dict(size=12, color=COLORS["text_primary"]),
        hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
    )])

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text_primary"], size=11),
        margin=dict(l=20, r=20, t=40, b=20),
        height=250,
        showlegend=True,
        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.05),
    )

    return fig


# ============================================================================
# Session State Management
# ============================================================================

def init_session_state():
    """Initialize Streamlit session state with all required keys."""
    defaults = {
        "session": SessionState(),
        "api_client": APIClient(API_BASE_URL),
        "result_queue": queue.Queue(maxsize=200),
        "video_processor_config": {"confidence_threshold": 0.5},
        "backend_online": False,
        "opencv_capture": None,
        "video_mode": "WebRTC (Browser Camera)",
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


def check_backend_health():
    """Check if FastAPI backend is reachable."""
    st.session_state.backend_online = st.session_state.api_client.health_check()
    return st.session_state.backend_online


def start_session():
    """Start a new monitoring session."""
    if not st.session_state.backend_online:
        st.error("Backend offline. Please start FastAPI server first.")
        return

    session_id = st.session_state.api_client.start_session()
    if session_id:
        s = st.session_state.session
        s.active = True
        s.session_id = session_id
        s.frame_count = 0
        s.total_blinks = 0
        s.total_yawns = 0
        s.fatigue_counts = {"alert": 0, "drowsy": 0, "fatigued": 0}
        s.metrics_history = []
        s.last_error = None
        
        st.session_state.video_processor_config["confidence_threshold"] = st.session_state.get("confidence_threshold", 0.5)
        st.toast(f"✅ Session started: {session_id[:8]}...", icon="🟢")
    else:
        st.error("Failed to start session")


def stop_session():
    """Stop the current monitoring session."""
    if st.session_state.session.active and st.session_state.session.session_id:
        summary = st.session_state.api_client.stop_session(st.session_state.session.session_id)
        if summary:
            st.toast(f"🛑 Session stopped. Frames: {summary.get('frames_processed', 0)}", icon="🔴")

    st.session_state.session.active = False
    st.session_state.session.session_id = None
    
    # Stop OpenCV capture if running
    if st.session_state.opencv_capture:
        st.session_state.opencv_capture.stop()
        st.session_state.opencv_capture = None


def process_results():
    """Process queued inference results and update session state."""
    s = st.session_state.session
    while not st.session_state.result_queue.empty():
        try:
            m = st.session_state.result_queue.get_nowait()
            s.metrics_history.append(m)
            s.frame_count += 1

            if m.blink_detected:
                s.total_blinks += 1
            if m.yawn_detected:
                s.total_yawns += 1

            s.fatigue_counts[m.fatigue_state] = s.fatigue_counts.get(m.fatigue_state, 0) + 1

            # Keep history bounded
            if len(s.metrics_history) > 500:
                s.metrics_history = s.metrics_history[-500:]
        except queue.Empty:
            break


# ============================================================================
# UI Components
# ============================================================================

def render_sidebar():
    """Render the professional sidebar control panel."""
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0; border-bottom: 1px solid {};">
            <h2 style="margin: 0; font-size: 1.25rem; font-weight: 700;">🎛️ Control Panel</h2>
        </div>
        """.format(COLORS["border"]), unsafe_allow_html=True)

        # Connection Status
        status_color = COLORS["success"] if st.session_state.backend_online else COLORS["danger"]
        status_text = "Online" if st.session_state.backend_online else "Offline"
        st.markdown(f"""
        <div class="connection-status">
            <div class="status-dot {'status-online' if st.session_state.backend_online else 'status-offline'}"></div>
            <div>
                <strong>Backend Status</strong><br>
                <span style="color: {status_color}; font-weight: 600;">{status_text}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🔄 Check Connection", use_container_width=True):
            check_backend_health()
            st.rerun()

        st.markdown("<div class='sidebar-section'></div>", unsafe_allow_html=True)

        # Video Source Selection
        st.markdown("<div class='sidebar-label'>📷 Video Source</div>", unsafe_allow_html=True)
        st.radio(
            "Mode",
            options=["WebRTC (Browser Camera)", "OpenCV (Local Camera)"],
            index=0,
            help="WebRTC: Browser camera via WebRTC (needs STUN). OpenCV: Local camera directly (same machine only).",
            key="video_mode",
            label_visibility="collapsed",
        )

        st.markdown("<div class='sidebar-section'></div>", unsafe_allow_html=True)

        # Session Controls
        if not st.session_state.session.active:
            if st.button("🟢 Start Monitoring", type="primary", use_container_width=True):
                start_session()
                st.rerun()
        else:
            if st.button("🔴 Stop Monitoring", type="secondary", use_container_width=True):
                stop_session()
                st.rerun()

        st.markdown("<div class='sidebar-section'></div>", unsafe_allow_html=True)

        # Settings
        st.markdown("<div class='sidebar-label'>⚙️ Settings</div>", unsafe_allow_html=True)
        confidence = st.slider(
            "Confidence Threshold",
            min_value=0.1, max_value=0.9,
            value=st.session_state.video_processor_config.get("confidence_threshold", 0.5),
            step=0.05,
            help="Minimum confidence for displaying predictions",
            key="confidence_threshold",
        )
        st.session_state.video_processor_config["confidence_threshold"] = confidence

        st.markdown("<div class='sidebar-section'></div>", unsafe_allow_html=True)

        # Session Stats (when active)
        if st.session_state.session.active:
            st.markdown("<div class='sidebar-label'>📊 Session Stats</div>", unsafe_allow_html=True)
            s = st.session_state.session
            
            elapsed = 0
            if s.metrics_history:
                elapsed = s.metrics_history[-1].timestamp - s.metrics_history[0].timestamp
            
            stats = [
                ("Frames Processed", str(s.frame_count)),
                ("Blinks Detected", str(s.total_blinks)),
                ("Yawns Detected", str(s.total_yawns)),
                ("Duration", f"{elapsed:.0f}s"),
            ]
            
            for label, value in stats:
                st.markdown(f"""
                <div class="stat-item">
                    <span class="stat-label">{label}</span>
                    <span class="stat-value">{value}</span>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<div class='sidebar-section'></div>", unsafe_allow_html=True)
        st.caption("💡 Ensure FastAPI backend is running on port 8000")


def render_video_feed():
    """Render the live video feed based on selected mode."""
    st.markdown('<div class="section-title">📹 Live Video Feed</div>', unsafe_allow_html=True)

    if not st.session_state.backend_online:
        st.markdown(f"""
        <div class="info-box">
            <strong>⚠️ Backend Offline</strong><br>
            Please start the FastAPI server on port 8000 to begin monitoring.
            <br><code style="font-size: 0.75rem;">cd driver-fatigue-monitor && python -m uvicorn inference.main:app --host 0.0.0.0 --port 8000</code>
        </div>
        """, unsafe_allow_html=True)
        return

    video_mode = st.session_state.get("video_mode", "WebRTC (Browser Camera)")

    if video_mode == "OpenCV (Local Camera)":
        render_opencv_feed()
    else:
        render_webrtc_feed()


def render_webrtc_feed():
    """Render video feed using WebRTC (browser camera)."""
    # Capture dependencies in main thread (has session context)
    api_client = st.session_state.api_client
    result_queue = st.session_state.result_queue
    confidence_threshold = st.session_state.video_processor_config.get("confidence_threshold", 0.5)

    def create_processor():
        return VideoProcessor(
            api_client, 
            result_queue, 
            confidence_threshold,
            inference_interval=3  # Process every 3rd frame to balance performance
        )

    # WebRTC streamer with proper configuration
    ctx = webrtc_streamer(
        key="fatigue-monitor-webrtc",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIG,
        video_processor_factory=create_processor,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 640},
                "height": {"ideal": 480},
                "frameRate": {"ideal": 30, "max": 30}
            }, 
            "audio": False
        },
        async_processing=True,
    )
    
    # Show WebRTC connection status
    if ctx.state.playing:
        st.success("🟢 WebRTC Connected - Camera Active")
    elif ctx.state.signalling:
        st.info("🔄 Connecting to camera...")
    else:
        st.warning("🟡 Camera not started - Click 'Start Monitoring' to begin")


def render_opencv_feed():
    """Render video feed using OpenCV (local camera)."""
    st.markdown("""
    <div class="info-box">
        <strong>📷 OpenCV Mode Active</strong><br>
        Using local camera via OpenCV. Camera must be on the same machine as Streamlit server.
    </div>
    """, unsafe_allow_html=True)
    
    camera_index = st.selectbox(
        "Camera", 
        options=[0, 1, 2], 
        index=0,
        format_func=lambda x: f"Camera {x}",
        key="opencv_camera_index"
    )
    
    frame_placeholder = st.empty()
    
    if "opencv_capture" not in st.session_state:
        st.session_state.opencv_capture = None
    
    if st.session_state.session.active:
        if st.session_state.opencv_capture is None:
            try:
                st.session_state.opencv_capture = OpenCVCapture(
                    api_client=st.session_state.api_client,
                    result_queue=st.session_state.result_queue,
                    confidence_threshold=st.session_state.video_processor_config.get("confidence_threshold", 0.5),
                    camera_index=camera_index
                )
                st.session_state.opencv_capture.start(frame_placeholder)
            except Exception as e:
                st.error(f"Failed to start camera: {e}")
                st.session_state.session.active = False
                st.rerun()
    else:
        if st.session_state.opencv_capture:
            st.session_state.opencv_capture.stop()
            st.session_state.opencv_capture = None
        frame_placeholder.info("Session stopped. Click 'Start Monitoring' to begin.")


def render_telemetry_panel():
    """Render the right-side telemetry panel with metrics and charts."""
    s = st.session_state.session
    
    st.markdown('<div class="section-title">📊 Real-Time Telemetry</div>', unsafe_allow_html=True)

    # Alert banner for drowsy/fatigued
    if s.metrics_history:
        latest = s.metrics_history[-1]
        if latest.fatigue_state == "drowsy":
            st.markdown(create_alert_banner("drowsy", "Drowsiness Detected — Driver shows signs of fatigue"), unsafe_allow_html=True)
        elif latest.fatigue_state == "fatigued":
            st.markdown(create_alert_banner("fatigued", "High Fatigue Risk — Immediate break recommended"), unsafe_allow_html=True)

    if not s.metrics_history:
        st.info("No data yet. Start a session to see live metrics.")
        return

    latest = s.metrics_history[-1]
    cfg = STATE_CONFIG.get(latest.fatigue_state, STATE_CONFIG["unknown"])

    # Current State Card
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {cfg['color']}20 0%, {cfg['color']}10 100%);
        border: 1px solid {cfg['color']};
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        text-align: center;
    ">
        <div style="font-size: 3rem; margin-bottom: 0.5rem;">{cfg['icon']}</div>
        <div style="font-size: 1.5rem; font-weight: 700; color: {cfg['color']};">{cfg['label']}</div>
        <div style="color: {COLORS['text_secondary']}; margin-top: 0.5rem;">Confidence: {latest.confidence:.1%}</div>
    </div>
    """, unsafe_allow_html=True)

    # Metric Cards Grid
    col1, col2, col3, col4 = st.columns(4)
    
    ear_delta = "⚠️ Below threshold" if latest.ear < 0.25 else "✅ Normal"
    ear_color = "inverse" if latest.ear < 0.25 else "normal"
    
    perclos_delta = "⚠️ Elevated" if latest.perclos > 0.15 else "✅ Normal"
    perclos_color = "inverse" if latest.perclos > 0.15 else "normal"
    
    with col1:
        st.markdown(create_metric_card("EAR (Eye Aspect Ratio)", f"{latest.ear:.3f}", ear_delta, ear_color), unsafe_allow_html=True)
    
    with col2:
        st.markdown(create_metric_card("PERCLOS", f"{latest.perclos:.1%}", perclos_delta, perclos_color), unsafe_allow_html=True)
    
    with col3:
        blink_rate = s.total_blinks / max(1, len(s.metrics_history) / 30) if s.metrics_history else 0
        st.markdown(create_metric_card("Blink Rate", f"{blink_rate:.1f}/min"), unsafe_allow_html=True)
    
    with col4:
        st.markdown(create_metric_card("Head Pose", f"P:{latest.pitch:.0f}° Y:{latest.yaw:.0f}° R:{latest.roll:.0f}°"), unsafe_allow_html=True)

    # Charts
    st.markdown('<div class="section-title">📈 Session Trends</div>', unsafe_allow_html=True)
    
    chart = create_charts(s.metrics_history)
    if chart:
        st.plotly_chart(chart, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("Collecting data... charts will appear after a few frames.")

    # Fatigue Distribution
    if s.fatigue_counts and sum(s.fatigue_counts.values()) > 0:
        st.markdown('<div class="section-title">📊 Fatigue State Distribution</div>', unsafe_allow_html=True)
        dist_fig = create_fatigue_distribution(s.fatigue_counts)
        if dist_fig:
            st.plotly_chart(dist_fig, use_container_width=True, config={"displayModeBar": False})


def render_instructions():
    """Render usage instructions in an expander."""
    with st.expander("📖 How to Use", expanded=not st.session_state.backend_online):
        st.markdown(f"""
        ### Quick Start Guide
        
        **1. Start FastAPI Backend (Terminal 1):**
        ```bash
        cd driver-fatigue-monitor
        python -m uvicorn inference.main:app --host 0.0.0.0 --port 8000
        ```

        **2. Start Streamlit Frontend (Terminal 2):**
        ```bash
        cd driver-fatigue-monitor
        streamlit run frontend/app.py --server.headless true
        ```

        **3. Begin Monitoring:**
        - Click **🟢 Start Monitoring** in the sidebar
        - Allow camera access when prompted
        - View live fatigue detection and metrics

        ### Video Modes
        | Mode | Camera Location | STUN Required | Best For |
        |------|----------------|---------------|----------|
        | **WebRTC** | Your browser device | Yes | Remote access, demos |
        | **OpenCV** | Streamlit server machine | No | Local development, reliable |

        ### API Endpoints Used
        - `GET /health` — Backend health check
        - `POST /session/start` — Initialize session
        - `POST /inference/frame` — Single frame inference
        - `POST /session/{{id}}/stop` — End session

        ### Troubleshooting
        - **WebRTC connection fails** → Switch to OpenCV mode in sidebar
        - **Backend shows offline** → Ensure FastAPI is running on port 8000
        - **No camera detected** → Check camera index in OpenCV mode (0, 1, 2)
        """)


# ============================================================================
# Main Application
# ============================================================================

def main():
    """Main Streamlit application entry point."""
    # Page configuration
    st.set_page_config(
        page_title="Driver Fatigue Monitor",
        page_icon="🚗",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Inject custom CSS
    inject_custom_css()

    # Initialize session state
    init_session_state()

    # Check backend on first load
    if not st.session_state.backend_online:
        check_backend_health()

    # Header
    st.markdown('<h1 class="main-title">🚗 Driver Fatigue Monitor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="main-subtitle">Real-time cognitive workload & fatigue monitoring using hybrid CNN-LSTM-Transformer</p>', unsafe_allow_html=True)

    # Sidebar
    render_sidebar()

    # Process any queued results
    process_results()

    # Main Dashboard Layout
    col1, col2 = st.columns([2, 1], gap="large")

    with col1:
        render_video_feed()

    with col2:
        render_telemetry_panel()

    # Instructions
    render_instructions()

    # Auto-refresh for live updates
    if st.session_state.session.active:
        time.sleep(0.1)
        st.rerun()


if __name__ == "__main__":
    main()