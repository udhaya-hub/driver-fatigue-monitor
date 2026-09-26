"""
Streamlit MVP Frontend for Driver Fatigue Monitoring System.

This dashboard provides:
1. Real-time video feed with fatigue annotations
2. Live metrics visualization (EAR, PERCLOS, Head Pose)
3. Fatigue state history and alerts
4. Configuration panel
5. Session management
"""

import streamlit as st
import cv2
import numpy as np
import base64
import asyncio
import websockets
import json
import time
import threading
import queue
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict
from enum import Enum
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import requests
import yaml
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Driver Fatigue Monitor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1f2937;
    }
    .metric-label {
        font-size: 0.875rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .alert-drowsy {
        background: #fef3c7;
        border-left: 4px solid #f59e0b;
    }
    .alert-fatigued {
        background: #fee2e2;
        border-left: 4px solid #ef4444;
    }
    .alert-alert {
        background: #dcfce7;
        border-left: 4px solid #22c55e;
    }
    .stVideo > div {
        border-radius: 12px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Data Classes
# ============================================================================

class FatigueState(str, Enum):
    ALERT = "alert"
    DROWSY = "drowsy"
    FATIGUED = "fatigued"
    UNKNOWN = "unknown"


@dataclass
class FrameMetrics:
    timestamp: float
    frame_id: int
    fatigue_state: FatigueState
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


# ============================================================================
# Session State Management
# ============================================================================

def init_session_state():
    """Initialize Streamlit session state."""
    if "ws_client" not in st.session_state:
        st.session_state.ws_client = None
    if "ws_thread" not in st.session_state:
        st.session_state.ws_thread = None
    if "frame_queue" not in st.session_state:
        st.session_state.frame_queue = queue.Queue(maxsize=10)
    if "metrics_history" not in st.session_state:
        st.session_state.metrics_history = []
    if "current_frame" not in st.session_state:
        st.session_state.current_frame = None
    if "current_metrics" not in st.session_state:
        st.session_state.current_metrics = None
    if "session_active" not in st.session_state:
        st.session_state.session_active = False
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    if "api_base_url" not in st.session_state:
        st.session_state.api_base_url = "http://localhost:8000"
    if "ws_url" not in st.session_state:
        st.session_state.ws_url = "ws://localhost:8000/ws"
    if "total_frames" not in st.session_state:
        st.session_state.total_frames = 0
    if "total_blinks" not in st.session_state:
        st.session_state.total_blinks = 0
    if "total_yawns" not in st.session_state:
        st.session_state.total_yawns = 0
    if "fatigue_counts" not in st.session_state:
        st.session_state.fatigue_counts = {"alert": 0, "drowsy": 0, "fatigued": 0}


init_session_state()


# ============================================================================
# WebSocket Client
# ============================================================================

class WebSocketClient:
    """WebSocket client for real-time communication with backend."""

    def __init__(self, ws_url: str, session_id: str, frame_queue: queue.Queue):
        self.ws_url = f"{ws_url}/{session_id}"
        self.session_id = session_id
        self.frame_queue = frame_queue
        self.running = False
        self.loop = None

    def start(self):
        """Start WebSocket connection in background thread."""
        self.running = True
        self.loop = asyncio.new_event_loop()
        thread = threading.Thread(target=self._run_loop, daemon=True)
        thread.start()
        return thread

    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._connect())

    async def _connect(self):
        while self.running:
            try:
                async with websockets.connect(self.ws_url) as websocket:
                    st.session_state.ws_client = websocket
                    # Send ping to keep alive
                    asyncio.create_task(self._keep_alive(websocket))

                    async for message in websocket:
                        if not self.running:
                            break
                        data = json.loads(message)
                        await self._handle_message(data)

            except Exception as e:
                print(f"WebSocket error: {e}")
                if self.running:
                    await asyncio.sleep(5)  # Reconnect delay

    async def _keep_alive(self, websocket):
        while self.running:
            await asyncio.sleep(30)
            try:
                await websocket.send(json.dumps({"type": "ping"}))
            except:
                break

    async def _handle_message(self, data: Dict):
        if data.get("type") == "result":
            metrics = FrameMetrics(
                timestamp=data.get("timestamp", time.time()),
                frame_id=data.get("frame_id", 0),
                fatigue_state=FatigueState(data.get("fatigue_state", "unknown")),
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
            # Put in queue for UI thread
            try:
                self.frame_queue.put_nowait(metrics)
            except queue.Full:
                try:
                    self.frame_queue.get_nowait()
                    self.frame_queue.put_nowait(metrics)
                except queue.Empty:
                    pass

    def send_frame(self, frame_b64: str, physio: Optional[list] = None):
        """Send frame to backend."""
        if self.loop and self.loop.is_running():
            message = {
                "type": "frame",
                "data": frame_b64,
                "timestamp": time.time(),
                "physio": physio,
            }
            asyncio.run_coroutine_threadsafe(
                self._send_message(message), self.loop
            )

    async def _send_message(self, message: Dict):
        if st.session_state.ws_client:
            try:
                await st.session_state.ws_client.send(json.dumps(message))
            except Exception as e:
                print(f"Send error: {e}")

    def stop(self):
        self.running = False
        if self.loop:
            self.loop.call_soon_threadsafe(self.loop.stop)


# ============================================================================
# Helper Functions
# ============================================================================

def frame_to_base64(frame: np.ndarray) -> str:
    """Convert frame to base64 encoded JPEG."""
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buffer).decode('utf-8')


def draw_annotations(frame: np.ndarray, metrics: FrameMetrics) -> np.ndarray:
    """Draw fatigue annotations on frame."""
    annotated = frame.copy()
    h, w = frame.shape[:2]

    # Color based on fatigue state
    if metrics.fatigue_state == FatigueState.ALERT:
        color = (0, 255, 0)  # Green
    elif metrics.fatigue_state == FatigueState.DROWSY:
        color = (0, 255, 255)  # Yellow
    else:
        color = (0, 0, 255)  # Red

    # PERCLOS bar
    bar_width = int(200 * metrics.perclos)
    cv2.rectangle(annotated, (10, 30), (10 + bar_width, 50), color, -1)
    cv2.rectangle(annotated, (10, 30), (210, 50), (255, 255, 255), 2)
    cv2.putText(annotated, f"PERCLOS: {metrics.perclos:.1%}", (10, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # EAR
    cv2.putText(annotated, f"EAR: {metrics.ear:.3f}", (10, 65),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Blink indicator
    if metrics.blink_detected:
        cv2.putText(annotated, "BLINK!", (w - 100, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Yawn indicator
    if metrics.yawn_detected:
        cv2.putText(annotated, "YAWN!", (w - 100, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

    # Fatigue state
    cv2.putText(annotated, f"State: {metrics.fatigue_state.value.upper()} ({metrics.confidence:.2f})",
               (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Head pose
    cv2.putText(annotated, f"Pitch: {metrics.pitch:.1f} Yaw: {metrics.yaw:.1f} Roll: {metrics.roll:.1f}",
               (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # FPS
    if metrics.processing_time_ms > 0:
        fps = 1000 / metrics.processing_time_ms
        cv2.putText(annotated, f"FPS: {fps:.1f}", (w - 100, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return annotated


def create_metrics_chart(history: list) -> go.Figure:
    """Create real-time metrics chart."""
    if not history:
        fig = go.Figure()
        fig.update_layout(
            title="Real-time Metrics",
            xaxis_title="Time",
            yaxis_title="Value",
            template="plotly_dark",
            height=300,
        )
        return fig

    df = pd.DataFrame([asdict(m) for m in history])

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=("EAR & PERCLOS", "Head Pose (degrees)", "Processing Time (ms)"),
        row_heights=[0.4, 0.3, 0.3],
    )

    # EAR and PERCLOS
    fig.add_trace(
        go.Scatter(x=df["timestamp"], y=df["ear"], name="EAR", line=dict(color="#3b82f6")),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df["timestamp"], y=df["perclos"], name="PERCLOS", line=dict(color="#ef4444")),
        row=1, col=1
    )

    # Blink markers
    blinks = df[df["blink_detected"]]
    if not blinks.empty:
        fig.add_trace(
            go.Scatter(
                x=blinks["timestamp"], y=blinks["ear"],
                mode="markers", name="Blink",
                marker=dict(color="#f59e0b", size=8, symbol="star"),
            ),
            row=1, col=1
        )

    # Head pose
    fig.add_trace(
        go.Scatter(x=df["timestamp"], y=df["pitch"], name="Pitch", line=dict(color="#22c55e")),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=df["timestamp"], y=df["yaw"], name="Yaw", line=dict(color="#3b82f6")),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=df["timestamp"], y=df["roll"], name="Roll", line=dict(color="#f59e0b")),
        row=2, col=1
    )

    # Processing time
    fig.add_trace(
        go.Scatter(x=df["timestamp"], y=df["processing_time_ms"], name="Processing", line=dict(color="#8b5cf6")),
        row=3, col=1
    )

    fig.update_layout(
        template="plotly_dark",
        height=500,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=50, b=50),
    )

    fig.update_xaxes(rangeslider_visible=False)
    fig.update_yaxes(range=[0, 0.5], row=1, col=1)
    fig.update_yaxes(range=[-45, 45], row=2, col=1)

    return fig


def create_fatigue_pie_chart(counts: Dict[str, int]) -> go.Figure:
    """Create fatigue state distribution pie chart."""
    labels = list(counts.keys())
    values = list(counts.values())
    colors = ["#22c55e", "#f59e0b", "#ef4444"]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker_colors=colors,
        hole=0.4,
        textinfo="label+percent",
        textfont_size=12,
    )])

    fig.update_layout(
        title="Fatigue State Distribution",
        template="plotly_dark",
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
    )

    return fig


# ============================================================================
# Main Dashboard
# ============================================================================

def main():
    st.markdown('<h1 class="main-header">🚗 Driver Fatigue Monitor</h1>', unsafe_allow_html=True)
    st.markdown("Real-time cognitive workload & fatigue monitoring using hybrid CNN-LSTM-Transformer")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")

        # API Configuration
        api_url = st.text_input("API Base URL", value=st.session_state.api_base_url)
        ws_url = st.text_input("WebSocket URL", value=st.session_state.ws_url)

        if api_url != st.session_state.api_base_url:
            st.session_state.api_base_url = api_url
        if ws_url != st.session_state.ws_url:
            st.session_state.ws_url = ws_url

        st.divider()

        # Session Controls
        st.subheader("Session Control")

        if not st.session_state.session_active:
            if st.button("🟢 Start Monitoring", type="primary", use_container_width=True):
                start_session()
        else:
            if st.button("🔴 Stop Monitoring", type="secondary", use_container_width=True):
                stop_session()

        st.divider()

        # Video Source
        st.subheader("Video Source")
        video_source = st.selectbox(
            "Select Source",
            ["Webcam (0)", "Webcam (1)", "Video File", "RTSP Stream"],
            index=0,
        )

        if video_source == "Video File":
            uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
        elif video_source == "RTSP Stream":
            rtsp_url = st.text_input("RTSP URL", placeholder="rtsp://...")

        st.divider()

        # Thresholds
        st.subheader("Detection Thresholds")
        ear_threshold = st.slider("EAR Threshold", 0.1, 0.4, 0.25, 0.01)
        perclos_threshold = st.slider("PERCLOS Alert Threshold", 0.05, 0.5, 0.15, 0.01)
        mar_threshold = st.slider("MAR Yawn Threshold", 0.3, 1.0, 0.6, 0.05)

        st.divider()

        # Model Info
        st.subheader("Model Info")
        if st.button("Check Model Status"):
            check_model_status()

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📹 Live Video Feed")
        video_placeholder = st.empty()

        st.subheader("📊 Real-time Metrics")
        chart_placeholder = st.empty()

    with col2:
        st.subheader("📈 Session Statistics")
        stats_placeholder = st.empty()

        st.subheader("⚠️ Alerts")
        alerts_placeholder = st.empty()

        st.subheader("📋 Fatigue Distribution")
        pie_placeholder = st.empty()

    # Process frame queue
    process_frame_queue()

    # Update UI
    update_video_feed(video_placeholder)
    update_metrics_chart(chart_placeholder)
    update_statistics(stats_placeholder)
    update_alerts(alerts_placeholder)
    update_pie_chart(pie_placeholder)

    # Auto-refresh
    if st.session_state.session_active:
        time.sleep(0.1)
        st.rerun()


def start_session():
    """Start monitoring session."""
    try:
        # Create session via API
        response = requests.post(
            f"{st.session_state.api_base_url}/session/start",
            json={"source": "webcam"},
            timeout=5
        )
        if response.status_code == 200:
            data = response.json()
            st.session_state.session_id = data["session_id"]
            st.session_state.session_active = True
            st.session_state.metrics_history = []
            st.session_state.total_frames = 0
            st.session_state.total_blinks = 0
            st.session_state.total_yawns = 0
            st.session_state.fatigue_counts = {"alert": 0, "drowsy": 0, "fatigued": 0}

            # Start WebSocket client
            ws_client = WebSocketClient(
                st.session_state.ws_url,
                st.session_state.session_id,
                st.session_state.frame_queue
            )
            st.session_state.ws_thread = ws_client.start()

            st.success(f"Session started: {st.session_state.session_id}")
        else:
            st.error("Failed to start session")
    except Exception as e:
        st.error(f"Error starting session: {e}")


def stop_session():
    """Stop monitoring session."""
    st.session_state.session_active = False

    # Stop WebSocket
    if st.session_state.ws_client:
        # Note: In practice, you'd properly close the WebSocket
        pass

    # Stop session via API
    if st.session_state.session_id:
        try:
            requests.post(
                f"{st.session_state.api_base_url}/session/{st.session_state.session_id}/stop",
                timeout=5
            )
        except:
            pass

    st.session_state.session_id = None
    st.success("Session stopped")


def check_model_status():
    """Check model status via API."""
    try:
        response = requests.get(f"{st.session_state.api_base_url}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            st.sidebar.success(f"Model: {data['status']}")
            st.sidebar.info(f"Device: {data['device']}")
            st.sidebar.info(f"Uptime: {data['uptime_seconds']:.0f}s")
        else:
            st.sidebar.error("API unavailable")
    except Exception as e:
        st.sidebar.error(f"Connection failed: {e}")


def process_frame_queue():
    """Process incoming frames from WebSocket queue."""
    while not st.session_state.frame_queue.empty():
        try:
            metrics = st.session_state.frame_queue.get_nowait()
            st.session_state.metrics_history.append(metrics)

            # Keep history limited
            if len(st.session_state.metrics_history) > 500:
                st.session_state.metrics_history = st.session_state.metrics_history[-500:]

            # Update counters
            st.session_state.total_frames += 1
            if metrics.blink_detected:
                st.session_state.total_blinks += 1
            if metrics.yawn_detected:
                st.session_state.total_yawns += 1
            st.session_state.fatigue_counts[metrics.fatigue_state.value] += 1

            st.session_state.current_metrics = metrics
        except queue.Empty:
            break


def update_video_feed(placeholder):
    """Update video feed display."""
    # In a real implementation, this would show the actual webcam feed
    # For now, show a placeholder
    if st.session_state.current_metrics:
        # Create a synthetic frame with annotations
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = draw_annotations(frame, st.session_state.current_metrics)
        placeholder.image(frame, channels="BGR", use_column_width=True)
    else:
        placeholder.info("Waiting for video feed... Start a session to begin monitoring.")


def update_metrics_chart(placeholder):
    """Update metrics chart."""
    if st.session_state.metrics_history:
        fig = create_metrics_chart(st.session_state.metrics_history)
        placeholder.plotly_chart(fig, use_container_width=True)
    else:
        fig = create_metrics_chart([])
        placeholder.plotly_chart(fig, use_container_width=True)


def update_statistics(placeholder):
    """Update session statistics."""
    with placeholder.container():
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{st.session_state.total_frames}</div>
                <div class="metric-label">Frames Processed</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{st.session_state.total_blinks}</div>
                <div class="metric-label">Blinks Detected</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{st.session_state.total_yawns}</div>
                <div class="metric-label">Yawns Detected</div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            elapsed = 0
            if st.session_state.metrics_history:
                elapsed = st.session_state.metrics_history[-1].timestamp - st.session_state.metrics_history[0].timestamp
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{elapsed:.0f}s</div>
                <div class="metric-label">Session Duration</div>
            </div>
            """, unsafe_allow_html=True)

        # Current state
        if st.session_state.current_metrics:
            m = st.session_state.current_metrics
            state_class = f"alert-{m.fatigue_state.value}"
            st.markdown(f"""
            <div class="metric-card {state_class}" style="margin-top: 1rem;">
                <div class="metric-label">Current State</div>
                <div class="metric-value" style="color: {'#22c55e' if m.fatigue_state == FatigueState.ALERT else '#f59e0b' if m.fatigue_state == FatigueState.DROWSY else '#ef4444'};">
                    {m.fatigue_state.value.upper()}
                </div>
                <div class="metric-label">Confidence: {m.confidence:.1%}</div>
            </div>
            """, unsafe_allow_html=True)


def update_alerts(placeholder):
    """Update alerts panel."""
    with placeholder.container():
        if not st.session_state.metrics_history:
            st.info("No alerts yet")
            return

        recent = st.session_state.metrics_history[-10:]
        alerts = []

        for m in recent:
            if m.fatigue_state == FatigueState.DROWSY:
                alerts.append(f"⚠️ **Drowsiness detected** at {datetime.fromtimestamp(m.timestamp).strftime('%H:%M:%S')} (conf: {m.confidence:.0%})")
            elif m.fatigue_state == FatigueState.FATIGUED:
                alerts.append(f"🚨 **Fatigue detected** at {datetime.fromtimestamp(m.timestamp).strftime('%H:%M:%S')} (conf: {m.confidence:.0%})")
            if m.yawn_detected:
                alerts.append(f"😮 **Yawn detected** at {datetime.fromtimestamp(m.timestamp).strftime('%H:%M:%S')}")

        if alerts:
            for alert in reversed(alerts[-5:]):  # Show last 5
                st.markdown(alert)
        else:
            st.success("✅ No fatigue alerts")


def update_pie_chart(placeholder):
    """Update fatigue distribution pie chart."""
    fig = create_fatigue_pie_chart(st.session_state.fatigue_counts)
    placeholder.plotly_chart(fig, use_container_width=True)


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    main()