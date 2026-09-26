"""
FastAPI Backend for Real-Time Cognitive Workload & Fatigue Monitoring.

This module provides:
1. REST API endpoints for health checks and configuration
2. WebSocket endpoint for real-time video frame streaming and inference
3. Model inference pipeline with batching support
4. Prometheus metrics for monitoring
"""

import asyncio
import base64
import io
import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field
from enum import Enum

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, BackgroundTasks, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
import yaml

from models.hybrid_model import HybridFatigueModel, create_model, ModelOutput
from .vision_processor import VisionProcessor, VisionFrameResult, FatigueState, VideoCaptureManager
from utils.config_loader import load_config
from utils.logger import setup_logger

logger = setup_logger(__name__)


# ============================================================================
# Prometheus Metrics
# ============================================================================

INFERENCE_REQUESTS = Counter(
    "fatigue_inference_requests_total",
    "Total inference requests",
    ["endpoint", "status"]
)

INFERENCE_LATENCY = Histogram(
    "fatigue_inference_latency_seconds",
    "Inference latency in seconds",
    ["endpoint"]
)

ACTIVE_CONNECTIONS = Gauge(
    "fatigue_active_websocket_connections",
    "Number of active WebSocket connections"
)

FRAMES_PROCESSED = Counter(
    "fatigue_frames_processed_total",
    "Total frames processed",
    ["source"]
)

FATIGUE_STATE_DISTRIBUTION = Counter(
    "fatigue_state_total",
    "Distribution of fatigue states",
    ["state"]
)


# ============================================================================
# Pydantic Models
# ============================================================================

class FatigueStateEnum(str, Enum):
    ALERT = "alert"
    DROWSY = "drowsy"
    FATIGUED = "fatigued"
    UNKNOWN = "unknown"


class EyeMetricsResponse(BaseModel):
    left_ear: float
    right_ear: float
    avg_ear: float
    left_eye_open: bool
    right_eye_open: bool
    blink_detected: bool
    perclos: float


class MouthMetricsResponse(BaseModel):
    mar: float
    yawn_detected: bool


class HeadPoseResponse(BaseModel):
    pitch: float
    yaw: float
    roll: float
    forward_vector: List[float]


class InferenceResponse(BaseModel):
    frame_id: int
    timestamp: float
    face_detected: bool
    fatigue_state: FatigueStateEnum
    confidence: float
    eye_metrics: Optional[EyeMetricsResponse] = None
    mouth_metrics: Optional[MouthMetricsResponse] = None
    head_pose: Optional[HeadPoseResponse] = None
    processing_time_ms: float
    model_inference_time_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    uptime_seconds: float
    active_connections: int


class ConfigResponse(BaseModel):
    model: Dict
    vision: Dict
    inference: Dict
    api: Dict


# Load config early for middleware setup
_app_config = load_config("config/config.yaml")


# ============================================================================
# Global State
# ============================================================================

@dataclass
class AppState:
    model: Optional[HybridFatigueModel] = None
    vision_processor: Optional[VisionProcessor] = None
    config: Dict = None
    device: torch.device = torch.device("cpu")
    start_time: float = time.time()
    active_sessions: Dict[str, Dict] = field(default_factory=dict)


app_state = AppState()


# ============================================================================
# Lifespan Management
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown."""
    # Startup
    logger.info("Starting Fatigue Monitoring API...")

    # Load configuration
    app_state.config = load_config("config/config.yaml")

    # Determine device
    device_config = app_state.config.get("inference", {}).get("device", "auto")
    if device_config == "auto":
        if torch.cuda.is_available():
            app_state.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            app_state.device = torch.device("mps")
        else:
            app_state.device = torch.device("cpu")
    else:
        app_state.device = torch.device(device_config)

    logger.info(f"Using device: {app_state.device}")

    # Load model
    model_config = app_state.config.get("model", {})
    model_config["physiological"] = model_config.get("physiological", {})
    model_config["physiological"]["feature_dim"] = 32

    app_state.model = create_model(model_config)
    app_state.model.to(app_state.device)
    app_state.model.eval()

    # Load checkpoint if exists
    checkpoint_path = "models/checkpoints/best_model.pt"
    try:
        checkpoint = torch.load(checkpoint_path, map_location=app_state.device)
        app_state.model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded model checkpoint from {checkpoint_path}")
    except FileNotFoundError:
        logger.warning(f"No checkpoint found at {checkpoint_path}, using random weights")

    # Initialize vision processor
    app_state.vision_processor = VisionProcessor(app_state.config, str(app_state.device))

    logger.info("API startup complete")

    yield

    # Shutdown
    logger.info("Shutting down...")
    if app_state.vision_processor:
        app_state.vision_processor.close()
    logger.info("Shutdown complete")


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="Driver Fatigue Monitoring API",
    description="Real-time cognitive workload and fatigue monitoring using hybrid CNN-LSTM-Transformer",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=_app_config.get("api", {}).get("cors_origins", ["*"]),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Helper Functions
# ============================================================================

def get_app_state() -> AppState:
    return app_state


def frame_to_tensor(frame: np.ndarray, vision_processor: VisionProcessor) -> torch.Tensor:
    """Convert frame to model input tensor."""
    return vision_processor.preprocess_for_model(frame)


def tensor_to_base64(tensor: torch.Tensor) -> str:
    """Convert tensor to base64 encoded JPEG."""
    # Denormalize
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = tensor.squeeze(0) * std + mean
    img = (img.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()

    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buffer).decode('utf-8')


async def run_inference(
    frames: torch.Tensor,
    physio_features: Optional[torch.Tensor] = None,
) -> ModelOutput:
    """Run model inference asynchronously."""
    loop = asyncio.get_event_loop()

    def _inference():
        with torch.no_grad():
            frames = frames.to(app_state.device)
            if physio_features is not None:
                physio_features = physio_features.to(app_state.device)
            return app_state.model(frames, physio_features)

    return await loop.run_in_executor(None, _inference)


def vision_result_to_response(
    result: VisionFrameResult,
    model_output: Optional[ModelOutput] = None,
    model_time_ms: float = 0.0,
) -> InferenceResponse:
    """Convert VisionFrameResult to API response."""
    response = InferenceResponse(
        frame_id=result.frame_id,
        timestamp=result.timestamp,
        face_detected=result.face_detected,
        fatigue_state=FatigueStateEnum(result.fatigue_state.value),
        confidence=result.confidence,
        processing_time_ms=result.preprocessing_time_ms,
        model_inference_time_ms=model_time_ms,
    )

    if result.eye_metrics:
        response.eye_metrics = EyeMetricsResponse(
            left_ear=result.eye_metrics.left_ear,
            right_ear=result.eye_metrics.right_ear,
            avg_ear=result.eye_metrics.avg_ear,
            left_eye_open=result.eye_metrics.left_eye_open,
            right_eye_open=result.eye_metrics.right_eye_open,
            blink_detected=result.eye_metrics.blink_detected,
            perclos=result.eye_metrics.perclos,
        )

    if result.mouth_metrics:
        response.mouth_metrics = MouthMetricsResponse(
            mar=result.mouth_metrics.mar,
            yawn_detected=result.mouth_metrics.yawn_detected,
        )

    if result.head_pose:
        response.head_pose = HeadPoseResponse(
            pitch=result.head_pose.pitch,
            yaw=result.head_pose.yaw,
            roll=result.head_pose.roll,
            forward_vector=result.head_pose.forward_vector.tolist(),
        )

    # Override with model prediction if available
    if model_output is not None:
        fatigue_probs = F.softmax(model_output.fatigue_logits, dim=-1)
        pred_class = fatigue_probs.argmax(dim=-1).item()
        response.fatigue_state = FatigueStateEnum(
            ["alert", "drowsy", "fatigued"][pred_class]
        )
        response.confidence = fatigue_probs.max().item()

    return response


# ============================================================================
# REST Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check(state: AppState = Depends(get_app_state)):
    """Health check endpoint."""
    INFERENCE_REQUESTS.labels(endpoint="health", status="success").inc()
    return HealthResponse(
        status="healthy" if state.model is not None else "degraded",
        model_loaded=state.model is not None,
        device=str(state.device),
        uptime_seconds=time.time() - state.start_time,
        active_connections=len(state.active_sessions),
    )


@app.get("/config", response_model=ConfigResponse)
async def get_config(state: AppState = Depends(get_app_state)):
    """Get current configuration."""
    return ConfigResponse(
        model=state.config.get("model", {}),
        vision=state.config.get("vision", {}),
        inference=state.config.get("inference", {}),
        api=state.config.get("api", {}),
    )


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/inference/frame", response_model=InferenceResponse)
async def inference_single_frame(
    frame_data: str = Body(..., description="base64 encoded JPEG image"),
    state: AppState = Depends(get_app_state),
):
    """
    Run inference on a single frame (REST endpoint).
    Expects base64 encoded JPEG image.
    """
    start_time = time.perf_counter()
    INFERENCE_REQUESTS.labels(endpoint="inference_frame", status="started").inc()

    try:
        # Decode base64 image
        img_data = base64.b64decode(frame_data)
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image data")

        # Process with vision processor
        vision_result = state.vision_processor.process_frame(frame)

        # Prepare tensor for model
        frame_tensor = frame_to_tensor(frame, state.vision_processor).unsqueeze(0)  # [1, 1, 3, 224, 224]

        # Run model inference
        model_start = time.perf_counter()
        model_output = await run_inference(frame_tensor)
        model_time = (time.perf_counter() - model_start) * 1000

        # Convert to response
        response = vision_result_to_response(vision_result, model_output, model_time)

        FRAMES_PROCESSED.labels(source="rest").inc()
        FATIGUE_STATE_DISTRIBUTION.labels(state=response.fatigue_state.value).inc()
        INFERENCE_LATENCY.labels(endpoint="inference_frame").observe(time.perf_counter() - start_time)
        INFERENCE_REQUESTS.labels(endpoint="inference_frame", status="success").inc()

        return response

    except Exception as e:
        logger.error(f"Inference error: {e}")
        INFERENCE_REQUESTS.labels(endpoint="inference_frame", status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/inference/sequence", response_model=List[InferenceResponse])
async def inference_sequence(
    frames_data: List[str],  # List of base64 encoded JPEGs
    physio_data: Optional[List[List[float]]] = None,  # Physiological features per frame
    state: AppState = Depends(get_app_state),
):
    """
    Run inference on a sequence of frames for temporal modeling.
    """
    start_time = time.perf_counter()
    INFERENCE_REQUESTS.labels(endpoint="inference_sequence", status="started").inc()

    try:
        if len(frames_data) == 0:
            raise HTTPException(status_code=400, detail="No frames provided")

        sequence_length = state.config.get("inference", {}).get("sequence_length", 30)
        if len(frames_data) > sequence_length:
            frames_data = frames_data[-sequence_length:]
            if physio_data:
                physio_data = physio_data[-sequence_length:]

        # Process all frames
        frames = []
        vision_results = []

        for i, frame_data in enumerate(frames_data):
            img_data = base64.b64decode(frame_data)
            nparr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                continue

            vision_result = state.vision_processor.process_frame(frame)
            vision_results.append(vision_result)

            frame_tensor = frame_to_tensor(frame, state.vision_processor)
            frames.append(frame_tensor)

        if not frames:
            raise HTTPException(status_code=400, detail="No valid frames")

        # Stack frames [T, 3, H, W] -> [1, T, 3, H, W]
        frames_tensor = torch.stack(frames).unsqueeze(0)

        # Prepare physiological features
        physio_tensor = None
        if physio_data:
            physio_tensor = torch.tensor(physio_data, dtype=torch.float32).unsqueeze(0)  # [1, T, D]

        # Run model inference
        model_start = time.perf_counter()
        model_output = await run_inference(frames_tensor, physio_tensor)
        model_time = (time.perf_counter() - model_start) * 1000

        # Convert to responses
        responses = []
        for i, vision_result in enumerate(vision_results):
            response = vision_result_to_response(vision_result, model_output, model_time / len(vision_results))
            responses.append(response)
            FRAMES_PROCESSED.labels(source="sequence").inc()
            FATIGUE_STATE_DISTRIBUTION.labels(state=response.fatigue_state.value).inc()

        INFERENCE_LATENCY.labels(endpoint="inference_sequence").observe(time.perf_counter() - start_time)
        INFERENCE_REQUESTS.labels(endpoint="inference_sequence", status="success").inc()

        return responses

    except Exception as e:
        logger.error(f"Sequence inference error: {e}")
        INFERENCE_REQUESTS.labels(endpoint="inference_sequence", status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# WebSocket Endpoint
# ============================================================================

class ConnectionManager:
    """Manages active WebSocket connections."""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.session_data: Dict[str, Dict] = {}

    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket
        self.session_data[session_id] = {
            "frame_buffer": [],
            "physio_buffer": [],
            "frame_count": 0,
            "last_inference": 0,
        }
        ACTIVE_CONNECTIONS.set(len(self.active_connections))
        logger.info(f"Client connected: {session_id} (total: {len(self.active_connections)})")

    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]
        if session_id in self.session_data:
            del self.session_data[session_id]
        ACTIVE_CONNECTIONS.set(len(self.active_connections))
        logger.info(f"Client disconnected: {session_id} (total: {len(self.active_connections)})")

    async def send_json(self, session_id: str, data: Dict):
        if session_id in self.active_connections:
            try:
                await self.active_connections[session_id].send_json(data)
            except Exception as e:
                logger.error(f"Error sending to {session_id}: {e}")


manager = ConnectionManager()


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time video streaming and inference.

    Message format (client -> server):
    {
        "type": "frame",
        "data": "base64_jpeg",
        "timestamp": 1234567890.123,
        "physio": [0.1, 0.2, ...]  // optional physiological features
    }

    Message format (server -> client):
    {
        "type": "result",
        "frame_id": 123,
        "fatigue_state": "alert",
        "confidence": 0.95,
        "eye_metrics": {...},
        "processing_time_ms": 15.2
    }
    """
    await manager.connect(websocket, session_id)

    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            message = json.loads(data)

            if message.get("type") == "frame":
                await handle_frame_message(session_id, message)
            elif message.get("type") == "config":
                await handle_config_message(session_id, message)
            elif message.get("type") == "ping":
                await manager.send_json(session_id, {"type": "pong", "timestamp": time.time()})
            else:
                logger.warning(f"Unknown message type: {message.get('type')}")

    except WebSocketDisconnect:
        manager.disconnect(session_id)
    except Exception as e:
        logger.error(f"WebSocket error for {session_id}: {e}")
        manager.disconnect(session_id)


async def handle_frame_message(session_id: str, message: Dict):
    """Process incoming frame message."""
    session = manager.session_data.get(session_id)
    if not session:
        return

    frame_data = message.get("data")
    timestamp = message.get("timestamp", time.time())
    physio = message.get("physio")

    try:
        # Decode frame
        img_data = base64.b64decode(frame_data)
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            await manager.send_json(session_id, {"type": "error", "message": "Invalid frame data"})
            return

        # Vision processing
        vision_result = app_state.vision_processor.process_frame(frame)
        session["frame_count"] += 1

        # Buffer frames for temporal inference
        frame_tensor = frame_to_tensor(frame, app_state.vision_processor)
        session["frame_buffer"].append(frame_tensor)
        if physio:
            session["physio_buffer"].append(torch.tensor(physio, dtype=torch.float32))

        # Keep buffer at sequence length
        seq_len = app_state.config.get("inference", {}).get("sequence_length", 30)
        if len(session["frame_buffer"]) > seq_len:
            session["frame_buffer"].pop(0)
            if session["physio_buffer"]:
                session["physio_buffer"].pop(0)

        # Run model inference periodically (every N frames)
        inference_interval = app_state.config.get("inference", {}).get("inference_interval", 5)
        if session["frame_count"] % inference_interval == 0 and len(session["frame_buffer"]) >= 5:
            model_start = time.perf_counter()

            frames_tensor = torch.stack(session["frame_buffer"]).unsqueeze(0)
            physio_tensor = None
            if session["physio_buffer"]:
                physio_tensor = torch.stack(session["physio_buffer"]).unsqueeze(0)

            model_output = await run_inference(frames_tensor, physio_tensor)
            model_time = (time.perf_counter() - model_start) * 1000

            # Send result with model prediction
            response = vision_result_to_response(vision_result, model_output, model_time)
            await manager.send_json(session_id, {"type": "result", **response.model_dump()})

            # Update metrics
            FRAMES_PROCESSED.labels(source="websocket").inc()
            FATIGUE_STATE_DISTRIBUTION.labels(state=response.fatigue_state.value).inc()
        else:
            # Send vision-only result
            response = vision_result_to_response(vision_result)
            await manager.send_json(session_id, {"type": "result", **response.model_dump()})

    except Exception as e:
        logger.error(f"Frame processing error: {e}")
        await manager.send_json(session_id, {"type": "error", "message": str(e)})


async def handle_config_message(session_id: str, message: Dict):
    """Handle configuration update message."""
    config_updates = message.get("config", {})
    # Update vision processor config dynamically
    if "ear_threshold" in config_updates:
        app_state.vision_processor.ear_threshold = config_updates["ear_threshold"]
    if "perclos_window" in config_updates:
        app_state.vision_processor.perclos_config["window_size"] = config_updates["perclos_window"]
        app_state.vision_processor.ear_history = deque(
            app_state.vision_processor.ear_history,
            maxlen=config_updates["perclos_window"]
        )

    await manager.send_json(session_id, {"type": "config_ack", "config": config_updates})


# ============================================================================
# Background Tasks
# ============================================================================

@app.post("/session/start")
async def start_session(
    background_tasks: BackgroundTasks,
    source: str = "webcam",
    config: Optional[Dict] = None,
    state: AppState = Depends(get_app_state),
):
    """Start a new monitoring session."""
    session_id = str(uuid.uuid4())
    session_config = config or {}

    state.active_sessions[session_id] = {
        "source": source,
        "config": session_config,
        "started_at": time.time(),
        "frame_count": 0,
    }

    logger.info(f"Started session {session_id} with source {source}")
    return {"session_id": session_id, "status": "started"}


@app.post("/session/{session_id}/stop")
async def stop_session(session_id: str, state: AppState = Depends(get_app_state)):
    """Stop a monitoring session."""
    if session_id in state.active_sessions:
        session = state.active_sessions.pop(session_id)
        logger.info(f"Stopped session {session_id} after {session['frame_count']} frames")
        return {"status": "stopped", "frames_processed": session["frame_count"]}
    raise HTTPException(status_code=404, detail="Session not found")


@app.get("/session/{session_id}/status")
async def session_status(session_id: str, state: AppState = Depends(get_app_state)):
    """Get session status."""
    if session_id in state.active_sessions:
        session = state.active_sessions[session_id]
        return {
            "session_id": session_id,
            "source": session["source"],
            "frame_count": session["frame_count"],
            "duration_seconds": time.time() - session["started_at"],
        }
    raise HTTPException(status_code=404, detail="Session not found")


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    config = load_config("config/config.yaml")
    api_config = config.get("api", {})

    uvicorn.run(
        "inference.main:app",
        host=api_config.get("host", "0.0.0.0"),
        port=api_config.get("port", 8000),
        workers=api_config.get("workers", 1),
        reload=True,
        log_level="info",
    )