# Driver Fatigue Monitoring System

A production-grade, real-time cognitive workload and fatigue monitoring system that fuses visual cues (webcam/facial landmarks) with physiological time-series data using a hybrid CNN-LSTM-Transformer architecture.

## 🏗️ Architecture Overview

```
┌─────────────────┐     ┌──────────────────┐     ┌────────────────────┐
│  Video Stream   │     │  Physiological   │     │   Model Input      │
│  (Webcam/RTSP)  │     │  Signals (ECG,   │     │  [B, T, C, H, W]   │
└────────┬────────┘     │  EDA, RESP, TEMP)│     │  [B, T, D_physio]  │
         │              └────────┬─────────┘     └────────┬───────────┘
         ▼                       ▼                        ▼
┌─────────────────┐     ┌──────────────────┐     ┌────────────────────┐
│ MediaPipe Face  │     │  Physio          │     │  Spatial CNN       │
│ Mesh (468 lmks) │     │  Preprocessor    │     │  (EfficientNet)    │
└────────┬────────┘     └────────┬─────────┘     └────────┬───────────┘
         │                       │                        │
         ▼                       ▼                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Temporal Encoder (LSTM/TCN)                  │
│                    [B, T, 2*CNN_dim] → [B, T, H]               │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Transformer Fusion Layer                       │
│         Cross-modal attention: Visual ↔ Physiological          │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────┬─────────────────────┬─────────────────────────┐
│  Fatigue Head   │   Workload Head     │      Blink Head         │
│  (3-class)      │   (Regression)      │      (Binary)           │
└─────────────────┴─────────────────────┴─────────────────────────┘
```

## 📁 Project Structure

```
driver-fatigue-monitor/
├── config/
│   └── config.yaml              # Main configuration file
├── data/
│   ├── __init__.py
│   ├── preprocessing.py         # Data augmentation & physio feature extraction
│   └── dataset.py               # PyTorch Dataset classes
├── models/
│   ├── __init__.py
│   └── hybrid_model.py          # CNN-LSTM-Transformer model
├── inference/
│   ├── __init__.py
│   ├── vision_processor.py      # MediaPipe face processing & PERCLOS
│   └── main.py                  # FastAPI backend with WebSocket
├── frontend/
│   ├── __init__.py
│   └── streamlit_app.py         # Streamlit dashboard
├── scripts/
│   └── train.py                 # Training script
├── utils/
│   ├── __init__.py
│   ├── config_loader.py         # YAML config loader
│   └── logger.py                # Loguru setup
├── tests/
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (for GPU acceleration)
- Docker & Docker Compose (for containerized deployment)

### Local Development Setup

```bash
# Clone and navigate
cd driver-fatigue-monitor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Run with Docker (Recommended)

```bash
# Build and start all services
docker-compose up --build

# Or run individually
docker-compose up api frontend
```

Services will be available at:
- **API**: http://localhost:8000
- **Frontend**: http://localhost:8501
- **API Docs**: http://localhost:8000/docs
- **Prometheus** (optional): http://localhost:9090
- **Grafana** (optional): http://localhost:3000 (admin/admin)

### Run Locally (Without Docker)

```bash
# Terminal 1: Start API server
python -m inference.main

# Terminal 2: Start Streamlit frontend
streamlit run frontend/streamlit_app.py
```

## ⚙️ Configuration

Edit `config/config.yaml` to customize:

```yaml
model:
  cnn:
    backbone: "efficientnet_b0"  # efficientnet_b0, mobilenet_v3, resnet18
    output_dim: 256
  temporal:
    type: "lstm"  # lstm, gru, tcn
    hidden_dim: 128
  transformer:
    d_model: 256
    nhead: 8
    num_encoder_layers: 3

vision:
  perclos:
    ear_threshold: 0.25
    window_size: 60  # frames

inference:
  device: "auto"  # auto, cpu, cuda, mps
  sequence_length: 30
```

## 📡 API Endpoints

### REST API

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/config` | Get current configuration |
| GET | `/metrics` | Prometheus metrics |
| POST | `/inference/frame` | Single frame inference |
| POST | `/inference/sequence` | Sequence inference |
| POST | `/session/start` | Start monitoring session |
| POST | `/session/{id}/stop` | Stop session |
| GET | `/session/{id}/status` | Get session status |

### WebSocket API

Connect to `ws://localhost:8000/ws/{session_id}`

**Client → Server:**
```json
{
  "type": "frame",
  "data": "base64_jpeg_string",
  "timestamp": 1234567890.123,
  "physio": [0.1, 0.2, 0.3, ...]
}
```

**Server → Client:**
```json
{
  "type": "result",
  "frame_id": 123,
  "fatigue_state": "alert",
  "confidence": 0.95,
  "eye_metrics": { "avg_ear": 0.32, "perclos": 0.05, "blink_detected": false },
  "mouth_metrics": { "mar": 0.2, "yawn_detected": false },
  "head_pose": { "pitch": 2.1, "yaw": -1.5, "roll": 0.3 },
  "processing_time_ms": 12.5
}
```

## 🧪 Training

Prepare your data in the following structure:

```
data/
├── train.txt          # List of session directories
├── val.txt
└── subject_001/
    └── session_001/
        ├── frames/
        │   ├── frame_000001.jpg
        │   └── ...
        ├── annotations.json
        └── physio/
            ├── ecg.csv
            ├── eda.csv
            └── resp.csv
```

Run training:

```bash
python scripts/train.py \
  --data-root ./data \
  --epochs 100 \
  --batch-size 16 \
  --sequence-length 30 \
  --lr 1e-4 \
  --checkpoint-dir ./models/checkpoints
```

## 📊 Key Features

### Vision Processing (MediaPipe)
- **468 facial landmarks** with sub-pixel accuracy
- **Eye Aspect Ratio (EAR)** for blink detection
- **PERCLOS** (Percentage of Eyelid Closure) over sliding window
- **Head pose estimation** (pitch, yaw, roll) via PnP
- **Yawn detection** via Mouth Aspect Ratio (MAR)
- Real-time annotated video output

### Hybrid Model
- **Spatial CNN**: EfficientNet/MobileNet/ResNet backbone
- **Temporal Encoder**: LSTM/GRU/TCN for sequence modeling
- **Transformer Fusion**: Cross-modal attention between visual & physiological streams
- **Multi-task heads**: Fatigue classification, workload regression, blink detection

### Production Ready
- **FastAPI** with async inference & WebSocket streaming
- **Prometheus metrics** for monitoring
- **Docker** multi-stage build with GPU support
- **Loguru** structured logging
- **Pydantic** request/response validation
- **CORS** configured for frontend integration

## 📈 Monitoring & Metrics

The system exposes Prometheus metrics at `/metrics`:

- `fatigue_inference_requests_total` - Request counts by endpoint/status
- `fatigue_inference_latency_seconds` - Inference latency histogram
- `fatigue_active_websocket_connections` - Active WebSocket connections
- `fatigue_frames_processed_total` - Frames processed by source
- `fatigue_state_total` - Fatigue state distribution

## 🔧 Development

### Code Quality

```bash
# Format code
black .
isort .

# Type checking
mypy .

# Run tests
pytest tests/ -v --cov
```

### Pre-commit Hooks

```bash
pre-commit install
pre-commit run --all-files
```

## 📝 License

MIT License - See LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## 📚 References

- MediaPipe Face Mesh: https://github.com/google/mediapipe
- PERCLOS: Wierwille et al., "Research on vehicle-based driver status/performance monitoring"
- EfficientNet: Tan & Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"
- Transformer: Vaswani et al., "Attention Is All You Need"