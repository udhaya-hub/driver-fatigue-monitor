# Driver Fatigue Monitor - Complete Testing Checklist

## 📋 Overview
This checklist validates the entire stack: FastAPI backend → Streamlit frontend → Model inference → Real-time UI updates.

---

## 1. Backend API Tests (`http://localhost:8000`)

### 1.1 Core Endpoints
| Test | Command | Expected | ✅ |
|------|---------|----------|----|
| Health Check | `curl -s http://localhost:8000/health` | `{"status":"healthy","model_loaded":true,"device":"cpu"}` | ☐ |
| Config Endpoint | `curl -s http://localhost:8000/config` | Contains `model`, `vision`, `inference`, `api` keys | ☐ |
| Metrics Endpoint | `curl -s http://localhost:8000/metrics` | Prometheus format with `fatigue_*` metrics | ☐ |

### 1.2 Session Lifecycle
| Test | Command | Expected | ✅ |
|------|---------|----------|----|
| Start Session | `curl -X POST http://localhost:8000/session/start -d '{"source":"test"}'` | Returns `session_id` | ☐ |
| Session Status | `curl http://localhost:8000/session/{id}/status` | Returns `frame_count`, `duration` | ☐ |
| Stop Session | `curl -X POST http://localhost:8000/session/{id}/stop` | Returns `frames_processed` | ☐ |

### 1.3 Inference Endpoints
| Test | Command | Expected | ✅ |
|------|---------|----------|----|
| Single Frame | `curl -X POST /inference/frame -d '{"frame_data":"<base64>"}'` | `fatigue_state`, `confidence`, `eye_metrics`, `mouth_metrics`, `head_pose` | ☐ |
| Sequence | `curl -X POST /inference/sequence -d '{"frames_data":["<b64>","<b64>"]}'` | Array of results matching frame count | ☐ |

### 1.4 Error Handling
| Test | Command | Expected | ✅ |
|------|---------|----------|----|
| Invalid Frame | `curl -X POST /inference/frame -d '{"frame_data":"bad"}'` | HTTP 400/500 | ☐ |
| Invalid Session | `curl -X POST /session/bad/stop` | HTTP 404 | ☐ |

---

## 2. Automated Test Execution

### Run Python Test Suite
```bash
cd driver-fatigue-monitor
python tests/test_api.py --verbose
```

### Run Bash/curl Tests
```bash
cd driver-fatigue-monitor
bash tests/test_api.sh
```

### Custom URL
```bash
python tests/test_api.py --url http://192.168.1.50:8000
```

---

## 3. Frontend Integration Tests (`http://localhost:8501`)

### 3.1 Page Load & Initialization
| Step | Action | Expected | ✅ |
|------|--------|----------|----|
| 1 | Open `http://localhost:8501` | Page loads, title "🚗 Driver Fatigue Monitor" | ☐ |
| 2 | Check sidebar | "🎛️ Control Panel" visible | ☐ |
| 3 | Backend status | Shows "🟢 Online" (if backend running) | ☐ |
| 4 | Video Source radio | "WebRTC (Browser Camera)" & "OpenCV (Local Camera)" | ☐ |
| 5 | Start button | "🟢 Start Monitoring" enabled | ☐ |
| 6 | Instructions expander | Opens with usage guide | ☐ |

### 3.2 WebRTC Mode (Browser Camera)
| Step | Action | Expected | ✅ |
|------|--------|----------|----|
| 1 | Select "WebRTC (Browser Camera)" | Radio selected | ☐ |
| 2 | Click "🟢 Start Monitoring" | Button changes to "🔴 Stop Monitoring" | ☐ |
| 3 | Browser prompts | "Allow camera access?" | ☐ |
| 4 | Allow camera | Video feed appears in main panel | ☐ |
| 5 | Overlays visible | EAR, PERCLOS, State badge on video | ☐ |
| 6 | Telemetry panel updates | Metric cards show live values | ☐ |
| 7 | Charts render | Plotly charts appear below metrics | ☐ |
| 8 | Click "🔴 Stop Monitoring" | Feed stops, button reverts | ☐ |

### 3.3 OpenCV Mode (Local Camera)
| Step | Action | Expected | ✅ |
|------|--------|----------|----|
| 1 | Select "OpenCV (Local Camera)" | Radio selected | ☐ |
| 2 | Camera index dropdown | Options: Camera 0, 1, 2 | ☐ |
| 3 | Click "🟢 Start Monitoring" | Local camera feed appears | ☐ |
| 4 | Overlays on feed | Same as WebRTC mode | ☐ |
| 5 | Telemetry updates | Real-time metric cards | ☐ |
| 6 | Click "🔴 Stop Monitoring" | Feed stops, "Session stopped" | ☐ |

### 3.4 Real-Time Data Validation
| Metric | Expected Range | Visual Indicator | ✅ |
|--------|----------------|------------------|----|
| EAR | 0.0 - 0.5 | Green if >0.25, Red if <0.25 | ☐ |
| PERCLOS | 0.0 - 1.0 | Green if <0.15, Amber if >0.15 | ☐ |
| Blink Rate | 0 - 50/min | Updates every few seconds | ☐ |
| Head Pose | -45° to +45° | Pitch/Yaw/Roll displayed | ☐ |
| Fatigue State | Alert/Drowsy/Fatigued | Large badge with color | ☐ |
| Confidence | 0.0 - 1.0 | Shown in state badge | ☐ |
| FPS | 10 - 30 | Bottom-right of video | ☐ |

### 3.5 Alert System
| Scenario | Trigger | Expected UI | ✅ |
|----------|---------|-------------|----|
| Drowsy | PERCLOS > 0.15 | 🟡 Amber banner "Drowsiness Detected" | ☐ |
| Fatigued | PERCLOS > 0.3 or EAR < 0.2 | 🔴 Red banner "High Fatigue Risk" | ☐ |
| Blink | EAR < 0.25 for 3 frames | "BLINK!" on video overlay | ☐ |
| Yawn | MAR > 0.6 | "YAWN!" on video overlay | ☐ |

### 3.6 Charts & Visualizations
| Chart | Data | Updates | ✅ |
|-------|------|---------|----|
| EAR & PERCLOS | Time series | Live scrolling | ☐ |
| Head Pose | Pitch/Yaw/Roll | Live scrolling | ☐ |
| Performance | FPS + Latency | Live scrolling | ☐ |
| Fatigue Distribution | Donut chart | Updates per frame | ☐ |

---

## 4. End-to-End Integration Test

### Full Flow Test
```bash
# Terminal 1: Start Backend
cd driver-fatigue-monitor
python -m uvicorn inference.main:app --host 0.0.0.0 --port 8000

# Terminal 2: Start Frontend
cd driver-fatigue-monitor
streamlit run frontend/app.py --server.headless true

# Terminal 3: Run API Tests
cd driver-fatigue-monitor
python tests/test_api.py --verbose
```

### Manual E2E Validation
| Step | Action | Verify | ✅ |
|------|--------|--------|----|
| 1 | Backend starts | `http://localhost:8000/health` → healthy | ☐ |
| 2 | Frontend starts | `http://localhost:8501` loads | ☐ |
| 3 | Frontend shows backend online | Sidebar: "🟢 Online" | ☐ |
| 4 | Start session (WebRTC) | Camera permission prompt | ☐ |
| 5 | Allow camera | Video feed + overlays | ☐ |
| 6 | Telemetry panel | 4 metric cards + charts | ☐ |
| 7 | Make drowsy face (close eyes) | PERCLOS rises, state → Drowsy | ☐ |
| 8 | Alert banner appears | "⚠️ Drowsiness Detected" | ☐ |
| 9 | Stop session | Feed stops, stats shown | ☐ |
| 10 | Run API tests | All pass | ☐ |

---

## 5. Error Resilience Tests

### 5.1 Backend Offline During Session
| Step | Action | Expected Frontend Behavior | ✅ |
|------|--------|----------------------------|----|
| 1 | Start session, video running | Normal operation | ☐ |
| 2 | Kill backend (Ctrl+C Terminal 1) | Frontend shows error toasts | ☐ |
| 3 | Backend status in sidebar | Changes to "🔴 Offline" | ☐ |
| 4 | Video feed | Shows "Backend offline" message | ☐ |
| 5 | Restart backend | Auto-recovers on next frame | ☐ |

### 5.2 Camera Disconnect
| Step | Action | Expected | ✅ |
|------|--------|----------|----|
| 1 | Unplug camera during OpenCV mode | Graceful error, no crash | ☐ |
| 2 | Plug back in, restart session | Works again | ☐ |

### 5.3 Network Issues
| Test | Method | Expected | ✅ |
|------|--------|----------|----|
| High latency | `tc qdisc add dev lo root netem delay 500ms` | Timeouts handled gracefully | ☐ |
| Packet loss | `tc qdisc add dev lo root netem loss 10%` | Retries or errors shown | ☐ |

---

## 6. Performance Benchmarks

| Metric | Target | Measurement | ✅ |
|--------|--------|-------------|----|
| API Latency (P95) | < 200ms | `/inference/frame` | ☐ |
| WebRTC Latency | < 500ms | Frame → overlay | ☐ |
| OpenCV FPS | ≥ 20 FPS | Local camera | ☐ |
| Memory (Backend) | < 2GB | `docker stats` | ☐ |
| Memory (Frontend) | < 500MB | Browser task manager | ☐ |
| CPU (Backend) | < 80% | 1 concurrent session | ☐ |

---

## 7. Cross-Browser Testing

| Browser | WebRTC | OpenCV | Notes |
|---------|--------|--------|-------|
| Chrome 120+ | ✅ | N/A | Best WebRTC support |
| Firefox 120+ | ✅ | N/A | Good WebRTC support |
| Edge 120+ | ✅ | N/A | Chromium-based |
| Safari 17+ | ⚠️ | N/A | May need HTTPS for camera |

---

## 8. CI/CD Integration

### GitHub Actions Example
```yaml
# .github/workflows/test.yml
name: API Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    services:
      redis:
        image: redis:7
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.11' }
      - run: pip install -r requirements.txt
      - run: python -m uvicorn inference.main:app --host 0.0.0.0 --port 8000 &
      - run: sleep 10 && python tests/test_api.py
```

---

## 9. Sign-Off

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Backend Engineer | | | |
| Frontend Engineer | | | |
| QA Engineer | | | |
| Project Lead | | | |

---

## Quick Commands Reference

```bash
# Start everything
make up          # docker-compose up -d
# OR manually:
python -m uvicorn inference.main:app --host 0.0.0.0 --port 8000 &
streamlit run frontend/app.py --server.headless true &

# Run tests
python tests/test_api.py --verbose
bash tests/test_api.sh

# Check logs
docker-compose logs -f api
docker-compose logs -f frontend

# Stop everything
make down        # docker-compose down
# OR
pkill -f "uvicorn\|streamlit"
```

---

*Generated for Driver Fatigue Monitor v1.0.0*
*Last Updated: $(date)*