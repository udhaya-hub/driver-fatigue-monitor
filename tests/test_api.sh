#!/bin/bash
# Quick API Test Script using curl
# Run from project root: bash tests/test_api.sh

set -e

API_URL="${API_URL:-http://localhost:8000}"
TIMEOUT=10

echo "=========================================="
echo "  Driver Fatigue Monitor - API Tests"
echo "  Target: $API_URL"
echo "=========================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

pass_count=0
fail_count=0

test() {
    local name="$1"
    local cmd="$2"
    local expect="$3"
    
    echo -n "Testing $name... "
    if output=$(eval "$cmd" 2>/dev/null); then
        if [[ -n "$expect" ]] && ! echo "$output" | grep -q "$expect"; then
            echo -e "${RED}FAIL${NC} (unexpected response)"
            echo "  Expected: $expect"
            echo "  Got: $output"
            ((fail_count++))
        else
            echo -e "${GREEN}PASS${NC}"
            ((pass_count++))
        fi
    else
        echo -e "${RED}FAIL${NC} (command failed)"
        ((fail_count++))
    fi
}

# 1. Health Check
echo "--- Core Endpoints ---"
test "Health Check" \
    "curl -s -f --max-time $TIMEOUT $API_URL/health" \
    '"status":"healthy"'

test "Config Endpoint" \
    "curl -s -f --max-time $TIMEOUT $API_URL/config" \
    '"model"'

test "Metrics Endpoint" \
    "curl -s -f --max-time $TIMEOUT $API_URL/metrics" \
    "fatigue_inference"

# 2. Session Lifecycle
echo ""
echo "--- Session Lifecycle ---"

# Start session
echo -n "Testing Session Start... "
SESSION_RESPONSE=$(curl -s -f --max-time $TIMEOUT -X POST "$API_URL/session/start" \
    -H "Content-Type: application/json" -d '{"source":"test"}')
if echo "$SESSION_RESPONSE" | grep -q "session_id"; then
    SESSION_ID=$(echo "$SESSION_RESPONSE" | sed -n 's/.*"session_id":"\([^"]*\)".*/\1/p')
    echo -e "${GREEN}PASS${NC} (session_id: ${SESSION_ID:0:8}...)"
    ((pass_count++))
else
    echo -e "${RED}FAIL${NC}"
    echo "  Response: $SESSION_RESPONSE"
    ((fail_count++))
    exit 1
fi

# Session Status
test "Session Status" \
    "curl -s -f --max-time $TIMEOUT $API_URL/session/$SESSION_ID/status" \
    "frame_count"

# 3. Create test frame
echo ""
echo "--- Inference Tests ---"
echo -n "Creating test frame... "
python3 -c "
import cv2, numpy as np, base64
frame = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.rectangle(frame, (200, 150), (440, 350), (200, 200, 200), -1)
cv2.circle(frame, (280, 220), 30, (100, 100, 100), -1)
cv2.circle(frame, (360, 220), 30, (100, 100, 100), -1)
cv2.ellipse(frame, (320, 300), (40, 20), 0, 0, 180, (100, 100, 100), -1)
_, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
print(base64.b64encode(buf).decode('utf-8'))
" > /tmp/test_frame.b64
FRAME_B64=$(cat /tmp/test_frame.b64)
echo -e "${GREEN}DONE${NC}"

# Single Frame Inference
echo -n "Testing Single Frame Inference... "
INFERENCE_RESPONSE=$(curl -s -f --max-time $TIMEOUT -X POST "$API_URL/inference/frame" \
    -H "Content-Type: application/json" \
    -d "{\"frame_data\":\"$FRAME_B64\"}")
if echo "$INFERENCE_RESPONSE" | grep -q "fatigue_state"; then
    STATE=$(echo "$INFERENCE_RESPONSE" | sed -n 's/.*"fatigue_state":"\([^"]*\)".*/\1/p')
    CONF=$(echo "$INFERENCE_RESPONSE" | sed -n 's/.*"confidence":\([0-9.]*\).*/\1/p')
    echo -e "${GREEN}PASS${NC} (state=$STATE, conf=$CONF)"
    ((pass_count++))
else
    echo -e "${RED}FAIL${NC}"
    echo "  Response: $INFERENCE_RESPONSE"
    ((fail_count++))
fi

# Sequence Inference (3 frames)
echo -n "Testing Sequence Inference... "
SEQ_RESPONSE=$(curl -s -f --max-time $TIMEOUT -X POST "$API_URL/inference/sequence" \
    -H "Content-Type: application/json" \
    -d "{\"frames_data\":[\"$FRAME_B64\",\"$FRAME_B64\",\"$FRAME_B64\"]}")
if echo "$SEQ_RESPONSE" | grep -q '\['; then
    COUNT=$(echo "$SEQ_RESPONSE" | grep -o 'fatigue_state' | wc -l)
    echo -e "${GREEN}PASS${NC} (returned $COUNT results)"
    ((pass_count++))
else
    echo -e "${RED}FAIL${NC}"
    echo "  Response: $SEQ_RESPONSE"
    ((fail_count++))
fi

# 4. Stop Session
echo ""
echo "--- Session Cleanup ---"
test "Session Stop" \
    "curl -s -f --max-time $TIMEOUT -X POST $API_URL/session/$SESSION_ID/stop" \
    "frames_processed"

# 5. Error Handling
echo ""
echo "--- Error Handling ---"
test "Invalid Frame Data" \
    "curl -s -w '%{http_code}' -o /dev/null --max-time $TIMEOUT -X POST $API_URL/inference/frame -H 'Content-Type: application/json' -d '{\"frame_data\":\"invalid\"}'" \
    "400\|500"

test "Invalid Session Stop" \
    "curl -s -w '%{http_code}' -o /dev/null --max-time $TIMEOUT -X POST $API_URL/session/invalid-id/stop" \
    "404"

# Summary
echo ""
echo "=========================================="
echo "  Results: ${GREEN}$pass_count passed${NC}, ${RED}$fail_count failed${NC}"
echo "=========================================="

if [ $fail_count -eq 0 ]; then
    echo -e "${GREEN}🎉 All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}⚠️  Some tests failed${NC}"
    exit 1
fi