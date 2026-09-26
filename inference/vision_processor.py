"""
Real-time Vision Processor for Driver Fatigue Monitoring.

This module handles:
1. Face detection and facial landmark extraction using MediaPipe Face Mesh
2. Eye Aspect Ratio (EAR) calculation for blink detection
3. PERCLOS (Percentage of Eyelid Closure) computation
4. Head pose estimation
5. Yawn detection via mouth aspect ratio
6. Real-time frame preprocessing for model inference
"""

import cv2
import numpy as np
import torch
import mediapipe as mp
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from collections import deque
import time
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class FatigueState(Enum):
    """Enumeration of fatigue states."""
    ALERT = "alert"
    DROWSY = "drowsy"
    FATIGUED = "fatigued"
    UNKNOWN = "unknown"


@dataclass
class EyeMetrics:
    """Container for eye-related metrics."""
    left_ear: float
    right_ear: float
    avg_ear: float
    left_eye_open: bool
    right_eye_open: bool
    blink_detected: bool
    perclos: float


@dataclass
class MouthMetrics:
    """Container for mouth-related metrics."""
    mar: float  # Mouth Aspect Ratio
    yawn_detected: bool


@dataclass
class HeadPose:
    """Container for head pose estimation."""
    pitch: float
    yaw: float
    roll: float
    forward_vector: np.ndarray


@dataclass
class VisionFrameResult:
    """Complete result for a single processed frame."""
    timestamp: float
    frame_id: int
    face_detected: bool
    landmarks: Optional[np.ndarray] = None  # [468, 3] normalized coordinates
    eye_metrics: Optional[EyeMetrics] = None
    mouth_metrics: Optional[MouthMetrics] = None
    head_pose: Optional[HeadPose] = None
    fatigue_state: FatigueState = FatigueState.UNKNOWN
    confidence: float = 0.0
    preprocessing_time_ms: float = 0.0


class VisionProcessor:
    """
    Real-time vision processor using MediaPipe Face Mesh.

    Features:
    - 468 facial landmarks with sub-pixel accuracy
    - Real-time EAR and PERCLOS calculation
    - Head pose estimation (pitch, yaw, roll)
    - Yawn detection
    - Blink detection with temporal smoothing
    - GPU acceleration support
    """

    # MediaPipe Face Mesh landmark indices
    LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
    MOUTH_INDICES = [61, 84, 17, 314, 405, 320, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324]
    POSE_INDICES = [1, 33, 61, 291, 199]  # Nose tip, left eye, left mouth, right mouth, chin

    def __init__(
        self,
        config: Dict,
        device: str = "cpu",
    ):
        """
        Initialize vision processor.

        Args:
            config: Configuration dictionary from config.yaml
            device: Compute device ('cpu', 'cuda', 'mps')
        """
        self.config = config
        self.device = device
        self.vision_config = config.get("vision", {})
        self.mediapipe_config = self.vision_config.get("mediapipe", {})
        self.perclos_config = self.vision_config.get("perclos", {})
        self.preprocessing_config = self.vision_config.get("preprocessing", {})

        # MediaPipe Face Mesh
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=self.mediapipe_config.get("max_num_faces", 1),
            refine_landmarks=self.mediapipe_config.get("refine_landmarks", True),
            min_detection_confidence=self.mediapipe_config.get("min_detection_confidence", 0.7),
            min_tracking_confidence=self.mediapipe_config.get("min_tracking_confidence", 0.7),
        )

        # Drawing utilities
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # PERCLOS tracking
        self.ear_threshold = self.perclos_config.get("ear_threshold", 0.25)
        self.consecutive_frames = self.perclos_config.get("consecutive_frames", 3)
        self.window_size = self.perclos_config.get("window_size", 60)

        # Blink detection state
        self.eye_closed_frames = 0
        self.blink_count = 0
        self.ear_history = deque(maxlen=self.window_size)
        self.blink_history = deque(maxlen=self.window_size)

        # Yawn detection
        self.mar_threshold = 0.6
        self.yawn_frames = 0

        # Head pose estimation
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left mouth corner
            (150.0, -150.0, -125.0),     # Right mouth corner
        ], dtype=np.float32)

        # Camera matrix (approximate, should be calibrated)
        self.camera_matrix = None
        self.dist_coeffs = np.zeros((4, 1))

        # Preprocessing
        self.target_size = tuple(self.preprocessing_config.get("target_size", [224, 224]))
        self.norm_mean = np.array(self.preprocessing_config.get("normalize_mean", [0.485, 0.456, 0.406]))
        self.norm_std = np.array(self.preprocessing_config.get("normalize_std", [0.229, 0.224, 0.225]))

        # Frame counter
        self.frame_id = 0

        logger.info("VisionProcessor initialized")

    def _calculate_ear(self, eye_landmarks: np.ndarray) -> float:
        """
        Calculate Eye Aspect Ratio (EAR).

        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

        Args:
            eye_landmarks: 6 landmarks for one eye [6, 3] (x, y, z)
        Returns:
            EAR value
        """
        # Vertical distances
        v1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
        v2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])

        # Horizontal distance
        h = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])

        if h == 0:
            return 0.0

        ear = (v1 + v2) / (2.0 * h)
        return float(ear)

    def _calculate_mar(self, mouth_landmarks: np.ndarray) -> float:
        """
        Calculate Mouth Aspect Ratio (MAR).

        MAR = (||p2-p10|| + ||p4-p8||) / (2 * ||p1-p6||)

        Args:
            mouth_landmarks: Mouth landmarks [N, 3]
        Returns:
            MAR value
        """
        # Use specific indices for MAR calculation
        # Vertical distances
        v1 = np.linalg.norm(mouth_landmarks[1] - mouth_landmarks[9])   # 84-178
        v2 = np.linalg.norm(mouth_landmarks[3] - mouth_landmarks[7])   # 314-95

        # Horizontal distance
        h = np.linalg.norm(mouth_landmarks[0] - mouth_landmarks[6])    # 61-78

        if h == 0:
            return 0.0

        mar = (v1 + v2) / (2.0 * h)
        return float(mar)

    def _estimate_head_pose(self, landmarks: np.ndarray, image_shape: Tuple[int, int]) -> HeadPose:
        """
        Estimate head pose using PnP algorithm.

        Args:
            landmarks: 468 facial landmarks [468, 3] normalized
            image_shape: (height, width) of image
        Returns:
            HeadPose object with pitch, yaw, roll
        """
        h, w = image_shape

        # Get 2D image points from landmarks
        image_points = np.array([
            landmarks[1][:2] * [w, h],      # Nose tip
            landmarks[199][:2] * [w, h],    # Chin
            landmarks[33][:2] * [w, h],     # Left eye left corner
            landmarks[263][:2] * [w, h],    # Right eye right corner
            landmarks[61][:2] * [w, h],     # Left mouth corner
            landmarks[291][:2] * [w, h],    # Right mouth corner
        ], dtype=np.float32)

        # Camera matrix (approximate)
        if self.camera_matrix is None:
            focal_length = w
            center = (w / 2, h / 2)
            self.camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype=np.float32)

        # Solve PnP
        success, rotation_vec, translation_vec = cv2.solvePnP(
            self.model_points,
            image_points,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return HeadPose(0.0, 0.0, 0.0, np.array([0.0, 0.0, 1.0]))

        # Convert rotation vector to Euler angles
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        pose_mat = cv2.hconcat((rotation_mat, translation_vec))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)

        pitch, yaw, roll = euler_angles.flatten()[:3]

        # Forward vector (where nose is pointing)
        forward = rotation_mat @ np.array([0, 0, 1], dtype=np.float32)

        return HeadPose(
            pitch=float(pitch),
            yaw=float(yaw),
            roll=float(roll),
            forward_vector=forward,
        )

    def _calculate_perclos(self) -> float:
        """Calculate PERCLOS from EAR history."""
        if len(self.ear_history) == 0:
            return 0.0

        closed_count = sum(1 for ear in self.ear_history if ear < self.ear_threshold)
        return closed_count / len(self.ear_history)

    def _determine_fatigue_state(
        self,
        perclos: float,
        blink_rate: float,
        yawn_detected: bool,
        head_pose: HeadPose,
    ) -> Tuple[FatigueState, float]:
        """
        Determine fatigue state based on multiple cues.

        Returns:
            (state, confidence)
        """
        # Thresholds (can be tuned)
        perclos_drowsy = 0.15
        perclos_fatigued = 0.30
        blink_rate_high = 30  # blinks per minute
        pitch_threshold = 20  # degrees

        confidence = 0.5
        state = FatigueState.ALERT

        # Head pose check (looking away/down)
        head_down = abs(head_pose.pitch) > pitch_threshold

        if perclos >= perclos_fatigued or (perclos >= perclos_drowsy and head_down):
            state = FatigueState.FATIGUED
            confidence = min(0.9, 0.5 + perclos)
        elif perclos >= perclos_drowsy or blink_rate > blink_rate_high or yawn_detected:
            state = FatigueState.DROWSY
            confidence = min(0.8, 0.4 + perclos * 2)
        else:
            state = FatigueState.ALERT
            confidence = max(0.5, 1.0 - perclos * 2)

        return state, confidence

    def process_frame(self, frame: np.ndarray) -> VisionFrameResult:
        """
        Process a single frame for fatigue indicators.

        Args:
            frame: BGR image [H, W, 3]
        Returns:
            VisionFrameResult with all metrics
        """
        start_time = time.perf_counter()
        self.frame_id += 1

        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]

        # Process with MediaPipe
        results = self.face_mesh.process(rgb_frame)

        result = VisionFrameResult(
            timestamp=time.time(),
            frame_id=self.frame_id,
            face_detected=False,
        )

        if results.multi_face_landmarks:
            result.face_detected = True
            face_landmarks = results.multi_face_landmarks[0]

            # Extract all 468 landmarks
            landmarks = np.array([
                [lm.x, lm.y, lm.z] for lm in face_landmarks.landmark
            ])
            result.landmarks = landmarks

            # Eye metrics
            left_eye_lm = landmarks[self.LEFT_EYE_INDICES]
            right_eye_lm = landmarks[self.RIGHT_EYE_INDICES]

            left_ear = self._calculate_ear(left_eye_lm)
            right_ear = self._calculate_ear(right_eye_lm)
            avg_ear = (left_ear + right_ear) / 2.0

            # Update EAR history
            self.ear_history.append(avg_ear)

            # Blink detection
            left_open = left_ear > self.ear_threshold
            right_open = right_ear > self.ear_threshold
            both_open = left_open and right_open

            blink_detected = False
            if not both_open:
                self.eye_closed_frames += 1
            else:
                if self.eye_closed_frames >= self.consecutive_frames:
                    blink_detected = True
                    self.blink_count += 1
                    self.blink_history.append(1)
                else:
                    self.blink_history.append(0)
                self.eye_closed_frames = 0

            # PERCLOS
            perclos = self._calculate_perclos()

            # Blink rate (blinks per minute)
            blink_rate = (sum(self.blink_history) / len(self.blink_history)) * 60 * 30 if self.blink_history else 0

            result.eye_metrics = EyeMetrics(
                left_ear=left_ear,
                right_ear=right_ear,
                avg_ear=avg_ear,
                left_eye_open=left_open,
                right_eye_open=right_open,
                blink_detected=blink_detected,
                perclos=perclos,
            )

            # Mouth metrics
            mouth_lm = landmarks[self.MOUTH_INDICES]
            mar = self._calculate_mar(mouth_lm)
            yawn_detected = mar > self.mar_threshold

            if yawn_detected:
                self.yawn_frames += 1
            else:
                self.yawn_frames = max(0, self.yawn_frames - 1)

            result.mouth_metrics = MouthMetrics(
                mar=mar,
                yawn_detected=yawn_detected,
            )

            # Head pose
            head_pose = self._estimate_head_pose(landmarks, (h, w))
            result.head_pose = head_pose

            # Fatigue state
            state, confidence = self._determine_fatigue_state(
                perclos, blink_rate, yawn_detected, head_pose
            )
            result.fatigue_state = state
            result.confidence = confidence

        # Preprocessing time
        result.preprocessing_time_ms = (time.perf_counter() - start_time) * 1000

        return result

    def preprocess_for_model(self, frame: np.ndarray) -> torch.Tensor:
        """
        Preprocess frame for model inference.

        Args:
            frame: BGR image [H, W, 3]
        Returns:
            Tensor [1, 3, H, W] normalized
        """
        # Resize
        resized = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_LINEAR)

        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1]
        normalized = rgb.astype(np.float32) / 255.0

        # Normalize with ImageNet stats
        normalized = (normalized - self.norm_mean) / self.norm_std

        # Convert to tensor [C, H, W]
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)

        return tensor

    def draw_annotations(
        self,
        frame: np.ndarray,
        result: VisionFrameResult,
        draw_landmarks: bool = True,
        draw_metrics: bool = True,
    ) -> np.ndarray:
        """
        Draw visualization annotations on frame.

        Args:
            frame: BGR image
            result: VisionFrameResult
            draw_landmarks: Whether to draw facial landmarks
            draw_metrics: Whether to draw metric overlays
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        h, w = frame.shape[:2]

        if result.face_detected and result.landmarks is not None:
            # Draw face mesh
            if draw_landmarks:
                # Convert landmarks to MediaPipe format for drawing
                landmarks_proto = mp.framework.formats.landmark_pb2.NormalizedLandmarkList()
                for lm in result.landmarks:
                    landmarks_proto.landmark.add(x=lm[0], y=lm[1], z=lm[2])

                self.mp_drawing.draw_landmarks(
                    image=annotated,
                    landmark_list=landmarks_proto,
                    connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style(),
                )

                # Draw eye contours
                self.mp_drawing.draw_landmarks(
                    image=annotated,
                    landmark_list=landmarks_proto,
                    connections=mp.solutions.face_mesh.FACEMESH_LEFT_EYE,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style(),
                )
                self.mp_drawing.draw_landmarks(
                    image=annotated,
                    landmark_list=landmarks_proto,
                    connections=mp.solutions.face_mesh.FACEMESH_RIGHT_EYE,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style(),
                )

            # Draw metrics overlay
            if draw_metrics and result.eye_metrics:
                metrics = result.eye_metrics
                color = (0, 255, 0) if result.fatigue_state == FatigueState.ALERT else \
                        (0, 255, 255) if result.fatigue_state == FatigueState.DROWSY else \
                        (0, 0, 255)

                # PERCLOS bar
                bar_width = int(200 * metrics.perclos)
                cv2.rectangle(annotated, (10, 30), (10 + bar_width, 50), color, -1)
                cv2.rectangle(annotated, (10, 30), (210, 50), (255, 255, 255), 2)
                cv2.putText(annotated, f"PERCLOS: {metrics.perclos:.2%}", (10, 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # EAR
                cv2.putText(annotated, f"EAR: {metrics.avg_ear:.3f}", (10, 65),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # Blink indicator
                if metrics.blink_detected:
                    cv2.putText(annotated, "BLINK!", (w - 100, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                # Fatigue state
                cv2.putText(annotated, f"State: {result.fatigue_state.value.upper()} ({result.confidence:.2f})",
                           (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Head pose
                if result.head_pose:
                    hp = result.head_pose
                    cv2.putText(annotated, f"Pitch: {hp.pitch:.1f} Yaw: {hp.yaw:.1f} Roll: {hp.roll:.1f}",
                               (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # FPS
        if result.preprocessing_time_ms > 0:
            fps = 1000 / result.preprocessing_time_ms
            cv2.putText(annotated, f"FPS: {fps:.1f}", (w - 100, h - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return annotated

    def reset_state(self):
        """Reset internal state for new session."""
        self.eye_closed_frames = 0
        self.blink_count = 0
        self.ear_history.clear()
        self.blink_history.clear()
        self.yawn_frames = 0
        self.frame_id = 0
        logger.info("VisionProcessor state reset")

    def close(self):
        """Release resources."""
        self.face_mesh.close()
        logger.info("VisionProcessor closed")


class VideoCaptureManager:
    """
    Manages video capture from webcam or video file with buffering.
    """

    def __init__(
        self,
        source: Union[int, str] = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        buffer_size: int = 1,
    ):
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.buffer_size = buffer_size

        self.cap = None
        self.running = False
        self.frame_queue = deque(maxlen=buffer_size)

    def start(self) -> bool:
        """Start video capture."""
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            logger.error(f"Failed to open video source: {self.source}")
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)

        self.running = True
        logger.info(f"Video capture started: {self.source} ({self.width}x{self.height}@{self.fps}fps)")
        return True

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a frame from capture."""
        if not self.running or self.cap is None:
            return False, None

        ret, frame = self.cap.read()
        if ret:
            self.frame_queue.append(frame)
        return ret, frame

    def get_latest(self) -> Optional[np.ndarray]:
        """Get latest frame from buffer."""
        if self.frame_queue:
            return self.frame_queue[-1]
        return None

    def stop(self):
        """Stop video capture."""
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        logger.info("Video capture stopped")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


if __name__ == "__main__":
    # Demo usage
    import yaml

    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    processor = VisionProcessor(config)

    # Test with webcam
    with VideoCaptureManager(0, 640, 480, 30) as cap:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = processor.process_frame(frame)
            annotated = processor.draw_annotations(frame, result)

            cv2.imshow("Fatigue Monitor", annotated)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    processor.close()
    cv2.destroyAllWindows()