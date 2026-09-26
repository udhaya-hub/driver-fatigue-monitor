"""
Inference module for fatigue monitoring.
"""

from inference.vision_processor import (
    VisionProcessor,
    VisionFrameResult,
    EyeMetrics,
    MouthMetrics,
    HeadPose,
    FatigueState,
    VideoCaptureManager,
)

__all__ = [
    "VisionProcessor",
    "VisionFrameResult",
    "EyeMetrics",
    "MouthMetrics",
    "HeadPose",
    "FatigueState",
    "VideoCaptureManager",
]