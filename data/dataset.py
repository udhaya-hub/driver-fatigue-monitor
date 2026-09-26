"""
Dataset classes for fatigue monitoring training data.
"""

import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class FatigueVideoDataset(Dataset):
    """
    Dataset for video-based fatigue monitoring.

    Expected directory structure:
    data_root/
        subject_001/
            session_001/
                frames/
                    frame_000001.jpg
                    frame_000002.jpg
                    ...
                annotations.json
                physio/
                    ecg.csv
                    eda.csv
                    resp.csv
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        sequence_length: int = 30,
        stride: int = 1,
        transform=None,
        cache_annotations: bool = True,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.sequence_length = sequence_length
        self.stride = stride
        self.transform = transform
        self.cache_annotations = cache_annotations

        self.samples = self._build_sample_index()

        logger.info(f"Loaded {len(self.samples)} samples from {split} split")

    def _build_sample_index(self) -> List[Dict]:
        """Build index of all valid sequences."""
        samples = []

        # Look for split file or use directory structure
        split_file = self.data_root / f"{self.split}.txt"
        if split_file.exists():
            with open(split_file, "r") as f:
                session_dirs = [line.strip() for line in f if line.strip()]
        else:
            # Auto-discover sessions
            session_dirs = [d.name for d in self.data_root.iterdir() if d.is_dir()]

        for session_dir in session_dirs:
            session_path = self.data_root / session_dir
            frames_dir = session_path / "frames"
            annotations_path = session_path / "annotations.json"

            if not frames_dir.exists() or not annotations_path.exists():
                continue

            # Load annotations
            with open(annotations_path, "r") as f:
                annotations = json.load(f)

            # Get frame files
            frame_files = sorted(frames_dir.glob("*.jpg")) + sorted(frames_dir.glob("*.png"))
            if len(frame_files) < self.sequence_length:
                continue

            # Create sequences with stride
            for i in range(0, len(frame_files) - self.sequence_length + 1, self.stride):
                frame_seq = frame_files[i:i + self.sequence_length]
                sample = {
                    "session_id": session_dir,
                    "frame_paths": [str(p) for p in frame_seq],
                    "start_idx": i,
                    "annotations": annotations.get(str(i), {}),
                }
                samples.append(sample)

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Load frames
        frames = []
        for frame_path in sample["frame_paths"]:
            frame = cv2.imread(frame_path)
            if frame is None:
                # Create black frame as fallback
                frame = np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if self.transform:
                transformed = self.transform(image=frame)
                frame = transformed["image"]
            else:
                # Default preprocessing
                frame = cv2.resize(frame, (224, 224))
                frame = frame.astype(np.float32) / 255.0
                frame = torch.from_numpy(frame).permute(2, 0, 1)

            frames.append(frame)

        frames_tensor = torch.stack(frames)  # [T, C, H, W]

        # Load physiological features if available
        physio_features = self._load_physio(sample["session_id"], sample["start_idx"])

        # Get labels
        labels = self._get_labels(sample["annotations"])

        return {
            "frames": frames_tensor,
            "physio": physio_features,
            "labels": labels,
            "session_id": sample["session_id"],
            "start_idx": sample["start_idx"],
        }

    def _load_physio(self, session_id: str, start_idx: int) -> torch.Tensor:
        """Load physiological features for sequence."""
        physio_dir = self.data_root / session_id / "physio"
        if not physio_dir.exists():
            return torch.zeros(self.sequence_length, 32)  # Default feature dim

        # In practice, load and align physio data with video frames
        # This is a placeholder
        return torch.zeros(self.sequence_length, 32)

    def _get_labels(self, annotations: Dict) -> Dict[str, torch.Tensor]:
        """Extract labels from annotations."""
        # Fatigue state: 0=alert, 1=drowsy, 2=fatigued
        fatigue_map = {"alert": 0, "drowsy": 1, "fatigued": 2}
        fatigue_state = annotations.get("fatigue_state", "alert")
        fatigue_label = fatigue_map.get(fatigue_state, 0)

        # Workload score [0, 1]
        workload = annotations.get("workload", 0.5)

        # Blink label
        blink = annotations.get("blink", 0)

        return {
            "fatigue": torch.tensor(fatigue_label, dtype=torch.long),
            "workload": torch.tensor(workload, dtype=torch.float32),
            "blink": torch.tensor(blink, dtype=torch.long),
        }


class FatiguePhysioDataset(Dataset):
    """
    Dataset for physiological signal-based fatigue monitoring.
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        window_size: int = 30,  # seconds
        sampling_rate: int = 64,
        overlap: float = 0.5,
        transform=None,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.window_size = window_size
        self.sampling_rate = sampling_rate
        self.overlap = overlap
        self.transform = transform

        self.samples = self._build_index()

    def _build_index(self) -> List[Dict]:
        """Build index of physiological recording segments."""
        samples = []
        split_dir = self.data_root / self.split

        if not split_dir.exists():
            logger.warning(f"Split directory not found: {split_dir}")
            return samples

        for subject_dir in split_dir.iterdir():
            if not subject_dir.is_dir():
                continue

            for session_file in subject_dir.glob("*.npz"):
                # Load metadata
                data = np.load(session_file, allow_pickle=True)
                duration = data.get("duration", 0)

                if duration < self.window_size:
                    continue

                samples.append({
                    "path": str(session_file),
                    "subject": subject_dir.name,
                    "duration": duration,
                })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        data = np.load(sample["path"], allow_pickle=True)

        # Extract signals
        ecg = data.get("ecg", np.array([]))
        eda = data.get("eda", np.array([]))
        resp = data.get("resp", np.array([]))
        temp = data.get("temp", np.array([]))
        labels = data.get("labels", {})

        # Segment into windows
        window_samples = self.window_size * self.sampling_rate
        step_samples = int(window_samples * (1 - self.overlap))

        # For simplicity, take first window
        ecg_win = ecg[:window_samples] if len(ecg) >= window_samples else np.pad(ecg, (0, window_samples - len(ecg)))
        eda_win = eda[:window_samples] if len(eda) >= window_samples else np.pad(eda, (0, window_samples - len(eda)))
        resp_win = resp[:window_samples] if len(resp) >= window_samples else np.pad(resp, (0, window_samples - len(resp)))
        temp_win = temp[:window_samples] if len(temp) >= window_samples else np.pad(temp, (0, window_samples - len(temp)))

        return {
            "ecg": torch.from_numpy(ecg_win).float(),
            "eda": torch.from_numpy(eda_win).float(),
            "resp": torch.from_numpy(resp_win).float(),
            "temp": torch.from_numpy(temp_win).float(),
            "fatigue": torch.tensor(labels.get("fatigue", 0), dtype=torch.long),
            "workload": torch.tensor(labels.get("workload", 0.5), dtype=torch.float32),
        }


class MultiModalFatigueDataset(Dataset):
    """
    Combined video + physiological dataset for multi-modal training.
    """

    def __init__(
        self,
        video_data_root: str,
        physio_data_root: str,
        split: str = "train",
        sequence_length: int = 30,
        sync_tolerance: float = 0.1,  # seconds
        video_transform=None,
    ):
        self.video_dataset = FatigueVideoDataset(
            video_data_root, split, sequence_length, transform=video_transform
        )
        self.physio_dataset = FatiguePhysioDataset(
            physio_data_root, split
        )
        self.sync_tolerance = sync_tolerance

        # Build mapping between video and physio sessions
        self._build_sync_mapping()

    def _build_sync_mapping(self):
        """Map video sessions to physiological sessions."""
        self.sync_map = {}
        # In practice, match by subject ID, timestamp, etc.
        # Placeholder implementation
        for i, video_sample in enumerate(self.video_dataset.samples):
            self.sync_map[i] = i % len(self.physio_dataset)

    def __len__(self) -> int:
        return len(self.video_dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        video_data = self.video_dataset[idx]
        physio_idx = self.sync_map.get(idx, 0)
        physio_data = self.physio_dataset[physio_idx]

        return {
            "frames": video_data["frames"],
            "physio": physio_data,
            "labels": video_data["labels"],
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for variable-length sequences.

    Args:
        batch: List of samples
    Returns:
        Batched tensors
    """
    # Find max sequence length in batch
    max_len = max(item["frames"].shape[0] for item in batch)

    # Pad sequences
    frames_batch = []
    physio_batch = []
    labels_batch = {"fatigue": [], "workload": [], "blink": []}

    for item in batch:
        frames = item["frames"]
        T, C, H, W = frames.shape

        if T < max_len:
            # Pad with zeros
            padding = torch.zeros(max_len - T, C, H, W)
            frames = torch.cat([frames, padding], dim=0)

        frames_batch.append(frames)

        # Physio features
        physio = item.get("physio", torch.zeros(max_len, 32))
        if physio.shape[0] < max_len:
            physio = torch.cat([physio, torch.zeros(max_len - physio.shape[0], physio.shape[1])], dim=0)
        physio_batch.append(physio[:max_len])

        # Labels
        labels_batch["fatigue"].append(item["labels"]["fatigue"])
        labels_batch["workload"].append(item["labels"]["workload"])
        labels_batch["blink"].append(item["labels"]["blink"])

    return {
        "frames": torch.stack(frames_batch),
        "physio": torch.stack(physio_batch),
        "labels": {
            "fatigue": torch.stack(labels_batch["fatigue"]),
            "workload": torch.stack(labels_batch["workload"]),
            "blink": torch.stack(labels_batch["blink"]),
        },
    }