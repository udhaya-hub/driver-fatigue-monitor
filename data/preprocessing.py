"""
Data preprocessing and augmentation pipelines for fatigue monitoring.
"""

import torch
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing pipelines."""
    image_size: Tuple[int, int] = (224, 224)
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    augment: bool = True
    augment_prob: float = 0.5


def get_train_transforms(config: PreprocessingConfig) -> A.Compose:
    """Get training augmentation pipeline."""
    transforms = [
        A.Resize(config.image_size[0], config.image_size[1]),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.GaussNoise(var_limit=(10, 50), p=0.3),
        A.MotionBlur(blur_limit=3, p=0.2),
        A.RandomRotate90(p=0.2),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.1,
            rotate_limit=10,
            p=0.3,
            border_mode=cv2.BORDER_CONSTANT,
        ),
        A.CoarseDropout(
            max_holes=8,
            max_height=16,
            max_width=16,
            min_holes=1,
            fill_value=0,
            p=0.3,
        ),
        A.Normalize(mean=config.normalize_mean, std=config.normalize_std),
        ToTensorV2(),
    ]
    return A.Compose(transforms)


def get_val_transforms(config: PreprocessingConfig) -> A.Compose:
    """Get validation/test transforms (no augmentation)."""
    transforms = [
        A.Resize(config.image_size[0], config.image_size[1]),
        A.Normalize(mean=config.normalize_mean, std=config.normalize_std),
        ToTensorV2(),
    ]
    return A.Compose(transforms)


def get_inference_transforms(config: PreprocessingConfig) -> A.Compose:
    """Get inference transforms (same as validation)."""
    return get_val_transforms(config)


class PhysiologicalPreprocessor:
    """
    Preprocessor for physiological signals (ECG, EDA, RESP, TEMP).
    """

    def __init__(
        self,
        sampling_rate: int = 64,
        window_size: int = 30,  # seconds
        overlap: float = 0.5,
    ):
        self.sampling_rate = sampling_rate
        self.window_size = window_size
        self.overlap = overlap
        self.window_samples = window_size * sampling_rate
        self.step_samples = int(self.window_samples * (1 - overlap))

    def segment_signal(self, signal: np.ndarray) -> List[np.ndarray]:
        """
        Segment continuous signal into overlapping windows.

        Args:
            signal: 1D array of signal values
        Returns:
            List of windowed signals
        """
        windows = []
        for i in range(0, len(signal) - self.window_samples + 1, self.step_samples):
            window = signal[i:i + self.window_samples]
            if len(window) == self.window_samples:
                windows.append(window)
        return windows

    def extract_hrv_features(self, rri: np.ndarray) -> Dict[str, float]:
        """
        Extract HRV features from R-R intervals.

        Args:
            rri: R-R intervals in milliseconds
        Returns:
            Dictionary of HRV features
        """
        if len(rri) < 2:
            return {k: 0.0 for k in [
                "mean_rr", "sdnn", "rmssd", "pnn50", "lf_power", "hf_power", "lf_hf_ratio"
            ]}

        rri = np.array(rri, dtype=np.float32)

        # Time domain
        mean_rr = np.mean(rri)
        sdnn = np.std(rri, ddof=1)
        diff_rr = np.diff(rri)
        rmssd = np.sqrt(np.mean(diff_rr ** 2))
        pnn50 = np.sum(np.abs(diff_rr) > 50) / len(diff_rr) * 100

        # Frequency domain (simplified - would use proper spectral analysis)
        # Using Lomb-Scargle or FFT in practice
        from scipy.signal import welch
        try:
            freqs, psd = welch(rri, fs=4.0, nperseg=min(256, len(rri)))
            lf_band = (freqs >= 0.04) & (freqs <= 0.15)
            hf_band = (freqs >= 0.15) & (freqs <= 0.40)
            lf_power = np.trapz(psd[lf_band], freqs[lf_band])
            hf_power = np.trapz(psd[hf_band], freqs[hf_band])
            lf_hf_ratio = lf_power / (hf_power + 1e-10)
        except:
            lf_power = hf_power = lf_hf_ratio = 0.0

        return {
            "mean_rr": float(mean_rr),
            "sdnn": float(sdnn),
            "rmssd": float(rmssd),
            "pnn50": float(pnn50),
            "lf_power": float(lf_power),
            "hf_power": float(hf_power),
            "lf_hf_ratio": float(lf_hf_ratio),
        }

    def extract_eda_features(self, eda: np.ndarray) -> Dict[str, float]:
        """
        Extract EDA (skin conductance) features.

        Args:
            eda: EDA signal
        Returns:
            Dictionary of EDA features
        """
        if len(eda) == 0:
            return {"eda_tonic": 0.0, "eda_phasic": 0.0, "scr_count": 0, "scr_amplitude": 0.0}

        # Simple tonic/phasic decomposition using low-pass filter
        from scipy.signal import butter, filtfilt

        # Low-pass for tonic component
        b, a = butter(2, 0.05 / (self.sampling_rate / 2), btype='low')
        tonic = filtfilt(b, a, eda)
        phasic = eda - tonic

        # SCR detection (simple peak detection)
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(phasic, height=np.std(phasic), distance=self.sampling_rate)

        return {
            "eda_tonic": float(np.mean(tonic)),
            "eda_phasic": float(np.mean(np.abs(phasic))),
            "scr_count": float(len(peaks)),
            "scr_amplitude": float(np.mean(phasic[peaks]) if len(peaks) > 0 else 0.0),
        }

    def extract_resp_features(self, resp: np.ndarray) -> Dict[str, float]:
        """
        Extract respiration features.

        Args:
            resp: Respiration signal
        Returns:
            Dictionary of respiration features
        """
        if len(resp) == 0:
            return {"resp_rate": 0.0, "resp_amplitude": 0.0, "resp_regularity": 0.0}

        from scipy.signal import find_peaks

        # Find peaks (inhalation)
        peaks, _ = find_peaks(resp, distance=self.sampling_rate * 2)  # Min 2s between breaths

        if len(peaks) < 2:
            return {"resp_rate": 0.0, "resp_amplitude": 0.0, "resp_regularity": 0.0}

        # Respiratory rate (breaths per minute)
        intervals = np.diff(peaks) / self.sampling_rate
        resp_rate = 60.0 / np.mean(intervals)

        # Amplitude
        resp_amplitude = np.mean(resp[peaks]) - np.mean(resp)

        # Regularity (CV of intervals)
        resp_regularity = 1.0 - (np.std(intervals) / (np.mean(intervals) + 1e-10))

        return {
            "resp_rate": float(np.clip(resp_rate, 4, 40)),
            "resp_amplitude": float(resp_amplitude),
            "resp_regularity": float(np.clip(resp_regularity, 0, 1)),
        }

    def process_all(
        self,
        ecg: Optional[np.ndarray] = None,
        eda: Optional[np.ndarray] = None,
        resp: Optional[np.ndarray] = None,
        temp: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Process all physiological signals and return feature vector.

        Args:
            ecg: ECG signal (or RRI)
            eda: EDA signal
            resp: Respiration signal
            temp: Temperature signal
        Returns:
            Feature vector [n_features]
        """
        features = []

        if ecg is not None:
            hrv = self.extract_hrv_features(ecg)
            features.extend([
                hrv["mean_rr"], hrv["sdnn"], hrv["rmssd"], hrv["pnn50"],
                hrv["lf_power"], hrv["hf_power"], hrv["lf_hf_ratio"],
            ])

        if eda is not None:
            eda_feat = self.extract_eda_features(eda)
            features.extend([
                eda_feat["eda_tonic"], eda_feat["eda_phasic"],
                eda_feat["scr_count"], eda_feat["scr_amplitude"],
            ])

        if resp is not None:
            resp_feat = self.extract_resp_features(resp)
            features.extend([
                resp_feat["resp_rate"], resp_feat["resp_amplitude"], resp_feat["resp_regularity"],
            ])

        if temp is not None:
            features.extend([
                float(np.mean(temp)), float(np.std(temp)),
                float(np.min(temp)), float(np.max(temp)),
            ])

        return np.array(features, dtype=np.float32)


class VideoPreprocessor:
    """
    Preprocessor for video frames.
    """

    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.transform = get_inference_transforms(config)

    def preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """
        Preprocess single frame for model input.

        Args:
            frame: BGR image [H, W, 3]
        Returns:
            Tensor [3, H, W]
        """
        # Convert BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Apply transforms
        transformed = self.transform(image=rgb)
        return transformed["image"]

    def preprocess_sequence(self, frames: List[np.ndarray]) -> torch.Tensor:
        """
        Preprocess sequence of frames.

        Args:
            frames: List of BGR images
        Returns:
            Tensor [T, 3, H, W]
        """
        tensors = [self.preprocess_frame(f) for f in frames]
        return torch.stack(tensors)


def create_data_loaders(
    train_dataset,
    val_dataset,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
):
    """
    Create PyTorch DataLoaders.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        batch_size: Batch size
        num_workers: Number of workers
        pin_memory: Pin memory for GPU
    Returns:
        train_loader, val_loader
    """
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader