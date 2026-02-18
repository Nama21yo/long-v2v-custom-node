"""
LPIPS Video Quality Analyzer
============================
Measures perceptual similarity between video frames using deep learning.
Used for detecting visual discontinuities, drift, and transition quality.

"""

import io
import subprocess
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
from pathlib import Path

import torch
import numpy as np
from PIL import Image
import lpips
from .logger import logger


@dataclass
class SeamAnalysis:
    """Results from analyzing a single video seam."""
    seam_index: int
    lpips_score: Optional[float]
    file_a: str
    file_b: str
    frame_a_timestamp: float
    frame_b_timestamp: float
    quality_rating: str  # 'excellent', 'good', 'noticeable', 'poor'
    
    def __post_init__(self):
        """Auto-classify quality based on LPIPS score."""
        if self.quality_rating == 'unknown' or self.lpips_score is None or self.lpips_score < 0:
            return
        if self.lpips_score < 0.05:
            self.quality_rating = 'excellent'
        elif self.lpips_score < 0.1:
            self.quality_rating = 'good'
        elif self.lpips_score < 0.3:
            self.quality_rating = 'noticeable'
        else:
            self.quality_rating = 'poor'


@dataclass
class DriftAnalysis:
    """Results from temporal drift analysis."""
    frame_numbers: List[int]
    lpips_scores: List[float]
    reference_frame_index: int
    max_drift: float
    mean_drift: float
    drift_trend: str  # 'stable', 'gradual', 'sudden'

    def __post_init__(self):
        """Auto-classify drift trend from LPIPS scores when not provided."""
        if self.drift_trend:
            return
        if len(self.lpips_scores) < 3:
            self.drift_trend = 'stable'
            return
        differences = [self.lpips_scores[i+1] - self.lpips_scores[i] for i in range(len(self.lpips_scores)-1)]
        max_jump = max(abs(d) for d in differences)
        avg_change = np.mean(np.abs(differences))
        if max_jump > 0.2:
            self.drift_trend = 'sudden'
        elif avg_change > 0.05:
            self.drift_trend = 'gradual'
        else:
            self.drift_trend = 'stable'


class LPIPSAnalyzer:
    """
    Perceptual video quality analyzer using LPIPS metric.
    
    This class provides tools for:
    - Seam discontinuity detection (batch transitions)
    - Temporal drift analysis (character consistency)
    - Frame-to-frame stability measurement
    """
    
    def __init__(self, network: str = 'alex', device: str = 'cpu'):
        """
        Initialize LPIPS analyzer.
        
        Args:
            network: Neural network backbone ('alex', 'vgg', 'squeeze')
                    'alex' is fastest, 'vgg' is most accurate
            device: Computation device ('cpu' or 'cuda')
        """
        self.device = device
        self.loss_fn = lpips.LPIPS(net=network).to(device)
        self.loss_fn.eval()  # Set to evaluation mode
        
    def analyze_seams(
        self, 
        video_files: List[str],
        overlap_frames: int = 30,
        ffmpeg_path: str = 'ffmpeg'
    ) -> List[SeamAnalysis]:
        """
        Analyze transition quality between consecutive video chunks.
        
        Args:
            video_files: Ordered list of video file paths
            overlap_frames: Number of overlapping frames used to estimate transition frame positions
            ffmpeg_path: Path to FFmpeg executable
            
        Returns:
            List of SeamAnalysis objects, one per transition
        """
        if len(video_files) < 2:
            return []
            
        seam_results = []
        
        for i in range(len(video_files) - 1):
            file_a = video_files[i]
            file_b = video_files[i + 1]
            
            # Get video durations
            duration_a = self._get_video_duration(file_a, ffmpeg_path)
            fps_a = self._get_video_framerate(file_a, ffmpeg_path)
            fps_b = self._get_video_framerate(file_b, ffmpeg_path)
            
            # Extract transition frames
            # Last overlap frame of video A based on framerate
            if duration_a <= 0.05:
                logger.warning(f"[LPIPSAnalyzer] Very short video for seam analysis ({duration_a:.4f}s): {file_a}; using earliest frame")
                timestamp_a = 0.0
            else:
                overlap_seconds_a = overlap_frames / fps_a if fps_a > 0 else 0.0
                timestamp_a = max(0.0, duration_a - max(0.05, overlap_seconds_a))
            frame_a = self._extract_frame(file_a, timestamp_a, ffmpeg_path)
            
            # First frame of video B
            overlap_seconds_b = overlap_frames / fps_b if fps_b > 0 else 0.0
            timestamp_b = min(overlap_seconds_b, 0.05) if overlap_seconds_b > 0 else 0.0
            frame_b = self._extract_frame(file_b, timestamp_b, ffmpeg_path)
            
            if frame_a is None or frame_b is None:
                # Fallback score if extraction fails
                seam_results.append(SeamAnalysis(
                    seam_index=i,
                    lpips_score=None,
                    file_a=file_a,
                    file_b=file_b,
                    frame_a_timestamp=timestamp_a,
                    frame_b_timestamp=timestamp_b,
                    quality_rating='unknown'
                ))
                continue
            
            # Calculate LPIPS distance
            lpips_score = self._calculate_lpips(frame_a, frame_b)
            
            seam_results.append(SeamAnalysis(
                seam_index=i,
                lpips_score=lpips_score,
                file_a=file_a,
                file_b=file_b,
                frame_a_timestamp=timestamp_a,
                frame_b_timestamp=timestamp_b,
                quality_rating=''  # Will be auto-set by __post_init__
            ))
            
        return seam_results
    
    def analyze_temporal_drift(
        self,
        video_file: str,
        reference_timestamp: float,
        sample_timestamps: List[float],
        ffmpeg_path: str = 'ffmpeg'
    ) -> DriftAnalysis:
        """
        Measure how much video content drifts from a reference frame over time.
        
        Args:
            video_file: Path to video file
            reference_timestamp: Timestamp of reference frame (seconds)
            sample_timestamps: List of timestamps to compare against reference
            ffmpeg_path: Path to FFmpeg executable
            
        Returns:
            DriftAnalysis object with drift metrics
        """
        # Extract reference frame
        ref_frame = self._extract_frame(video_file, reference_timestamp, ffmpeg_path)
        if ref_frame is None:
            raise ValueError(f"Could not extract reference frame at {reference_timestamp}s")
        
        scores = []
        valid_timestamps = []
        
        for ts in sample_timestamps:
            sample_frame = self._extract_frame(video_file, ts, ffmpeg_path)
            if sample_frame is not None:
                score = self._calculate_lpips(ref_frame, sample_frame)
                scores.append(score)
                valid_timestamps.append(ts)
        
        if not scores:
            raise ValueError("No valid frames extracted for drift analysis")

        fps = self._get_video_framerate(video_file, ffmpeg_path)
        
        return DriftAnalysis(
            frame_numbers=[int(ts * fps) for ts in valid_timestamps],
            lpips_scores=scores,
            reference_frame_index=int(reference_timestamp * fps),
            max_drift=max(scores),
            mean_drift=float(np.mean(scores)),
            drift_trend=''
        )
    
    def _calculate_lpips(self, image_a: Image.Image, image_b: Image.Image) -> float:
        """
        Calculate LPIPS distance between two PIL images.
        
        Args:
            image_a: First image
            image_b: Second image
            
        Returns:
            LPIPS distance (0 = identical, higher = more different)
        """
        # Convert to tensors
        tensor_a = self._image_to_tensor(image_a)
        tensor_b = self._image_to_tensor(image_b)
        
        # Calculate LPIPS
        with torch.no_grad():
            distance = self.loss_fn(tensor_a, tensor_b).item()
        
        return distance
    
    def _image_to_tensor(self, pil_image: Image.Image) -> torch.Tensor:
        """
        Convert PIL Image to normalized tensor for LPIPS.
        
        Args:
            pil_image: Input PIL Image
            
        Returns:
            Tensor with shape [1, 3, H, W], normalized to [-1, 1]
        """
        # Ensure RGB
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Convert to numpy array and normalize to [0, 1]
        np_img = np.array(pil_image).astype(np.float32) / 255.0
        
        # Convert to tensor: HWC -> CHW
        tensor = torch.from_numpy(np_img).permute(2, 0, 1).unsqueeze(0)
        
        # Normalize to [-1, 1] as expected by LPIPS
        tensor = (tensor * 2.0) - 1.0
        
        return tensor.to(self.device)
    
    def _extract_frame(
        self, 
        video_path: str, 
        timestamp: float,
        ffmpeg_path: str
    ) -> Optional[Image.Image]:
        """
        Extract a single frame from video at specified timestamp.
        
        Args:
            video_path: Path to video file
            timestamp: Time in seconds
            ffmpeg_path: Path to FFmpeg executable
            
        Returns:
            PIL Image or None if extraction fails
        """
        cmd = [
            ffmpeg_path,
            '-ss', str(timestamp),
            '-i', video_path,
            '-frames:v', '1',
            '-f', 'image2',
            '-c:v', 'png',
            'pipe:1'
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                check=False,
                timeout=10
            )
            
            if result.returncode != 0:
                stderr = (result.stderr.decode('utf-8', errors='ignore') if isinstance(result.stderr, bytes) else str(result.stderr))
                logger.error(f"[LPIPSAnalyzer] FFmpeg frame extraction failed (code {result.returncode}) at {timestamp}s for {video_path}: {stderr.strip()}")
                return None

            if result.stdout:
                return Image.open(io.BytesIO(result.stdout))
            logger.error(f"[LPIPSAnalyzer] FFmpeg produced no frame output at {timestamp}s for {video_path}")
        except Exception as e:
            logger.error(f"[LPIPSAnalyzer] Frame extraction failed: {e}")
        
        return None
    
    def _get_video_duration(self, video_path: str, ffmpeg_path: str) -> float:
        """
        Get video duration in seconds.
        
        Args:
            video_path: Path to video file
            ffmpeg_path: Path to FFmpeg executable
            
        Returns:
            Duration in seconds
        """
        cmd = [ffmpeg_path, '-i', video_path]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )
        
        # Parse duration from FFmpeg output
        import re
        match = re.search(r"Duration: (\d{2}):(\d{2}):(\d{2}\.\d+)", result.stderr)
        if match:
            hours, minutes, seconds = map(float, match.groups())
            return hours * 3600 + minutes * 60 + seconds
        
        return 0.0

    def _get_video_framerate(self, video_path: str, ffmpeg_path: str) -> float:
        """Get video framerate from FFmpeg stream info, fallback to 30fps."""
        cmd = [ffmpeg_path, '-i', video_path]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )

        import re
        stderr = result.stderr or ""
        match = re.search(r"(\d+(?:\.\d+)?)\s*fps", stderr)
        if match:
            try:
                fps = float(match.group(1))
                if fps > 0:
                    return fps
            except ValueError:
                pass

        logger.warning(f"[LPIPSAnalyzer] Could not parse FPS for {video_path}; defaulting to 30.0")
        return 30.0
    
    def _classify_drift_trend(self, scores: List[float]) -> str:
        """
        Classify drift pattern based on score progression.
        
        Args:
            scores: List of LPIPS scores over time
            
        Returns:
            Drift classification: 'stable', 'gradual', or 'sudden'
        """
        if len(scores) < 3:
            return 'stable'
        
        # Calculate rate of change
        differences = [scores[i+1] - scores[i] for i in range(len(scores)-1)]
        max_jump = max(abs(d) for d in differences)
        avg_change = np.mean(np.abs(differences))
        
        # Classify
        if max_jump > 0.2:
            return 'sudden'
        elif avg_change > 0.05:
            return 'gradual'
        else:
            return 'stable'


def get_summary_statistics(seam_analyses: List[SeamAnalysis]) -> Dict:
    """
    Generate summary statistics from seam analysis results.
    
    Args:
        seam_analyses: List of SeamAnalysis objects
        
    Returns:
        Dictionary with summary metrics
    """
    if not seam_analyses:
        return {
            'total_seams': 0,
            'mean_score': 0.0,
            'max_score': 0.0,
            'min_score': 0.0,
            'std_score': 0.0,
            'quality_distribution': {}
        }
    
    scored_analyses = [s for s in seam_analyses if s.lpips_score is not None and s.lpips_score >= 0]
    scores = [float(s.lpips_score) for s in scored_analyses if s.lpips_score is not None]
    quality_counts = {}
    for s in seam_analyses:
        quality_counts[s.quality_rating] = quality_counts.get(s.quality_rating, 0) + 1

    if not scores:
        return {
            'total_seams': len(seam_analyses),
            'mean_score': 0.0,
            'max_score': 0.0,
            'min_score': 0.0,
            'std_score': 0.0,
            'quality_distribution': quality_counts,
            'worst_seam_index': -1,
            'best_seam_index': -1
        }
    
    return {
        'total_seams': len(seam_analyses),
    'mean_score': float(np.mean(scores)),
    'max_score': float(np.max(scores)),
    'min_score': float(np.min(scores)),
    'std_score': float(np.std(scores)),
        'quality_distribution': quality_counts,
        'worst_seam_index': int(np.argmax(scores)),
        'best_seam_index': int(np.argmin(scores))
    }
