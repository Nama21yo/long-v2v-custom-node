import os
import torch
from typing import Tuple, Dict, Any, Optional
from .utils import BIGMAX, DIMMAX, strip_path, validate_path, hash_path
from .load_video_nodes import load_video, ffmpeg_frame_generator


class FrameByFrameVideoProcessor:
    """
    Frame-by-frame video processor with automatic batch progression and
    robust handling for image vs VAE-latent outputs.

    Features:
    - Keeps a running frame counter (self.current_frame) and total_frames.
    - Converts requested skip frames -> ffmpeg start_time using source fps
      or forced framerate.
    - Returns actual batch frame count as an INT output (for Wan2.1 shape fixes).
    - Optionally auto-requeues the workflow when a batch completes.
    """

    def __init__(self):
        self.current_frame = 0
        self.total_frames = 0
        self.batch_size = 1
        self.is_complete = False
        self._source_fps = 0  # store source fps for frame->time conversion

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {
                    "placeholder": "X://path/to/video.mp4",
                    "vhs_path_extensions": ['webm', 'mp4', 'mkv', 'gif', 'mov']
                }),
                "frames_per_batch": ("INT", {
                    "default": 16,
                    "min": 1,
                    "max": 500,
                    "step": 1,
                    "tooltip": "Number of frames to process in each execution"
                }),
                "force_rate": ("FLOAT", {
                    "default": 0,
                    "min": 0,
                    "max": 60,
                    "step": 1,
                    "disable": 0
                }),
                "custom_width": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": DIMMAX,
                    "disable": 0
                }),
                "custom_height": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": DIMMAX,
                    "disable": 0
                }),
                "auto_continue": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Automatically queue next batch when current completes"
                }),
            },
            "optional": {
                "vae": ("VAE",),
                "reset_counter": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Reset frame counter to start from beginning"
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            }
        }

    # Added INT at index 2: batch_frame_count
    RETURN_TYPES = (
        "IMAGE", "LATENT", "INT", "INT", "INT", "INT", "BOOLEAN", "VHS_VIDEOINFO", "AUDIO"
    )
    RETURN_NAMES = (
        "frames",
        "latents",
        "batch_frame_count",      # actual frames delivered in this batch
        "current_frame_index",
        "batch_count",
        "frames_remaining",
        "is_complete",
        "video_info",
        "audio"
    )

    FUNCTION = "process_frame_batch"
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢"
    OUTPUT_NODE = True

    def process_frame_batch(
        self,
        video_path: str,
        frames_per_batch: int = 16,
        force_rate: float = 0,
        custom_width: int = 0,
        custom_height: int = 0,
        auto_continue: bool = True,
        vae: Optional[Any] = None,
        reset_counter: bool = False,
        unique_id: Optional[str] = None,
        prompt: Optional[Dict] = None,
        extra_pnginfo: Optional[Dict] = None,
    ) -> Tuple[Any, ...]:
        """
        Process one batch of frames, returning:
        (images, latents, batch_frame_count, current_frame_index, batch_count,
         frames_remaining, is_complete, video_info, audio)
        """

        # Normalize and validate path
        video_path = strip_path(video_path)
        if not validate_path(video_path):
            raise ValueError(f"Invalid video path: {video_path}")

        # Reset if requested
        if reset_counter:
            self.current_frame = 0
            self.is_complete = False
            print("[FrameProcessor] Counter reset to frame 0")

        # Load metadata if first run or after reset
        if self.total_frames == 0 or reset_counter:
            self._load_video_metadata(
                video_path=video_path,
                force_rate=force_rate,
                custom_width=custom_width,
                custom_height=custom_height,
                vae=vae
            )

        start_frame = self.current_frame
        end_frame = min(start_frame + frames_per_batch, self.total_frames)
        frames_to_load = max(0, end_frame - start_frame)

        print(f"[FrameProcessor] Processing frames {start_frame} to {end_frame - 1} "
              f"({frames_to_load} frames requested)")

        # If nothing to process, return empty result
        if frames_to_load <= 0 or start_frame >= self.total_frames:
            self.is_complete = True
            print(f"[FrameProcessor] No frames to process (start {start_frame}, total {self.total_frames})")
            return self._return_empty_result()

        # Load frames using load_video helper (converts skip_frames to start_time)
        frames, frame_count_loaded, audio, video_info = self._load_frame_batch(
            video_path=video_path,
            skip_frames=start_frame,
            frame_count=frames_to_load,
            force_rate=force_rate,
            custom_width=custom_width,
            custom_height=custom_height,
            vae=vae
        )

        # Determine the actual number of frames returned in a robust way
        actual_frame_count = self._determine_frame_count(frames, frame_count_loaded)

        # Update counters
        self.current_frame = start_frame + actual_frame_count
        batch_count = (start_frame // frames_per_batch) + 1
        frames_remaining = max(0, self.total_frames - self.current_frame)
        self.is_complete = (self.current_frame >= self.total_frames)

        # Prepare images / latents placeholders if necessary
        if vae is not None and isinstance(frames, dict):
            latents = frames
            # create a placeholder image array if downstream expects images shape
            images = torch.zeros((actual_frame_count, 64, 64, 3))
        else:
            images = frames
            # placeholder latents shape (kept minimal)
            latents = {"samples": torch.zeros((1, 4, 64, 64))}

        # Progress (guard divide-by-zero)
        if self.total_frames > 0:
            progress_pct = (self.current_frame / self.total_frames) * 100
        else:
            progress_pct = 100.0
        print(f"[FrameProcessor] Progress: {progress_pct:.1f}% ({self.current_frame}/{self.total_frames} frames)")

        # Auto-requeue next batch if applicable
        if not self.is_complete and auto_continue:
            self._requeue_workflow(unique_id=unique_id, prompt=prompt)

        return (
            images,
            latents,
            int(actual_frame_count),  # integer number of frames returned
            int(start_frame),
            int(batch_count),
            int(frames_remaining),
            bool(self.is_complete),
            video_info,
            audio
        )

    def _load_video_metadata(
        self,
        video_path: str,
        force_rate: float,
        custom_width: int,
        custom_height: int,
        vae: Optional[Any]
    ) -> None:
        """
        Loads minimal frames (frame_load_cap=1) to retrieve video_info metadata.
        """
        from .load_video_nodes import load_video, ffmpeg_frame_generator

        # Use start_time=0.0 (ffmpeg) to ask for metadata via a single-frame load
        _, _, _, video_info = load_video(
            video=video_path,
            force_rate=force_rate,
            custom_width=custom_width,
            custom_height=custom_height,
            frame_load_cap=1,
            start_time=0.0,
            vae=vae,
            generator=ffmpeg_frame_generator
        )

        # Store source fps and compute total frames
        self._source_fps = video_info.get('source_fps', 0) or 0
        if force_rate and force_rate > 0:
            # when force_rate is applied, total frames are duration * forced rate
            self.total_frames = int(video_info.get('source_duration', 0) * force_rate)
        else:
            self.total_frames = int(video_info.get('source_frame_count', 0) or 0)

        print(f"[FrameProcessor] Video loaded: {self.total_frames} total frames at {self._source_fps} fps")

    def _load_frame_batch(
        self,
        video_path: str,
        skip_frames: int,
        frame_count: int,
        force_rate: float,
        custom_width: int,
        custom_height: int,
        vae: Optional[Any]
    ) -> Tuple[Any, int, Dict[str, Any], Dict[str, Any]]:
        """
        Convert skip_frames -> start_time and call load_video to retrieve the batch.
        Returns: (frames, frame_count_loaded, audio, video_info)
        """
        from .load_video_nodes import load_video, ffmpeg_frame_generator

        # Compute start_time (seconds) for ffmpeg
        if force_rate and force_rate > 0:
            start_time = skip_frames / force_rate
        elif hasattr(self, '_source_fps') and self._source_fps and self._source_fps > 0:
            start_time = skip_frames / self._source_fps
        else:
            start_time = 0.0

        return load_video(
            video=video_path,
            force_rate=force_rate,
            custom_width=custom_width,
            custom_height=custom_height,
            frame_load_cap=frame_count,
            start_time=start_time,
            vae=vae,
            generator=ffmpeg_frame_generator
        ) # type: ignore

    @staticmethod
    def _determine_frame_count(frames: Any, frame_count_hint: int) -> int:
        """
        Robustly determine how many frames were actually returned.
        Handles:
         - torch.Tensor with shape (N, ...)
         - numpy.ndarray with .shape
         - list-like with len()
         - dict from VAE with 'samples' key (shape)
        Falls back to frame_count_hint if unable to detect.
        """
        try:
            # VAE dict case: expects 'samples' or similar
            if isinstance(frames, dict):
                if 'samples' in frames and hasattr(frames['samples'], 'shape'):
                    return int(frames['samples'].shape[0])
                # fallback to hint
                return int(frame_count_hint or 0)

            # torch.Tensor
            if isinstance(frames, torch.Tensor):
                return int(frames.shape[0])

            # numpy array
            import numpy as _np  # local import to avoid top-level dependency requirement
            if isinstance(frames, _np.ndarray):
                return int(frames.shape[0])

            # list/tuple
            if isinstance(frames, (list, tuple)):
                return len(frames)

            # some other object that exposes shape attribute
            if hasattr(frames, 'shape'):
                return int(frames.shape[0])

        except Exception:
            pass

        # fallback to provided hint (frame_count returned by load_video)
        try:
            return int(frame_count_hint or 0)
        except Exception:
            return 0

    def _return_empty_result(self) -> Tuple[Any, ...]:
        """Return zero-sized placeholders for each return slot."""
        empty_image = torch.zeros((1, 64, 64, 3))
        empty_latent = {"samples": torch.zeros((1, 4, 64, 64))}
        empty_audio = {'waveform': torch.zeros((1, 2, 1)), 'sample_rate': 44100}
        empty_video_info = {
            'source_fps': 0,
            'source_frame_count': 0,
            'source_duration': 0,
            'source_width': 0,
            'source_height': 0,
            'loaded_fps': 0,
            'loaded_frame_count': 0,
            'loaded_duration': 0,
            'loaded_width': 0,
            'loaded_height': 0,
        }

        return (
            empty_image,
            empty_latent,
            0,                  # batch_frame_count
            int(self.current_frame),  # current_frame_index
            0,                  # batch_count
            0,                  # frames_remaining
            True,               # is_complete
            empty_video_info,
            empty_audio
        )

    def _requeue_workflow(self, unique_id: Optional[str], prompt: Optional[Dict]) -> None:
        """Attempt to auto-requeue the calling workflow (no-ops if missing)."""
        if unique_id is None or prompt is None:
            return

        try:
            from .utils import requeue_workflow_unchecked
            print("[FrameProcessor] Auto-requeueing for next batch...")
            # original helper takes no args in the provided snippets
            requeue_workflow_unchecked()
        except Exception as e:
            print(f"[FrameProcessor] Failed to requeue: {str(e)}")

    @classmethod
    def IS_CHANGED(cls, video_path: str, **kwargs) -> str:
        return hash_path(video_path)

    @classmethod
    def VALIDATE_INPUTS(cls, video_path: str, **kwargs) -> bool:
        return validate_path(video_path, allow_none=False) # type: ignore


# Node registration mapping
NODE_CLASS_MAPPINGS = {
    "VHS_FrameByFrameProcessor": FrameByFrameVideoProcessor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VHS_FrameByFrameProcessor": "Frame-by-Frame Video Processor 🎥🅥🅗🅢",
}
