import os
import glob
import subprocess
from typing import Tuple
import folder_paths
from .utils import ffmpeg_path, strip_path, validate_path, ENCODE_ARGS


class VideoMerger:
    """
    Merges multiple video files from a directory into a single video with optional trimming and audio.
    Perfect for combining frame-by-frame processed video chunks.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {
                    "default": "longform_cache_processed/my_project",
                    "placeholder": "output/my_project",
                    "tooltip": "Relative path to folder containing MP4 files (relative to output directory)"
                }),
                "output_filename": ("STRING", {
                    "default": "merged_video",
                    "tooltip": "Name for the final merged video (without extension)"
                }),
                "fps": ("INT", {
                    "default": 30,
                    "min": 1,
                    "max": 120,
                    "step": 1,
                    "tooltip": "Frame rate for trimming calculations"
                }),
                "chunk_size": ("INT", {
                    "default": 60,
                    "min": 1,
                    "max": 1000,
                    "step": 1,
                    "tooltip": "Original size of each video chunk in frames"
                }),
                "overlap_frames": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Total overlapping frames. Will be split evenly (half cut from end of clip A, half from start of clip B)"
                }),
            },
            "optional": {
                "audio_source": ("STRING", {
                    "default": "",
                    "placeholder": "input/audio.mp4",
                    "tooltip": "Path to audio file (leave empty for silent video)"
                }),
                "save_output": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Save to output directory (False = temp directory)"
                }),
            },
        }

    RETURN_TYPES = ("STRING", "VHS_FILENAMES")
    RETURN_NAMES = ("merged_video_path", "filenames")
    FUNCTION = "merge_videos"
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢"
    OUTPUT_NODE = True

    def merge_videos(
        self,
        folder_path: str,
        output_filename: str,
        fps: int = 30,
        chunk_size: int = 60,
        overlap_frames: int = 0,
        audio_source: str = "",
        save_output: bool = True,
    ) -> Tuple[str, Tuple]:
        """
        Merge multiple video files from a directory into a single video.
        
        Args:
            folder_path: Relative path to folder containing videos
            output_filename: Name for output file (without extension)
            fps: Frame rate for trimming calculations
            chunk_size: Original frames per chunk
            overlap_frames: Total overlap (split evenly between clips)
            audio_source: Optional audio file path
            save_output: Save to output vs temp directory
            
        Returns:
            Tuple of (path to merged video, VHS_FILENAMES object)
        """
        
        if ffmpeg_path is None:
            raise RuntimeError("FFmpeg is required for Video Merger. Please install FFmpeg.")
        
        # Setup directories
        output_dir = (
            folder_paths.get_output_directory()
            if save_output
            else folder_paths.get_temp_directory()
        )
        
        # Clean and validate folder path
        folder_path = strip_path(folder_path)
        input_dir = os.path.join(output_dir, folder_path)
        
        if not os.path.exists(input_dir):
            raise FileNotFoundError(f"Folder not found: {input_dir}")
        
        # Find all MP4 files and sort them
        files = sorted(glob.glob(os.path.join(input_dir, "*.mp4")))
        
        if not files:
            raise FileNotFoundError(f"No MP4 files found in: {input_dir}")
        
        print(f"[VideoMerger] Found {len(files)} video files to merge")
        
        # Setup output paths
        temp_video = os.path.join(output_dir, f"{output_filename}_temp.mp4")
        final_output = os.path.join(output_dir, f"{output_filename}.mp4")
        list_file = os.path.join(input_dir, "concat_list.txt")
        
        # Create concatenation list with trimming
        self._create_concat_list(
            files, list_file, fps, chunk_size, overlap_frames
        )
        
        # Concatenate videos
        print(f"[VideoMerger] Merging videos...")
        self._concat_videos(list_file, temp_video)
        
        # Add audio if provided
        if audio_source and audio_source.strip():
            audio_path = self._resolve_audio_path(audio_source)
            if audio_path and os.path.exists(audio_path):
                print(f"[VideoMerger] Adding audio from: {audio_path}")
                self._add_audio(temp_video, audio_path, final_output)
            else:
                print(f"[VideoMerger] Audio file not found, saving without audio")
                if os.path.exists(final_output):
                    os.remove(final_output)
                os.rename(temp_video, final_output)
        else:
            print(f"[VideoMerger] No audio source specified")
            if os.path.exists(final_output):
                os.remove(final_output)
            os.rename(temp_video, final_output)
        
        # Cleanup
        if os.path.exists(list_file):
            os.remove(list_file)
        if os.path.exists(temp_video) and temp_video != final_output:
            os.remove(temp_video)
        
        print(f"[VideoMerger] Merge complete: {final_output}")
        
        # Return in VHS_FILENAMES format
        vhs_filenames = (save_output, [final_output])
        
        return (final_output, vhs_filenames)
    
    def _create_concat_list(
        self,
        files: list,
        list_file: str,
        fps: int,
        chunk_size: int,
        overlap_frames: int
    ) -> None:
        """
        Create FFmpeg concat demuxer file with intelligent overlap trimming.
        
        The overlap is split evenly:
        - Half removed from END of previous clip
        - Half removed from START of next clip
        
        Example: overlap_frames=20 means 10 frames cut from each side.
        """
        
        # Calculate half overlap (will be cut from each side)
        half_overlap = int(overlap_frames / 2)
        
        if overlap_frames > 0:
            print(f"[VideoMerger] Merging {len(files)} files with {overlap_frames} frame overlap "
                  f"(cutting {half_overlap} from end/start of each clip)")
        else:
            print(f"[VideoMerger] Merging {len(files)} files without overlap trimming")
        
        with open(list_file, "w") as f:
            for i, file_path in enumerate(files):
                # Convert to absolute path and escape for FFmpeg
                abs_path = os.path.abspath(file_path).replace("\\", "/")
                # Escape single quotes for FFmpeg
                abs_path = abs_path.replace("'", "'\\''")
                
                f.write(f"file '{abs_path}'\n")
                
                # --- Start Cut (inpoint) ---
                # Cut FIRST half of overlap from START of this clip
                # Don't cut the first file's start
                current_start_cut = 0
                if i > 0:
                    current_start_cut = half_overlap
                    inpoint_timestamp = current_start_cut / fps
                    f.write(f"inpoint {inpoint_timestamp:.3f}\n")
                
                # --- End Cut (duration) ---
                # Cut SECOND half of overlap from END of this clip
                # Don't cut the last file's end
                if i < len(files) - 1:
                    # Duration = Total Frames - Start Cut - End Cut
                    frames_to_play = chunk_size - current_start_cut - half_overlap
                    play_duration = frames_to_play / fps
                    f.write(f"duration {play_duration:.3f}\n")
    
    def _concat_videos(self, list_file: str, output_path: str) -> None:
        """Concatenate videos using FFmpeg concat demuxer."""
        
        cmd = [
            ffmpeg_path,
            "-y",  # Overwrite output
            "-v", "error",  # Only show errors
            "-f", "concat",  # Use concat demuxer
            "-safe", "0",  # Allow absolute paths
            "-i", list_file,
            "-c", "copy",  # Copy streams without re-encoding
            output_path
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.decode(*ENCODE_ARGS) if e.stderr else "Unknown error"
            raise RuntimeError(f"FFmpeg concatenation failed:\n{error_msg}")
    
    def _add_audio(
        self,
        video_path: str,
        audio_path: str,
        output_path: str
    ) -> None:
        """Add audio track to video."""
        
        cmd = [
            ffmpeg_path,
            "-y",  # Overwrite output
            "-v", "error",  # Only show errors
            "-i", video_path,  # Video input
            "-i", audio_path,  # Audio input
            "-c:v", "copy",  # Copy video stream
            "-c:a", "aac",  # Encode audio to AAC
            "-map", "0:v:0",  # Use video from first input
            "-map", "1:a:0",  # Use audio from second input
            "-shortest",  # End when shortest stream ends
            output_path
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.decode(*ENCODE_ARGS) if e.stderr else "Unknown error"
            raise RuntimeError(f"FFmpeg audio muxing failed:\n{error_msg}")
    
    def _resolve_audio_path(self, audio_source: str) -> str:
        """Resolve audio file path, checking multiple locations."""
        
        audio_path = strip_path(audio_source)
        
        # Check if absolute path exists
        if os.path.exists(audio_path):
            return audio_path
        
        # Check in input directory
        input_dir = folder_paths.get_input_directory()
        alt_path = os.path.join(input_dir, os.path.basename(audio_path))
        if os.path.exists(alt_path):
            return alt_path
        
        # Check as relative to input directory
        rel_path = os.path.join(input_dir, audio_path)
        if os.path.exists(rel_path):
            return rel_path
        
        return None


# Node registration mapping
NODE_CLASS_MAPPINGS = {
    "VHS_VideoMerger": VideoMerger,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VHS_VideoMerger": "Video Merger 🎥🅥🅗🅢",
}
