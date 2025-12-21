import os
import subprocess
import shutil
import random
import torch
import numpy as np
import gc
import folder_paths
import comfy.model_management
import cv2

class VideoChunker:
    """
    Splits video into chunks on the HARD DISK (Output folder) to avoid RAM disks.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video_path": ("STRING", {"forceInput": False, "default": "", "multiline": False}),
                "segment_duration": ("INT", {"default": 2, "min": 1, "max": 30, "step": 1}),
                "overlap_sec": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 5.0, "step": 0.1}),
                "unique_id": ("STRING", {"default": "my_project"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("chunk_paths",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "execute_chunking"
    CATEGORY = "LongFormVideo"

    def execute_chunking(self, video_path, segment_duration, overlap_sec, unique_id):
        if not os.path.isfile(video_path):
            input_dir = folder_paths.get_input_directory()
            potential_path = os.path.join(input_dir, str(video_path).strip('"'))
            if os.path.isfile(potential_path):
                video_path = potential_path
            else:
                raise FileNotFoundError(f"Video file not found: {video_path}")

        base_output_dir = folder_paths.get_output_directory()
        cache_dir = os.path.join(base_output_dir, "longform_cache_input", unique_id)
        
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
        os.makedirs(cache_dir, exist_ok=True)

        total_duration = 0.0
        try:
            probe_cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                         "-of", "default=noprint_wrappers=1:nokey=1", video_path]
            total_duration = float(subprocess.check_output(probe_cmd).decode().strip())
        except Exception:
            if cv2:
                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                total_duration = frames / fps if fps > 0 else 0
                cap.release()

        if total_duration <= 0:
            raise RuntimeError("Could not determine video duration.")

        chunk_paths = [] 
        current_time = 0.0
        chunk_idx = 0

        print(f"[Chunker] Writing to Disk: {cache_dir}")

        while current_time < total_duration:
            if total_duration - current_time < 0.2: break

            clip_start = current_time
            clip_duration = min(segment_duration + overlap_sec, total_duration - clip_start)
            
            output_file = os.path.join(cache_dir, f"chunk_{chunk_idx:04d}.mp4")

            cmd = [
                "ffmpeg", "-y", "-v", "error",
                "-ss", f"{clip_start:.3f}",
                "-t", f"{clip_duration:.3f}",
                "-i", video_path,
                "-c:v", "libx264", "-crf", "18", "-preset", "ultrafast",
                "-c:a", "aac",
                output_file
            ]
            subprocess.check_call(cmd)

            chunk_paths.append(output_file)
            current_time += segment_duration
            chunk_idx += 1

        return (chunk_paths,)

class LoadVideoSingleton:
    """
    Loads one chunk from Disk. Includes Resize to save RAM.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video_path": ("STRING", {"forceInput": True}),
                "frame_load_cap": ("INT", {"default": 0, "min": 0}),
                "force_resize_width": ("INT", {"default": 0, "min": 0, "max": 4096}),
                "force_resize_height": ("INT", {"default": 0, "min": 0, "max": 4096}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("images", "frame_count", "fps")
    FUNCTION = "load_video_safe"
    CATEGORY = "LongFormVideo"

    def load_video_safe(self, video_path, frame_load_cap, force_resize_width, force_resize_height):
        gc.collect()
        comfy.model_management.soft_empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Chunk missing: {video_path}")

        if cv2 is None:
            raise ImportError("OpenCV (cv2) is required for LoadVideoSingleton.")

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = [] 

        count = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            if force_resize_width > 0 and force_resize_height > 0:
                frame = cv2.resize(frame, (force_resize_width, force_resize_height), interpolation=cv2.INTER_AREA)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(torch.from_numpy(frame.astype(np.float32) / 255.0))
            count += 1
            if frame_load_cap > 0 and count >= frame_load_cap: break
        
        cap.release()

        if not frames: raise ValueError(f"No frames in {video_path}")

        images = torch.stack(frames)
        
        ram_usage = images.element_size() * images.nelement() / (1024 * 1024)
        print(f"[Loader] Loaded {len(frames)} frames. RAM Usage: {ram_usage:.2f} MB")

        return (images, len(frames), int(round(fps)))

class SimpleVideoSaver:
    """
    Saves processed chunks to HARD DISK (Output folder).
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "frame_rate": ("INT", {"default": 24, "min": 1, "max": 120}),
                "unique_id": ("STRING", {"default": "my_project"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    FUNCTION = "save_video"
    CATEGORY = "LongFormVideo"
    OUTPUT_NODE = True

    def save_video(self, images, frame_rate, unique_id):
        base_output_dir = folder_paths.get_output_directory()
        cache_dir = os.path.join(base_output_dir, "longform_cache_processed", unique_id)
        os.makedirs(cache_dir, exist_ok=True)
        
        filename = f"processed_{random.randint(100000, 999999)}.mp4"
        output_path = os.path.join(cache_dir, filename)

        if isinstance(images, list): images = torch.stack(images)

        np_images = (images.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        
        del images
        
        b, h, w, c = np_images.shape
        pad_h = h % 2
        pad_w = w % 2
        if pad_h or pad_w:
            new_h, new_w = h + pad_h, w + pad_w
            padded = np.zeros((b, new_h, new_w, 3), dtype=np.uint8)
            padded[:, :h, :w, :] = np_images
            np_images = padded

        cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{np_images.shape[1]}x{np_images.shape[2]}", "-r", str(frame_rate),
            "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "17",
            output_path
        ]

        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        for frame in np_images:
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
        proc.wait()

        del np_images
        gc.collect() 

        return (output_path,)

class VideoMerger:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video_paths": ("STRING", {"forceInput": True}),
                "filename_prefix": ("STRING", {"default": "wan_long_final"}),
                "cleanup_cache": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("merged_video_path",)
    INPUT_IS_LIST = True
    FUNCTION = "merge"
    CATEGORY = "LongFormVideo"

    def merge(self, video_paths, filename_prefix, cleanup_cache):
        if isinstance(filename_prefix, list): filename_prefix = filename_prefix[0]
        if isinstance(cleanup_cache, list): cleanup_cache = cleanup_cache[0]

        valid_paths = [p for p in video_paths if isinstance(p, str) and os.path.exists(p)]
        valid_paths.sort()

        if not valid_paths: return ("",)

        output_file = os.path.join(folder_paths.get_output_directory(),
                                   f"{filename_prefix}_{random.randint(1000, 9999)}.mp4")

        list_file = os.path.join(folder_paths.get_output_directory(), f"concat_list_{random.randint(0,99999)}.txt")
        
        with open(list_file, "w") as f:
            for p in valid_paths:
                safe_path = p.replace("'", "'\"'\"'")
                f.write(f"file '{safe_path}'\n")

        subprocess.check_call([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_file,
            "-c", "copy", output_file
        ])

        if os.path.exists(list_file): os.remove(list_file)
        
        if cleanup_cache:
            print("[Merger] Cleaning up intermediate cache files on disk...")
            for p in valid_paths:
                try:
                    os.remove(p)
                except OSError:
                    pass

        return (output_file,)

NODE_CLASS_MAPPINGS = {
    "VideoChunker": VideoChunker,
    "LoadVideoSingleton": LoadVideoSingleton,
    "SimpleVideoSaver": SimpleVideoSaver,
    "VideoMerger": VideoMerger
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoChunker": "Video Chunker",
    "LoadVideoSingleton": "Load Video Chunk ",
    "SimpleVideoSaver": "Simple Video Saver ",
    "VideoMerger": "Video Merger"
}