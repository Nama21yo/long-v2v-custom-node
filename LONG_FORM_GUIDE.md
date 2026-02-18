# Long-Form Video-to-Video: The "Endless" Workflow

### **The Problem**

Standard AI video models (like Stable Video Diffusion, AnimateDiff, or WanVideo) have a "Context Limit." They can usually only generate 2–4 seconds of video before they run out of memory or hallucinate.

### **The Solution**

This workflow uses a **"Sliding Window"** technique to generate infinite-length videos. It breaks a long input video (e.g., 2 minutes) into small chunks (e.g., 2 seconds), processes them, and seamlessly stitches them back together.

### **How It Works (Step-by-Step)**

#### **1. Input & Slicing (`FrameByFrameProcessor`)**

- **Input:** You load your long source video (e.g., `my_dance.mp4`).
- **Windowing:** The processor slices the video into batches.
- _Batch Size:_ 24 frames.
- _Context Overlap:_ 8 frames.

- **Why Overlap?** The last 8 frames of _Batch 1_ become the "context" for the start of _Batch 2_. This ensures the AI knows what the character was doing, preventing sudden jumps.

#### **2. Processing (The AI Model)**

- The standard Image-to-Video model (WanVideo/AnimateDiff) processes just one small batch at a time.
- **VRAM Impact:** Low. Even for a 1-hour video, the GPU only ever "sees" 24 frames at a time.

#### **3. Quality Check (`LPIPS Analyzer`)**

- Before saving, the workflow compares the generated frames against the original source structure.
- **Low Score (0.1):** Good. The AI followed the movement perfectly.
- **High Score (0.5+):** Bad. The AI hallucinated or lost the subject. You can use this score to automatically reject bad batches.

#### **4. Stitching (`Video Merger`)**

- The node takes the processed batches and reassembles them.
- **The "Seam" Fix:** It takes the _Overlap_ area (where Batch 1 ends and Batch 2 starts) and blends them using a **Linear Cross-Fade**. This makes the transition invisible to the human eye.

### **Recommended Settings**

| Parameter           | Recommended Value | Why?                                                          |
| ------------------- | ----------------- | ------------------------------------------------------------- |
| **Batch Size**      | 16 - 32 Frames    | Keeps VRAM usage low (fits on 12GB - 24GB cards).             |
| **Overlap**         | 4 - 8 Frames      | Sufficient context for the AI to maintain motion continuity.  |
| **Denoise**         | 0.35 - 0.45       | High enough to style the video, low enough to keep structure. |
| **LPIPS Threshold** | 0.3               | Values above this usually indicate the video has "broken."    |

### **Troubleshooting**

- **"Ghosting" at seams:** Increase your `Overlap` frames.
- **OOM (Out of Memory):** Reduce `Batch Size` (e.g., from 32 down to 16).
