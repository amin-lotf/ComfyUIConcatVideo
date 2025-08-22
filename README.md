```markdown
# comfyui-concat-video


Concatenate two image batches in ComfyUI and export as a single video. Useful when you render two clips separately (e.g., from two VAE Decode nodes) and want a single final video.


## Features
- Appends batch **B** after batch **A** (end-to-end concat)
- Ensures frame sizes match; optional auto-resize of the second batch to match the first
- Keeps tensor dtype/range expected by ComfyUI (`0..1`, `B x H x W x C`)
- Plays nice with the built-in **Images to Video** node


## Install


> Assuming your ComfyUI root is `~/ComfyUI`.


```bash
cd ~/ComfyUI/custom_nodes
git clone https://github.com/YOUR_USER/comfyui-concat-video.git
# (Optional) install extras if you plan to run tests or extend
pip install -r comfyui-concat-video/requirements.txt
```


Restart ComfyUI. You should see a new node under **Video/Batches → Concat Image Batches**.


## Usage
1. Create two branches that each result in an **IMAGE** batch (e.g., `VAE Decode → (IMAGE)` for clip A and clip B).
2. Feed both into **Concat Image Batches**.
3. Send the result into ComfyUI's **Images to Video** node to render a single video.


> **Note:** If the two branches produce different frame counts, all frames from both are appended. If sizes differ, enable *ensure_same_size* to auto-resize B to A using bilinear interpolation.


## Node I/O
- **Inputs:**
- `images_a: IMAGE` (required)
- `images_b: IMAGE` (required)
- `ensure_same_size: BOOLEAN` (default: `True`)
- `resize_method: ["fit","pad","stretch"]` (default: `fit`)
- **Output:**
- `IMAGE` (concatenated batch: A then B)


### Resize methods
- **fit**: Resizes B to A while preserving aspect ratio (letterboxed). Pads with black bars.
- **pad**: No resize; pads B to A's dimensions if smaller, or center-crops if larger.
- **stretch**: Directly resizes B to A's dimensions (may distort).


## Example
After concatenation, connect to **Images to Video** and set FPS (e.g., 24 or 30) and container (mp4). The timestamps will be continuous because the frames are just appended.