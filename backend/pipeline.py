import os
import json
import math
import pathlib
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from dotenv import load_dotenv


# -------------------------------
# Configuration
# -------------------------------

@dataclass
class PipelineConfig:
    input_path: str
    output_dir: str
    target_page_count: Optional[int] = None
    frames_per_page: int = 4  # 2x2
    page_width: int = 400
    page_height: int = 540
    enhancer_backend: str = "opencv_sr"  # opencv_sr | openai | stability | realesrgan
    model_vision: str = os.getenv("MODEL_VISION", "gpt-4o-mini")
    model_llm: str = os.getenv("MODEL_LLM", "gpt-4.1-mini")
    max_frames: Optional[int] = None
    fps_sample: float = 1.0


# -------------------------------
# Utilities
# -------------------------------

def ensure_dir(path: str) -> None:
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def write_json(path: str, data: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def read_video_metadata(input_path: str) -> Tuple[float, int]:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {input_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return fps, frames


def extract_frames(input_path: str, frames_dir: str, fps_sample: float = 1.0, max_frames: Optional[int] = None) -> List[str]:
    ensure_dir(frames_dir)
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {input_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    stride = max(1, int(round(video_fps / max(0.001, fps_sample))))

    saved_paths: List[str] = []
    idx = 0
    frame_idx = 0
    with tqdm(desc="Extracting frames", unit="f") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % stride == 0:
                out_path = os.path.join(frames_dir, f"frame_{idx:06d}.jpg")
                cv2.imwrite(out_path, frame)
                saved_paths.append(out_path)
                idx += 1
                pbar.update(1)
                if max_frames is not None and idx >= max_frames:
                    break
            frame_idx += 1
    cap.release()
    return saved_paths


# -------------------------------
# Emotional frame selection (heuristic)
# -------------------------------

def score_emotion(frame_bgr: np.ndarray) -> float:
    # Simple heuristic: high contrast + face-like regions estimation using Laplacian variance
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    contrast = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    # Saturation proxy
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    saturation = float(np.mean(hsv[:, :, 1]))
    return 0.7 * contrast + 0.3 * saturation


def select_emotional_frames(frame_paths: List[str], desired_count: int) -> List[str]:
    scored: List[Tuple[str, float]] = []
    for p in tqdm(frame_paths, desc="Scoring emotion"):
        img = cv2.imread(p)
        if img is None:
            continue
        s = score_emotion(img)
        scored.append((p, s))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [p for p, _ in scored[:desired_count]]


# -------------------------------
# Enhancement backends (stubs + simple upscaler as fallback)
# -------------------------------

def enhance_image(image_path: str, out_path: str, backend: str = "openai") -> None:
    # Try configured backend, then fallback to simple upscale
    ok = False
    try:
        if backend == "opencv_sr":
            from enhancers import enhance_with_opencv_sr
            ok = enhance_with_opencv_sr(image_path, out_path)
        elif backend == "realesrgan":
            from enhancers import enhance_with_realesrgan
            ok = enhance_with_realesrgan(image_path, out_path)
        elif backend == "openai":
            from enhancers import enhance_with_openai
            ok = enhance_with_openai(image_path, out_path)
        elif backend == "stability":
            from enhancers import enhance_with_stability
            ok = enhance_with_stability(image_path, out_path)
    except Exception:
        ok = False

    if not ok:
        try:
            image = Image.open(image_path).convert("RGB")
            image = image.resize((image.width * 2, image.height * 2), Image.LANCZOS)
            image.save(out_path, format="JPEG", quality=92)
            ok = True
        except Exception:
            img = Image.open(image_path).convert("RGB")
            img.save(out_path, format="JPEG", quality=92)


def enhance_batch(input_paths: List[str], enhanced_dir: str, backend: str) -> List[str]:
    ensure_dir(enhanced_dir)
    outputs: List[str] = []
    for p in tqdm(input_paths, desc="Enhancing"):
        name = os.path.basename(p)
        out_p = os.path.join(enhanced_dir, name)
        enhance_image(p, out_p, backend)
        outputs.append(out_p)
    return outputs


# -------------------------------
# Captioning / Summarization (stub; front-end edits later)
# -------------------------------

def chunk_into_pages(items: List[str], frames_per_page: int) -> List[List[str]]:
    pages: List[List[str]] = []
    for i in range(0, len(items), frames_per_page):
        pages.append(items[i : i + frames_per_page])
    return pages


def compute_target_pages(video_frames: int, video_fps: float, requested_pages: Optional[int]) -> int:
    if requested_pages is not None and requested_pages > 0:
        return requested_pages
    # scale pages roughly: 1 page per ~6 seconds, min 12
    seconds = video_frames / max(1.0, video_fps)
    pages = max(12, int(math.ceil(seconds / 6.0)))
    return pages


def dummy_captions_for_page(num_frames: int) -> List[str]:
    captions = []
    for i in range(num_frames):
        captions.append("...")
    return captions


def run_pipeline(cfg: PipelineConfig) -> None:
    load_dotenv()
    ensure_dir(cfg.output_dir)
    frames_dir = os.path.join(cfg.output_dir, "frames")
    enhanced_dir = os.path.join(cfg.output_dir, "enhanced")
    pages_dir = os.path.join(cfg.output_dir, "pages")
    ensure_dir(pages_dir)

    fps, total_frames = read_video_metadata(cfg.input_path)
    target_pages = compute_target_pages(total_frames, fps, cfg.target_page_count)
    desired_frames = target_pages * cfg.frames_per_page

    extracted = extract_frames(cfg.input_path, frames_dir, fps_sample=cfg.fps_sample, max_frames=None)
    selected = select_emotional_frames(extracted, desired_count=min(len(extracted), desired_frames))
    enhanced = enhance_batch(selected, enhanced_dir, cfg.enhancer_backend)

    pages = chunk_into_pages(enhanced, cfg.frames_per_page)

    # Minimal captions structure for front-end editing
    pages_payload: List[Dict] = []
    for page_idx, frame_paths in enumerate(pages):
        pages_payload.append(
            {
                "page_index": page_idx,
                "frames": [
                    {"path": os.path.relpath(p, cfg.output_dir), "caption": ""}
                    for p in frame_paths
                ],
                "bubbles": [
                    # default: one bubble per frame, editable in UI
                    {
                        "text": "",
                        "frame_index": fi,
                        "x": 10,
                        "y": 10,
                        "width": 160,
                        "height": 28,
                    }
                    for fi in range(len(frame_paths))
                ],
            }
        )

    write_json(
        os.path.join(cfg.output_dir, "project.json"),
        {
            "base_url": "/outputs",
            "page_width": cfg.page_width,
            "page_height": cfg.page_height,
            "frames_per_page": cfg.frames_per_page,
            "pages": pages_payload,
        },
    )
    print(f"Wrote project.json with {len(pages_payload)} pages to {cfg.output_dir}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--target_page_count", type=int, default=None)
    parser.add_argument("--fps_sample", type=float, default=1.0)
    parser.add_argument("--enhancer_backend", type=str, default=os.getenv("ENHANCER_BACKEND", "openai"))
    parser.add_argument("--frames_per_page", type=int, default=4)
    args = parser.parse_args()

    cfg = PipelineConfig(
        input_path=args.input,
        output_dir=args.out_dir,
        target_page_count=args.target_page_count,
        fps_sample=args.fps_sample,
        enhancer_backend=args.enhancer_backend,
        frames_per_page=args.frames_per_page,
    )
    run_pipeline(cfg)


if __name__ == "__main__":
    main()

