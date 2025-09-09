## Comic Summarizer & Layout (2x2 Grid, Draggable Message Strips)

This project creates comic-style pages from a video or image sequence:

- Extracts frames and selects emotional, reaction-heavy moments
- Summarizes into pages, scaling page count by video length
- Enhances frames with state-of-the-art AI image models
- Lays out 2x2 frames per page (size 400x540 PNG: 1 page = 4 frames)
- Renders rectangular "message strip" text bubbles (not round) that are draggable and editable (double-click)
- Exports each page as PNG and bundles into a ZIP

### Quick Start

1) Python env

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Back-end: Extract frames, select emotional moments, enhance, and generate captions

```
python backend/pipeline.py --input /workspace/data/inputs/video.mp4 --out_dir /workspace/data/outputs --target_page_count 48
```

- Page count scales automatically with video length; override via `--target_page_count`.
- Enhanced frames output: `/workspace/data/outputs/enhanced`
- Captions JSON: `/workspace/data/outputs/captions.json`

3) Front-end

```
cd web
npm i
npm run dev
```

Open the local URL, load the generated project data, arrange rectangular message strips, drag to position, double-click to edit, and export pages as PNG (400x540). All pages can also be downloaded as a ZIP.

### Notes

- Image Enhancement: configurable backends (OpenAI, Stability, or Real-ESRGAN) controlled via `.env` and CLI flags. Real-ESRGAN is used when available; otherwise, a high-quality resize fallback is used.
- LLM Summarization: uses GPT-style models for captions/summaries. Provide an API key in `.env`.
- Emotion Detection: combines vision embeddings and heuristic filtering for high-reaction frames.

### Project Layout

```
/workspace
  ├── backend
  │   ├── pipeline.py
  │   ├── enhancers/
  │   ├── summarizer/
  │   ├── selection/
  │   └── utils/
  ├── web
  │   ├── package.json
  │   ├── src/
  │   └── public/
  ├── data
  │   ├── inputs/
  │   └── outputs/
  │       ├── frames/
  │       ├── enhanced/
  │       └── pages/
  └── scripts/
```

### Environment

Create `/workspace/.env`:

```
OPENAI_API_KEY=...
STABILITY_API_KEY=...
ENHANCER_BACKEND=opencv_sr   # opencv_sr | openai | stability | realesrgan
MODEL_VISION=gpt-4o-mini
MODEL_LLM=gpt-4.1-mini

# Optional server
# DATA_OUTPUTS points FastAPI static server to outputs for easy access from web
DATA_OUTPUTS=/workspace/data/outputs
```

### Export Spec

- Page size: 400x540 PNG
- 2x2 frames per page (4 frames per page)
- Draggable rectangular message strips; editable on double-click
- Save as multiple PNGs and a ZIP archive

