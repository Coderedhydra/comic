#!/usr/bin/env bash
set -euo pipefail

export DATA_OUTPUTS=${DATA_OUTPUTS:-/workspace/data/outputs}
uvicorn backend.server:app --host 0.0.0.0 --port 8000

