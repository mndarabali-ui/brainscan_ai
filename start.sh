#!/usr/bin/env bash
set -e
PORT=${PORT:-8000}
uvicorn src.app:app --host 0.0.0.0 --port "$PORT"
