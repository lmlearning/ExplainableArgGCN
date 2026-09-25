#!/usr/bin/env bash
# Create an isolated CPU environment without changing data or project files.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON:-python3}" "$SCRIPT_DIR/scripts/setup_cpu.py" "$@"
