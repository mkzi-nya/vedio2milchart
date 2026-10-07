#!/bin/zsh
cd "${0:A:h}"
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
RUNTIME="${0:A:h}/.venv/bin/python"
if [[ ! -x "$RUNTIME" ]]; then RUNTIME="$(command -v python3)"; fi
if [[ -z "$RUNTIME" ]]; then print -u2 'Python 3 is required. Install it and the packages in requirements.txt.'; exit 1; fi
open 'http://127.0.0.1:8765'
exec "$RUNTIME" server.py 8765
