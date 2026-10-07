#!/bin/sh
set -eu
project=${1:?usage: scripts/verify.sh OUTPUT_DIR}
exec python3 "$(dirname "$0")/../verify_project.py" "$project"
