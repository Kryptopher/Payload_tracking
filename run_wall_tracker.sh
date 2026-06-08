#!/usr/bin/env bash
# Standalone OAK-D AprilTag wall/cube tracker.
#
#   ./run_wall_tracker.sh
#   ./run_wall_tracker.sh -- --layout tag_layout_box_100mm.json

set -euo pipefail

OAK_IP="${OAK_IP:-192.168.0.153}"
STREAM_PORT="${STREAM_PORT:-8090}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRACKER="${SCRIPT_DIR}/wall_tracker.py"
LAYOUT="${SCRIPT_DIR}/tag_layout_box_100mm.json"

if command -v ping >/dev/null 2>&1; then
  if ! ping -c1 -W2 "${OAK_IP}" &>/dev/null; then
    echo "WARN: OAK ${OAK_IP} not answering ping (tracker may still connect)"
  fi
fi

if [[ "${1:-}" == "--" ]]; then
  shift
fi

echo ""
echo "Starting wall tracker (MJPEG :${STREAM_PORT})"
echo "  OAK IP: ${OAK_IP}"
echo "  Layout: ${LAYOUT}"
echo ""

exec python3 "${TRACKER}" \
  --ip "${OAK_IP}" \
  --layout "${LAYOUT}" \
  --stream-port "${STREAM_PORT}" \
  "$@"
