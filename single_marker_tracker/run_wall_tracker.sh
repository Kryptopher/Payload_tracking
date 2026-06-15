#!/usr/bin/env bash
# Standalone USB OAK-D S2 AprilTag wall/cube tracker.
#
#   ./run_wall_tracker.sh
#   ./run_wall_tracker.sh -- --layout tag_layout_8x8x6in_36h11_12front_13left_14back_15right.json

set -euo pipefail

OAK_IP="${OAK_IP:-}"
STREAM_PORT="${STREAM_PORT:-8090}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRACKER="${SCRIPT_DIR}/wall_tracker.py"
LAYOUT="${SCRIPT_DIR}/tag_layout_8x8x6in_36h11_12front_13left_14back_15right.json"

if [[ -n "${OAK_IP}" ]] && command -v ping >/dev/null 2>&1; then
  if ! ping -c1 -W2 "${OAK_IP}" &>/dev/null; then
    echo "WARN: OAK ${OAK_IP} not answering ping (tracker may still connect)"
  fi
fi

if [[ "${1:-}" == "--" ]]; then
  shift
fi

echo ""
echo "Starting wall tracker (MJPEG :${STREAM_PORT})"
echo "  OAK device: ${OAK_IP:-USB auto-discover}"
echo "  Layout: ${LAYOUT}"
echo ""

IP_ARGS=()
if [[ -n "${OAK_IP}" ]]; then
  IP_ARGS=(--ip "${OAK_IP}")
fi

exec python3 "${TRACKER}" \
  "${IP_ARGS[@]}" \
  --layout "${LAYOUT}" \
  --stream-port "${STREAM_PORT}" \
  "$@"
