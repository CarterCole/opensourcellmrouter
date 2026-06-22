#!/usr/bin/env bash
# install-service.sh — installs opensourcellmrouter as a systemd service that
# runs in-place from this repo checkout (no copying binaries/config around).
#
# Logs:
#   journalctl -u opensourcellmrouter -f   — tracing/stdout logs
#   <repo>/<logs.path from config>         — JSONL request log, if [logging] enabled
#
# Usage: ./install-service.sh [config-file]
set -euo pipefail

CONFIG="${1:-config.toml}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_NAME="opensourcellmrouter"
UNIT_PATH="/etc/systemd/system/${SERVICE_NAME}.service"

[ -f "$REPO_DIR/$CONFIG" ] || { echo "error: config file '$REPO_DIR/$CONFIG' not found" >&2; exit 1; }

echo "Building release binary..."
cargo build --release --manifest-path "$REPO_DIR/Cargo.toml"

echo "Installing systemd unit at $UNIT_PATH..."
sudo tee "$UNIT_PATH" >/dev/null <<EOF
[Unit]
Description=opensourcellmrouter
After=network.target

[Service]
ExecStart=$REPO_DIR/target/release/opensourcellmrouter $REPO_DIR/$CONFIG
WorkingDirectory=$REPO_DIR
EnvironmentFile=-$REPO_DIR/.env
Environment=RUST_LOG=info
Restart=on-failure
User=$(whoami)

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now "$SERVICE_NAME"

LOG_PATH=$(awk -F'"' '/^path/{print $2}' "$REPO_DIR/$CONFIG" 2>/dev/null || true)

echo
echo "Installed and started. Useful commands:"
echo "  sudo systemctl status $SERVICE_NAME"
echo "  sudo systemctl restart $SERVICE_NAME"
echo "  journalctl -u $SERVICE_NAME -f"
if [ -n "$LOG_PATH" ]; then
    echo "  tail -f $REPO_DIR/$LOG_PATH   # JSONL request log"
fi
