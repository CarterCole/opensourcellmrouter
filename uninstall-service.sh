#!/usr/bin/env bash
# uninstall-service.sh — removes the systemd service installed by
# install-service.sh. Leaves the repo checkout, binary, .env, and logs alone.
#
# Usage: ./uninstall-service.sh
set -euo pipefail

SERVICE_NAME="opensourcellmrouter"
UNIT_PATH="/etc/systemd/system/${SERVICE_NAME}.service"

if systemctl list-unit-files "${SERVICE_NAME}.service" --no-legend 2>/dev/null | grep -q "${SERVICE_NAME}.service"; then
    sudo systemctl disable --now "$SERVICE_NAME"
else
    echo "$SERVICE_NAME service not registered, nothing to disable."
fi

sudo rm -f "$UNIT_PATH"
sudo systemctl daemon-reload

echo "Removed $UNIT_PATH. Repo checkout, binary, .env, and logs were left untouched."
