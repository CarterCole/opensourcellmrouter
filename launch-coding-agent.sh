#!/usr/bin/env bash
# launch-coding-agent.sh — points a coding-agent CLI at a running
# opensourcellmrouter instance instead of its default backend.
#
# See docs/coding-agents.md for what each tool needs and why.
#
# Usage: ./launch-coding-agent.sh <claude|copilot|codex> [host:port] [-- extra args]
#
# host:port defaults to localhost:8090 (this repo's default router port),
# but can point at a router running on another machine, e.g.:
#   ./launch-coding-agent.sh claude 192.168.1.50:8090
set -euo pipefail

usage() {
    echo "usage: $0 <claude|copilot|codex> [host:port] [-- extra args]" >&2
    exit 1
}

TOOL="${1:-}"
[ -n "$TOOL" ] || usage
shift

ADDR="localhost:8090"
if [ "${1:-}" != "" ] && [ "${1:-}" != "--" ]; then
    ADDR="$1"
    shift
fi
[ "${1:-}" = "--" ] && shift

case "$TOOL" in
claude)
    export ANTHROPIC_BASE_URL="http://${ADDR}"
    export ANTHROPIC_AUTH_TOKEN="dummy"
    exec claude "$@"
    ;;
copilot)
    export COPILOT_PROVIDER_BASE_URL="http://${ADDR}/v1"
    export COPILOT_PROVIDER_TYPE="openai"
    export COPILOT_PROVIDER_API_KEY="dummy"
    : "${COPILOT_MODEL:?set COPILOT_MODEL to a model name your router config can resolve}"
    exec copilot "$@"
    ;;
codex)
    export LOCAL_ROUTER_DUMMY_KEY="dummy"
    exec codex \
        -c model_provider='"local-router"' \
        -c 'model_providers.local-router.name="local-router"' \
        -c "model_providers.local-router.base_url=\"http://${ADDR}/v1\"" \
        -c 'model_providers.local-router.wire_api="responses"' \
        -c 'model_providers.local-router.env_key="LOCAL_ROUTER_DUMMY_KEY"' \
        "$@"
    ;;
*)
    usage
    ;;
esac
