#!/usr/bin/env bash
# demo.sh — builds and starts opensourcellmrouter using config.toml, then opens the TUI.
#
# llama-server is started automatically if :8080 is not answering.
# Ollama must already be running (it usually is as a system service).
# Re-running the script restarts the router with the current config.toml.
#
# Usage: ./demo.sh [router-port]
set -euo pipefail

ROUTER_PORT=${1:-8090}
LLAMA_PORT=8080
OLLAMA_PORT=11434
LLAMA_BIN=/home/carter/Code/llama.cpp/build/bin/llama-server
DEFAULT_MODEL=/home/carter/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json
LLAMA_LOG=/tmp/llama-server-demo.log
CONFIG=config.toml

BOLD=$'\e[1m'
DIM=$'\e[2m'
CYAN=$'\e[36m'
YELLOW=$'\e[33m'
RED=$'\e[31m'
RESET=$'\e[0m'

LLAMA_STARTED=false
ROUTER_PID=

step()  { echo; echo "${BOLD}${CYAN}▶  $*${RESET}"; }
ok()    { echo "  ${CYAN}✓${RESET}  $*"; }
warn()  { echo "  ${YELLOW}!${RESET}  $*"; }
die()   { echo "  ${RED}✗${RESET}  $*"; exit 1; }

cleanup() {
    [[ -n "$ROUTER_PID" ]] && kill "$ROUTER_PID" 2>/dev/null || true
    $LLAMA_STARTED && kill "$LLAMA_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

wait_for() {
    local name=$1 url=$2
    echo -n "  waiting for $name"
    for _ in $(seq 1 40); do
        if curl -sf "$url" > /dev/null 2>&1; then echo " ready"; return 0; fi
        echo -n "."; sleep 0.3
    done
    echo; warn "$name did not become ready"
    return 1
}

# ── 1. build ──────────────────────────────────────────────────────────────────

step "Building opensourcellmrouter…"
cargo build --quiet
ok "binary: ./target/debug/opensourcellmrouter"

# ── 2. llama.cpp server ───────────────────────────────────────────────────────

step "Checking llama.cpp server on :$LLAMA_PORT"
if curl -sf "http://localhost:$LLAMA_PORT/health" > /dev/null 2>&1; then
    MODEL_ID=$(curl -sf "http://localhost:$LLAMA_PORT/v1/models" \
        | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'])" 2>/dev/null \
        || echo "unknown")
    ok "already running  model=$MODEL_ID"
else
    warn "not running — starting with $DEFAULT_MODEL"
    [[ -f "$DEFAULT_MODEL" ]] || die "model file not found: $DEFAULT_MODEL"
    [[ -x "$LLAMA_BIN"    ]] || die "binary not found: $LLAMA_BIN"

    VK_ICD_FILENAMES=$VK_ICD_FILENAMES \
    "$LLAMA_BIN" \
        --model "$DEFAULT_MODEL" \
        --port "$LLAMA_PORT" \
        --host 127.0.0.1 \
        --n-gpu-layers 20 \
        --ctx-size 4096 \
        --log-disable \
        > "$LLAMA_LOG" 2>&1 &
    LLAMA_PID=$!
    LLAMA_STARTED=true
    wait_for "llama-server" "http://localhost:$LLAMA_PORT/health"
fi

# ── 3. ollama ─────────────────────────────────────────────────────────────────

step "Checking Ollama on :$OLLAMA_PORT"
if curl -sf "http://localhost:$OLLAMA_PORT/api/tags" > /dev/null 2>&1; then
    MODELS=$(curl -sf "http://localhost:$OLLAMA_PORT/api/tags" \
        | python3 -c "import sys,json; d=json.load(sys.stdin); print(', '.join(m['name'] for m in d.get('models',[])))" 2>/dev/null \
        || echo "?")
    ok "running  models: $MODELS"
else
    warn "Ollama not reachable on :$OLLAMA_PORT — ollama routing will pass through"
fi

# ── 4. start router ───────────────────────────────────────────────────────────

step "Starting opensourcellmrouter on :$ROUTER_PORT (config: $CONFIG)"
[[ -f "$CONFIG" ]] || die "config file not found: $CONFIG"
if [[ -f .env ]]; then
    # shellcheck disable=SC1091
    set -a; source .env; set +a
    ok "loaded .env"
fi
if fuser -k "${ROUTER_PORT}/tcp" 2>/dev/null; then
    ok "stopped previous router on :$ROUTER_PORT"
    sleep 0.3
fi
./target/debug/opensourcellmrouter "$CONFIG" &
ROUTER_PID=$!
wait_for "router" "http://127.0.0.1:$ROUTER_PORT/health"

# ── 5. print cheat-sheet and open TUI ────────────────────────────────────────

echo
echo "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo "${BOLD} opensourcellmrouter — local pipeline demo${RESET}"
echo
echo "  Config      ${DIM}$CONFIG${RESET}"
echo "  Dashboard   ${CYAN}http://127.0.0.1:$ROUTER_PORT/dashboard${RESET}"
echo "  Request log ${DIM}logs/requests.jsonl${RESET}"
echo
echo "  Active router: ${BOLD}random${RESET} — picks a model at random each request:"
echo "    ${DIM}local-llama  llama3.2-3b${RESET}"
echo "    ${DIM}ollama       llama3.1:8b  deepseek-r1:latest  gemma3:latest${RESET}"
echo "    ${DIM}cloudflare   llama-3.1-8b  llama-3.2-3b  deepseek-r1-distill  gemma-3-12b${RESET}"
echo "    ${DIM}openai       gpt-4o-mini  gpt-4o${RESET}  ${YELLOW}(needs OPENAI_API_KEY in .env)${RESET}"
echo "    ${DIM}anthropic    haiku-4.5  sonnet-4.6${RESET}  ${YELLOW}(needs ANTHROPIC_API_KEY in .env)${RESET}"
echo
echo "  Edit ${BOLD}config.toml${RESET} and re-run ${BOLD}./demo.sh${RESET} to apply changes."
echo "  See ${CYAN}docs/examples.md${RESET} for provider + router recipes."
echo
echo "  Keys: ${BOLD}Tab / i${RESET} focus chat  ${BOLD}↑↓${RESET} scroll feed  ${BOLD}q / Ctrl-C${RESET} quit"
echo "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo

./target/debug/opensourcellmrouter tui "http://127.0.0.1:$ROUTER_PORT"
