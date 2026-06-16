#!/usr/bin/env bash
# demo.sh — builds and starts opensourcellmrouter pointed at the local
# llama.cpp server (:8080) and Ollama (:11434), then opens the TUI.
#
# llama-server is started automatically if :8080 is not answering.
# Ollama must already be running (it usually is as a system service).
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

# ── 4. write config ───────────────────────────────────────────────────────────

step "Writing config"
CONFIG=$(mktemp /tmp/ollmr-demo-XXXXXX.toml)

cat > "$CONFIG" << TOML
[server]
host      = "127.0.0.1"
port      = $ROUTER_PORT
dashboard = true

[logging]
enabled = true
path    = "/tmp/ollmr-demo-requests.jsonl"

# ── providers ─────────────────────────────────────────────────────────────────

[[providers]]
name                      = "local-llama"
format                    = "openai"
base_url                  = "http://127.0.0.1:$LLAMA_PORT/v1"
cost_per_1m_tokens        = 0.0
quality                   = 55
latency_ms                = 900
throughput_tokens_per_sec = 20

# Ollama native API — base_url has no /v1 suffix.
# The "discover" router rule below auto-populates which models are available.
[[providers]]
name                      = "ollama"
format                    = "ollama"
base_url                  = "http://127.0.0.1:$OLLAMA_PORT"
cost_per_1m_tokens        = 0.0
quality                   = 75
latency_ms                = 600
throughput_tokens_per_sec = 30

# ── classifiers ───────────────────────────────────────────────────────────────

[classifiers.keyword]
enabled = true
[classifiers.keyword.tags]
vision = ["image", "photo", "picture", "screenshot", "visual"]
code   = ["function", "class", "import", "def ", "fn "]
nsfw   = []

# ── routers (first match wins) ────────────────────────────────────────────────

# "local/..." always goes to llama.cpp; model name is cosmetic.
[[routers]]
type          = "prefix"
model_prefix  = "local/"
provider      = "local-llama"
rewrite_model = "llama3.2-3b"

# Any model Ollama reports having (discovered at startup via GET /api/tags)
# is routed straight there — e.g. "llama3.1:8b", "deepseek-r1:latest".
[[routers]]
type     = "discover"
provider = "ollama"

# vision/code classifier tags → route to the larger Ollama model
[[routers]]
type          = "tag"
tag           = "vision"
provider      = "ollama"
rewrite_model = "llama3.1:8b"

[[routers]]
type          = "tag"
tag           = "code"
provider      = "ollama"
rewrite_model = "llama3.1:8b"

# Catch-all: score = 0.7*quality - 0.3*cost. With equal cost=0 this
# picks the highest-quality provider (ollama, quality=75 > local 55).
[[routers]]
type         = "fallback"
quality_bias = 0.7

# ── plugins ───────────────────────────────────────────────────────────────────

[plugins.response-healing]
enabled = true
TOML

ok "config: $CONFIG"

# ── 5. start router ───────────────────────────────────────────────────────────

step "Starting opensourcellmrouter on :$ROUTER_PORT"
./target/debug/opensourcellmrouter "$CONFIG" &
ROUTER_PID=$!
wait_for "router" "http://127.0.0.1:$ROUTER_PORT/health"

# ── 6. print cheat-sheet and open TUI ────────────────────────────────────────

echo
echo "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo "${BOLD} opensourcellmrouter — local pipeline demo${RESET}"
echo
echo "  Dashboard   ${CYAN}http://127.0.0.1:$ROUTER_PORT/dashboard${RESET}"
echo "  Request log ${DIM}/tmp/ollmr-demo-requests.jsonl${RESET}"
echo
echo "  Default model in TUI chat pane is ${BOLD}gpt-4${RESET} (hits fallback → ollama)."
echo "  Change it mid-session with  ${CYAN}:model llama3.1:8b${RESET}"
echo
echo "  Chat examples:"
echo "    ${DIM}hello world${RESET}                  fallback → ollama (best quality)"
echo "    ${DIM}write a Python function${RESET}      code tag → llama3.1:8b"
echo "    ${DIM}local/quick: ping${RESET}            prefix rule → llama.cpp"
echo "    ${DIM}(model: deepseek-r1:latest)${RESET}  discover rule → ollama as-is"
echo "    ${DIM}(model: llama3.1:8b)${RESET}         discover rule → ollama as-is"
echo
echo "  Keys: ${BOLD}Tab / i${RESET} focus chat  ${BOLD}↑↓${RESET} scroll feed  ${BOLD}q / Ctrl-C${RESET} quit"
echo "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo

./target/debug/opensourcellmrouter tui "http://127.0.0.1:$ROUTER_PORT"
