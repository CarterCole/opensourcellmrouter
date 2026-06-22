# Coding agents

Point a real coding-agent CLI at the router instead of straight at a
provider, so its requests flow through the router's classifiers, plugins,
and dashboard like everything else. Each tool speaks a different client
wire format, served by the router's three endpoints — `/v1/chat/completions`
(OpenAI), `/v1/messages` (Anthropic), `/v1/responses` (OpenAI Responses API).

First, run the router itself:

```bash
cargo run -- config.ollama.toml
```

This binds `http://localhost:8090` with the dashboard enabled at
`http://localhost:8090/dashboard` — the first place to check if any of the
setups below misbehaves, since it shows every request/response/tag live.

By default the router has no inbound auth check, so any non-empty
placeholder value works for the "API key" env vars each tool requires
below. If you've set `[server] api_key_env` (see
[`docs/security.md`](security.md) — required if `host = "0.0.0.0"`, e.g.
to reach the router from another machine via `launch-coding-agent.sh`),
use that key's actual value instead of `dummy` in every snippet below.

## Quick start: `launch-coding-agent.sh`

`./launch-coding-agent.sh <claude|copilot|codex> [host:port] [-- extra args]`
sets the right env vars/config overrides and execs the tool, so you don't
need to copy the per-tool snippets below by hand. `host:port` defaults to
`localhost:8090`; pass an explicit one to point at a router running on
another machine:

```bash
./launch-coding-agent.sh claude                      # -> localhost:8090
./launch-coding-agent.sh claude 192.168.1.50:8090    # -> a remote router
COPILOT_MODEL=llama3.1:8b ./launch-coding-agent.sh copilot
./launch-coding-agent.sh codex -- --some-codex-flag
```

`COPILOT_MODEL` must already be set in the environment (the script doesn't
pick a default) since it has to match a model your router config resolves.
The sections below explain what the script sets and why, for anyone setting
a tool up by hand instead.

## Claude Code

Claude Code speaks the Anthropic Messages API and always streams:

```bash
export ANTHROPIC_BASE_URL=http://localhost:8090
export ANTHROPIC_AUTH_TOKEN=dummy
claude
```

## GitHub Copilot CLI

Copilot CLI speaks plain OpenAI Chat Completions, and requires both
streaming and tool calling to be supported by whatever it's pointed at:

```bash
export COPILOT_PROVIDER_BASE_URL=http://localhost:8090/v1
export COPILOT_PROVIDER_TYPE=openai
export COPILOT_PROVIDER_API_KEY=dummy
export COPILOT_MODEL=<model name matching a router rule, e.g. llama3.1:8b>
copilot
```

Test a tool-calling prompt first (e.g. "list the files in this directory")
— streaming-plus-tool-calls is the newest code path here.

## Codex CLI / Codex App

OpenAI has removed Chat Completions support from both, so they only speak
the newer Responses API. Configure a custom `model_provider` in
`~/.codex/config.toml`:

```toml
model_provider = "local-router"

[model_providers.local-router]
name = "local-router"
base_url = "http://localhost:8090/v1"
wire_api = "responses"
env_key = "LOCAL_ROUTER_DUMMY_KEY"
```

```bash
export LOCAL_ROUTER_DUMMY_KEY=dummy
codex
```

`env_key` must point at *some* set environment variable — Codex won't start
without one — even though the router ignores its value.

## Troubleshooting

Check `http://localhost:8090/dashboard` first for any of the three: it
streams every request as it's classified, routed, and answered, including
which provider and model actually handled it.
