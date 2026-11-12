# Security

By default the router binds `127.0.0.1` and has no inbound auth — fine for
local-only use. A few knobs under `[server]` in `config.toml` control what's
exposed and who can reach it:

```toml
[server]
host        = "127.0.0.1"        # 0.0.0.0 to accept connections from other machines
port        = 8090
api_key_env = "ROUTER_API_KEY"   # omit to leave the server unauthenticated
# allowed_hosts         = ["router.lan"]  # host names an unauthenticated router answers to
# allow_unauthenticated = true            # see below; rarely what you want
```

## `host`

`127.0.0.1` (the default) means only this machine can reach the router.
Set `host = "0.0.0.0"` if you need another machine to reach it — e.g.
pointing [`launch-coding-agent.sh`](../launch-coding-agent.sh) at this
router from a different box. Then you need `api_key_env` too: the router
refuses to start on an address other machines can reach without one, since
anyone who reached it could spend your providers' credits and read every
request on the dashboard.

If something in front of the router already controls who reaches it — a
reverse proxy that authenticates, or a container whose port is published on
the host's `127.0.0.1` only — set `allow_unauthenticated = true` to start
anyway.

The router speaks plain HTTP. Across a network you don't trust, put it
behind something that terminates TLS (a reverse proxy, or an SSH tunnel),
or the key and every prompt cross the wire in the clear.

## `api_key_env`

Names an environment variable (resolved once at startup, not stored in
`config.toml`) holding a shared secret clients must present. When unset,
every route except `/health` is open to anyone who can reach `host:port`.
When set, every route except `/health` requires the key via one of:

- `Authorization: Bearer <key>`
- `x-api-key: <key>`
- `?api_key=<key>` query parameter (for `/dashboard`, since a browser's
  `EventSource` can't set custom headers — open
  `http://host:port/dashboard?api_key=<key>` directly). The router keeps
  the query string out of its request logs, but the URL still lands in
  your browser history.

A missing or wrong key gets `401 Unauthorized`. The router fails to start
if `api_key_env` is set but the named variable isn't, rather than silently
running unauthenticated.

```bash
# .env (loaded by demo.sh, or via EnvironmentFile= in a systemd unit)
ROUTER_API_KEY=<a long random value, e.g. `openssl rand -hex 32`>
```

```bash
curl http://localhost:8090/v1/chat/completions \
  -H 'x-api-key: <ROUTER_API_KEY value>' \
  -H 'content-type: application/json' \
  -d '{"model":"phi3:mini","messages":[{"role":"user","content":"hi"}]}'
```

`opensourcellmrouter watch`, `opensourcellmrouter tui` and
`launch-coding-agent.sh` send `$ROUTER_API_KEY` when it's set. If you're
pointing a coding-agent CLI at the router by hand (see
[`docs/coding-agents.md`](coding-agents.md)), set the tool's own API-key
env var to the same value instead of a dummy placeholder — e.g.
`ANTHROPIC_AUTH_TOKEN=$ROUTER_API_KEY` for Claude Code.

## `allowed_hosts`

Without `api_key_env`, the router answers only requests addressed to
`localhost`, a `*.localhost` name, or an IP address; any other `Host` gets
`403`. That stops a web page you visit from reaching an unauthenticated
router through DNS rebinding (pointing its own domain at `127.0.0.1`).
If you reach an unauthenticated router by another name — one in
`/etc/hosts`, say — list it in `allowed_hosts`. With `api_key_env` set the
key does this job, and any host name works.

## What the router keeps

The request log (`[logging]`) and the dashboard feed hold every system
prompt, message and response. A new log file is created readable by its
owner only (`0600`); one created by an older version keeps its mode, so
check it (`chmod 600 logs/requests.jsonl`). The `cev` classifier stores
what it's shown of each request (the system prompt, the latest user
message and its images) in its own database.
