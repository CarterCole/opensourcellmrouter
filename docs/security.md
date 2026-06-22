# Security

By default the router binds `127.0.0.1` and has no inbound auth — fine for
local-only use. Two independent knobs under `[server]` in `config.toml`
control what's exposed and who can reach it:

```toml
[server]
host        = "127.0.0.1"        # 0.0.0.0 to accept connections from other machines
port        = 8090
api_key_env = "ROUTER_API_KEY"   # omit to leave the server unauthenticated
```

## `host`

`127.0.0.1` (the default) means only this machine can reach the router.
Set `host = "0.0.0.0"` if you need another machine to reach it — e.g.
pointing [`launch-coding-agent.sh`](../launch-coding-agent.sh) at this
router from a different box. Whenever you do, set `api_key_env` too —
there's nothing else between the network and your provider credentials.

## `api_key_env`

Names an environment variable (resolved once at startup, not stored in
`config.toml`) holding a shared secret clients must present. When unset,
every route except `/health` is open to anyone who can reach `host:port`.
When set, every route except `/health` requires the key via one of:

- `Authorization: Bearer <key>`
- `x-api-key: <key>`
- `?api_key=<key>` query parameter (for `/dashboard`, since a browser's
  `EventSource` can't set custom headers — open
  `http://host:port/dashboard?api_key=<key>` directly)

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

If you're pointing a coding-agent CLI at the router (see
[`docs/coding-agents.md`](coding-agents.md)), set the tool's own API-key
env var to the same value instead of a dummy placeholder — e.g.
`ANTHROPIC_AUTH_TOKEN=$ROUTER_API_KEY` for Claude Code.
