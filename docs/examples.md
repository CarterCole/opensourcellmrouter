# Configuration examples

Each example is a complete or partial `config.toml` snippet. The router chain
is evaluated **top-to-bottom; first match wins.** Mix and match rules freely —
the only requirement is that some rule eventually resolves every request (a
`fallback` or `random` at the bottom is the usual way to guarantee that).

---

## 1. Local-only: single Ollama instance

```toml
[[providers]]
name     = "ollama"
format   = "ollama"
base_url = "http://localhost:11434"

# Route any model Ollama actually has pulled; discover reads /api/tags at startup.
[[routers]]
type     = "discover"
provider = "ollama"

# Catch-all for model names Ollama doesn't have — rewrite to a known good one.
[[routers]]
type          = "fallback"
rewrite_model = "llama3.1:8b"
```

---

## 2. Local + cloud fallback

Send requests to a local llama-server first; fall back to OpenAI if it's slow or
unavailable.

```toml
[[providers]]
name                      = "local"
format                    = "openai"
base_url                  = "http://localhost:8080/v1"
cost_per_1m_tokens        = 0.0
quality                   = 55
latency_ms                = 900
throughput_tokens_per_sec = 20

[[providers]]
name                      = "openai"
format                    = "openai"
base_url                  = "https://api.openai.com/v1"
api_key_env               = "OPENAI_API_KEY"
cost_per_1m_tokens        = 5.0
quality                   = 90
latency_ms                = 400
throughput_tokens_per_sec = 80

# Requests that explicitly ask for a local model stay local.
[[routers]]
type          = "prefix"
model_prefix  = "local/"
provider      = "local"
rewrite_model = "llama3.2-3b"

# Everything else: prefer local (cheapest), fall back to OpenAI.
[[routers]]
type     = "price"
providers = ["local", "openai"]
```

---

## 3. Tag-based routing (classifiers driving provider choice)

The `keyword` classifier tags requests based on words in the prompt. Hook those
tags to specific providers:

```toml
[classifiers.keyword]
enabled = true
[classifiers.keyword.tags]
vision = ["image", "photo", "screenshot", "diagram"]
code   = ["function", "class", "import", "def ", "fn ", "bug", "refactor"]
nsfw   = ["nsfw", "adult", "explicit"]

[[providers]]
name    = "local"
format  = "openai"
base_url = "http://localhost:8080/v1"

[[providers]]
name     = "ollama"
format   = "ollama"
base_url = "http://localhost:11434"

# Code questions → deepseek-r1 (strong at reasoning)
[[routers]]
type          = "tag"
tag           = "code"
provider      = "ollama"
rewrite_model = "deepseek-r1:latest"

# Anything flagged nsfw → local server only (no cloud, no content policy)
[[routers]]
type          = "tag"
tag           = "nsfw"
provider      = "local"
rewrite_model = "llama3.2-3b"

# Catch-all
[[routers]]
type          = "fallback"
quality_bias  = 0.7
rewrite_model = "llama3.1:8b"
```

---

## 4. Random load-balancing across all local models

Pick a different model on every request — useful for comparing outputs or
spreading load across a pool of models.

```toml
[[providers]]
name     = "local"
format   = "openai"
base_url = "http://localhost:8080/v1"

[[providers]]
name     = "ollama"
format   = "ollama"
base_url = "http://localhost:11434"

[[routers]]
type = "random"
candidates = [
  { provider = "local",  model = "llama3.2-3b"        },
  { provider = "ollama", model = "llama3.1:8b"        },
  { provider = "ollama", model = "deepseek-r1:latest" },
  { provider = "ollama", model = "gemma3:latest"      },
]
```

---

## 5. Anthropic Claude via cloud

```toml
[[providers]]
name               = "anthropic"
format             = "anthropic"
base_url           = "https://api.anthropic.com"
api_key_env        = "ANTHROPIC_API_KEY"
cost_per_1m_tokens = 15.0
quality            = 95

# Any model name starting with "claude" goes straight to Anthropic.
[[routers]]
type         = "prefix"
model_prefix = "claude"
provider     = "anthropic"

# Or use a fallback so every unknown model hits Anthropic:
[[routers]]
type          = "fallback"
providers     = ["anthropic"]
rewrite_model = "claude-sonnet-4-6"
```

---

## 6. Latency-first with quality fallback

Route to whichever provider responds fastest; if none declares latency, fall
back to highest quality.

```toml
[[providers]]
name       = "fast-local"
format     = "openai"
base_url   = "http://localhost:8080/v1"
latency_ms = 300
quality    = 50

[[providers]]
name       = "ollama"
format     = "ollama"
base_url   = "http://localhost:11434"
latency_ms = 600
quality    = 75

[[providers]]
name               = "openai"
format             = "openai"
base_url           = "https://api.openai.com/v1"
api_key_env        = "OPENAI_API_KEY"
latency_ms         = 400
quality            = 90

[[routers]]
type           = "latency"
max_latency_ms = 500          # only consider providers under 500 ms

[[routers]]
type         = "fallback"     # catches anything the latency rule skipped
quality_bias = 1.0            # pure quality, ignore cost
```

---

## 7. Combining rules: prefix → tag → discover → random

```toml
[[providers]]
name     = "local"
format   = "openai"
base_url = "http://localhost:8080/v1"

[[providers]]
name     = "ollama"
format   = "ollama"
base_url = "http://localhost:11434"

# Explicit "local/…" prefix pins to llama-server.
[[routers]]
type          = "prefix"
model_prefix  = "local/"
provider      = "local"
rewrite_model = "llama3.2-3b"

# Code tag → deepseek on Ollama.
[[routers]]
type          = "tag"
tag           = "code"
provider      = "ollama"
rewrite_model = "deepseek-r1:latest"

# Exact model name match → Ollama (e.g. client asks for "gemma3:latest").
[[routers]]
type     = "discover"
provider = "ollama"

# Everything else → random from the local pool.
[[routers]]
type = "random"
candidates = [
  { provider = "local",  model = "llama3.2-3b"  },
  { provider = "ollama", model = "llama3.1:8b"  },
  { provider = "ollama", model = "gemma3:latest" },
]
```
