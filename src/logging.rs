//! Appends one line of JSON per request/response pair to a log file, for
//! debugging and auditing what the router sent upstream and got back.
//!
//! The SSE feed (`/dashboard/events`) broadcasts [`RouterEvent`] values —
//! both mid-flight `Start` events (emitted as soon as a request enters the
//! pipeline) and final `Complete` events (emitted when a response is ready).

use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::canonical::{ChatResponse, Message};

/// Wire format for the SSE feed. Both the TUI and the browser dashboard
/// parse this; the `type` field discriminates the variant.
///
/// Note: `u128` is not supported by `serde_json`, so all timestamps and
/// durations use `u64` (ms-since-epoch fits comfortably in u64 until the
/// year 292 million).
#[derive(Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RouterEvent {
    /// Emitted immediately when a request enters `dispatch`, before
    /// classifiers or routing run. Lets the UI show in-flight requests.
    Start {
        id: u64,
        ts_ms: u64,
        /// Model name as the client sent it.
        model: String,
        /// Number of requests currently in flight (including this one).
        in_flight: u64,
    },
    /// Emitted after classifiers run, before routing. `tags` may be empty.
    Classified {
        id: u64,
        ts_ms: u64,
        tags: Vec<String>,
    },
    /// Emitted after the router resolves a provider and model, before the
    /// upstream call goes out.
    Routed {
        id: u64,
        ts_ms: u64,
        provider: String,
        model: String,
    },
    /// Emitted when the pipeline finishes (response or error). `id` matches
    /// the corresponding `Start` event.
    Complete {
        id: u64,
        #[serde(flatten)]
        entry: LogEntry,
    },
}

#[derive(Serialize, Deserialize)]
pub struct LogEntry {
    pub ts_ms: u64,
    pub provider: String,
    /// Model name as requested by the client.
    pub requested_model: String,
    /// Model name actually sent to the provider (after `rewrite_model`).
    pub sent_model: String,
    pub duration_ms: u64,
    /// Tags assigned by `classifiers` before routing (see
    /// [`crate::classifiers`]), e.g. `["vision"]`.
    pub tags: Vec<String>,
    /// OTel trace id for this request (hex string), if telemetry is enabled.
    /// Lets a JSONL log line be pasted straight into a tracing backend's
    /// search to find the matching trace.
    #[serde(default)]
    pub trace_id: Option<String>,
    /// Ids of plugins that actively mutated req/resp (see [`crate::plugins`]).
    pub plugins: Vec<String>,
    pub system: Option<String>,
    pub messages: Vec<Message>,
    pub response: Option<ChatResponse>,
    pub error: Option<String>,
}

impl LogEntry {
    pub fn now_ms() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0)
    }
}

pub struct RequestLogger {
    file: Mutex<std::fs::File>,
}

impl RequestLogger {
    pub fn new(path: &str) -> anyhow::Result<Self> {
        let path = Path::new(path);
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        Ok(RequestLogger {
            file: Mutex::new(file),
        })
    }

    /// Appends a pre-serialized JSON line to the log file.
    pub fn log_line(&self, line: &str) {
        let mut file = self.file.lock().unwrap_or_else(|e| e.into_inner());
        if let Err(err) = writeln!(file, "{line}") {
            tracing::warn!("failed to write log entry: {err}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn router_event_complete_roundtrip() {
        let json = r#"{"type":"complete","id":0,"ts_ms":1781573245225,"provider":"ollama","requested_model":"llama3.1:8b","sent_model":"llama3.1:8b","duration_ms":5978,"tags":[],"plugins":[],"system":null,"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}],"response":null,"error":null}"#;
        let event: RouterEvent = serde_json::from_str(json).expect("deser failed");
        match event {
            RouterEvent::Complete { id, entry } => {
                assert_eq!(id, 0);
                assert_eq!(entry.provider, "ollama");
            }
            _ => panic!("wrong variant"),
        }
    }
    #[test]
    fn router_event_start_roundtrip() {
        let json = r#"{"type":"start","id":1,"ts_ms":1781573328452,"model":"llama3.1:8b","in_flight":2}"#;
        let event: RouterEvent = serde_json::from_str(json).unwrap();
        assert!(matches!(event, RouterEvent::Start { id: 1, .. }));
    }
}
