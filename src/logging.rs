//! Appends one line of JSON per request/response pair to a log file, for
//! debugging and auditing what the router sent upstream and got back.

use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::canonical::{ChatResponse, Message};

#[derive(Serialize, Deserialize)]
pub struct LogEntry {
    pub ts_ms: u128,
    pub provider: String,
    /// Model name as requested by the client.
    pub requested_model: String,
    /// Model name actually sent to the provider (after `rewrite_model`).
    pub sent_model: String,
    pub duration_ms: u128,
    /// Tags assigned by `classifiers` before routing (see
    /// [`crate::classifiers`]), e.g. `["vision"]`.
    pub tags: Vec<String>,
    /// Ids of plugins that ran for this request (see [`crate::plugins`]).
    pub plugins: Vec<String>,
    pub system: Option<String>,
    pub messages: Vec<Message>,
    pub response: Option<ChatResponse>,
    pub error: Option<String>,
}

impl LogEntry {
    pub fn now_ms() -> u128 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis())
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

    /// Appends a pre-serialized JSON line (see [`LogEntry`]) to the log file.
    pub fn log_line(&self, line: &str) {
        let mut file = self.file.lock().unwrap_or_else(|e| e.into_inner());
        if let Err(err) = writeln!(file, "{line}") {
            tracing::warn!("failed to write log entry: {err}");
        }
    }
}
