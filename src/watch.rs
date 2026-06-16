//! `opensourcellmrouter watch [BASE_URL]`
//!
//! Connects to the server's `/dashboard/events` SSE feed and pretty-prints
//! each request as it's handled. Shares the same wire format as the browser
//! dashboard — both consume the same JSON [`crate::logging::LogEntry`] lines.

use anyhow::Context;
use tokio_stream::StreamExt;

use crate::canonical::Role;
use crate::logging::LogEntry;

// ANSI escape codes — no extra dependency needed.
const RESET: &str = "\x1b[0m";
const BOLD: &str = "\x1b[1m";
const DIM: &str = "\x1b[2m";
const CYAN: &str = "\x1b[36m";
const BLUE: &str = "\x1b[94m";
const YELLOW: &str = "\x1b[33m";
const MAGENTA: &str = "\x1b[35m";
const RED: &str = "\x1b[31m";

pub async fn run(base_url: &str) -> anyhow::Result<()> {
    let url = format!("{}/dashboard/events", base_url.trim_end_matches('/'));
    eprintln!("connecting to {url} …");

    let client = reqwest::Client::new();
    let resp = client
        .get(&url)
        .header("Accept", "text/event-stream")
        .send()
        .await
        .with_context(|| format!("connecting to {url}"))?;

    if !resp.status().is_success() {
        anyhow::bail!("server returned {}", resp.status());
    }

    eprintln!("connected — waiting for requests\n");

    let mut stream = resp.bytes_stream();
    let mut buf = String::new();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("reading SSE stream")?;
        buf.push_str(&String::from_utf8_lossy(&chunk));

        while let Some(nl) = buf.find('\n') {
            let line = buf[..nl].trim_end_matches('\r').to_owned();
            buf.drain(..=nl);

            if let Some(data) = line.strip_prefix("data: ") {
                match serde_json::from_str::<LogEntry>(data) {
                    Ok(entry) => print_entry(&entry),
                    Err(err) => eprintln!("{DIM}[parse error: {err}]{RESET}"),
                }
            }
        }
    }

    Ok(())
}

fn print_entry(e: &LogEntry) {
    // UTC time from the UNIX ms timestamp.
    let secs = (e.ts_ms / 1000) as u64;
    let time = format!("{:02}:{:02}:{:02}", (secs / 3600) % 24, (secs / 60) % 60, secs % 60);

    // ── header line ──────────────────────────────────────────────────────────
    let mut header = format!("{DIM}{time}{RESET}  {BOLD}{CYAN}{}{RESET}", e.provider);

    if e.requested_model != e.sent_model {
        header.push_str(&format!(
            "  {DIM}{}{RESET} {DIM}→{RESET} {BLUE}{}{RESET}",
            e.requested_model, e.sent_model
        ));
    } else {
        header.push_str(&format!("  {BLUE}{}{RESET}", e.sent_model));
    }

    for tag in &e.tags {
        header.push_str(&format!("  {YELLOW}[{tag}]{RESET}"));
    }
    for plugin in &e.plugins {
        header.push_str(&format!("  {MAGENTA}({plugin}){RESET}"));
    }

    let dur_color = if e.error.is_some() { RED } else { DIM };
    header.push_str(&format!("  {dur_color}{}ms{RESET}", e.duration_ms));

    println!("{header}");

    // ── prompt ───────────────────────────────────────────────────────────────
    let last_user = e
        .messages
        .iter()
        .rev()
        .find(|m| m.role == Role::User)
        .map(|m| m.content.as_str())
        .unwrap_or("");
    println!("  {DIM}prompt:{RESET}  {}", truncate(last_user, 120));

    // ── response or error ────────────────────────────────────────────────────
    if let Some(err) = &e.error {
        println!("  {RED}error:{RESET}   {}", truncate(err, 120));
    } else if let Some(resp) = &e.response {
        println!("  {DIM}reply:{RESET}   {}", truncate(&resp.content, 120));
        println!(
            "  {DIM}{} in / {} out · {:?}{RESET}",
            resp.usage.input_tokens, resp.usage.output_tokens, resp.stop_reason
        );
    }

    println!();
}

/// Truncates `s` to at most `max` Unicode scalar values, appending `…` if cut.
fn truncate(s: &str, max: usize) -> String {
    // Collapse newlines so multi-turn prompts stay on one line.
    let flat: String = s.chars().map(|c| if c == '\n' { ' ' } else { c }).collect();
    let mut chars = flat.chars();
    let head: String = chars.by_ref().take(max).collect();
    if chars.next().is_some() { head + "…" } else { head }
}
