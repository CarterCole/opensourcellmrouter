//! `opensourcellmrouter watch [BASE_URL]`
//!
//! Connects to the server's `/dashboard/events` SSE feed and pretty-prints
//! each request as it's handled. Shares the same [`RouterEvent`] wire format
//! as the browser dashboard and the TUI.
//!
//! `Start` events print a one-liner showing the model and in-flight count.
//! `Complete` events print the full routing/response summary.

use anyhow::Context;
use tokio_stream::StreamExt;

use crate::canonical::Role;
use crate::logging::{LogEntry, RouterEvent};

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
                match serde_json::from_str::<RouterEvent>(data) {
                    Ok(RouterEvent::Start { id: _, ts_ms, model, in_flight }) => {
                        let secs = ts_ms / 1000;
                        let time = format!(
                            "{:02}:{:02}:{:02}",
                            (secs / 3600) % 24,
                            (secs / 60) % 60,
                            secs % 60,
                        );
                        println!(
                            "{DIM}{time}  ⋯  {RESET}{BLUE}{model}{RESET}{DIM}  ({in_flight} in flight){RESET}"
                        );
                    }
                    Ok(RouterEvent::Classified { id: _, ts_ms: _, tags }) if !tags.is_empty() => {
                        let tag_str = tags.join(", ");
                        println!("{DIM}  → classified: [{tag_str}]{RESET}");
                    }
                    Ok(RouterEvent::Routed { id: _, ts_ms: _, provider, model }) => {
                        println!("{DIM}  → routed: {RESET}{BLUE}{provider}{RESET}{DIM}/{RESET}{BLUE}{model}{RESET}");
                    }
                    Ok(RouterEvent::Complete { id: _, entry }) => {
                        print_entry(&entry);
                    }
                    Ok(_) => {}
                    Err(err) => eprintln!("{DIM}[parse error: {err}]{RESET}"),
                }
            }
        }
    }

    Ok(())
}

fn print_entry(e: &LogEntry) {
    let secs = e.ts_ms / 1000;
    let time = format!("{:02}:{:02}:{:02}", (secs / 3600) % 24, (secs / 60) % 60, secs % 60);

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
    if let Some(resp) = &e.response {
        for tag in &resp.tags {
            header.push_str(&format!("  {RED}<{tag}>{RESET}"));
        }
    }

    let dur_color = if e.error.is_some() { RED } else { DIM };
    header.push_str(&format!("  {dur_color}{}ms{RESET}", e.duration_ms));

    println!("{header}");

    let last_user = e
        .messages
        .iter()
        .rev()
        .find(|m| m.role == Role::User)
        .map(|m| m.content.as_str())
        .unwrap_or("");
    println!("  {DIM}prompt:{RESET}  {}", truncate(last_user, 120));

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

fn truncate(s: &str, max: usize) -> String {
    let flat: String = s.chars().map(|c| if c == '\n' { ' ' } else { c }).collect();
    let mut chars = flat.chars();
    let head: String = chars.by_ref().take(max).collect();
    if chars.next().is_some() { head + "…" } else { head }
}
