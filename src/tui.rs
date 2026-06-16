//! `opensourcellmrouter tui [BASE_URL]`
//!
//! Three-pane terminal UI:
//!   top         — live pipeline feed (same SSE stream as the browser dashboard)
//!   bottom-left — running stats (provider breakdown, tag counts, latency)
//!   bottom-right — built-in chat client
//!
//! Key bindings
//!   q / Ctrl-C      quit (from any pane)
//!   Tab             cycle focus: Feed → Chat → Feed
//!   ↑ / k, ↓ / j   scroll the feed (when Feed is focused)
//!   i               jump to Chat pane
//!   Enter           send the typed message
//!   Esc             clear input / return to Feed
//!   :model <name>   change the model used by the chat client

use std::collections::HashMap;
use std::collections::VecDeque;
use std::io::Stdout;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crossterm::{
    event::{self, Event, KeyCode, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, Paragraph, Wrap},
    Frame, Terminal,
};
use tokio_stream::StreamExt;

use crate::canonical::Role;
use crate::logging::LogEntry;

const MAX_EVENTS: usize = 200;

// ── shared state ─────────────────────────────────────────────────────────────

#[derive(Default)]
struct Stats {
    total: u32,
    errors: u32,
    total_duration_ms: u128,
    by_provider: HashMap<String, u32>,
    by_tag: HashMap<String, u32>,
    /// Timestamp (ms since epoch) of the last *successful* response per provider.
    last_ok: HashMap<String, u128>,
}

impl Stats {
    fn ingest(&mut self, e: &LogEntry) {
        self.total += 1;
        if e.error.is_some() {
            self.errors += 1;
        } else {
            self.last_ok.insert(e.provider.clone(), e.ts_ms);
        }
        self.total_duration_ms += e.duration_ms;
        *self.by_provider.entry(e.provider.clone()).or_default() += 1;
        for tag in &e.tags {
            *self.by_tag.entry(tag.clone()).or_default() += 1;
        }
    }

    fn avg_latency_ms(&self) -> u64 {
        if self.total == 0 {
            0
        } else {
            (self.total_duration_ms / self.total as u128) as u64
        }
    }
}

#[derive(PartialEq, Clone, Copy)]
enum Focus {
    Feed,
    Chat,
}

struct AppState {
    events: VecDeque<LogEntry>,
    stats: Stats,
    scroll: usize,
    focus: Focus,
    input: String,
    model: String,
    last_response: Option<String>,
    /// Routing metadata from the most recent completed request (via SSE).
    last_routed: Option<RoutedMeta>,
    sending: bool,
    sse_status: String,
    base_url: String,
}

struct RoutedMeta {
    provider: String,
    sent_model: String,
    tags: Vec<String>,
}

impl AppState {
    fn new(base_url: &str) -> Self {
        AppState {
            events: VecDeque::new(),
            stats: Stats::default(),
            scroll: 0,
            focus: Focus::Feed,
            input: String::new(),
            model: "llama3.1:8b".to_string(),
            last_response: None,
            last_routed: None,
            sending: false,
            sse_status: "connecting…".to_string(),
            base_url: base_url.to_string(),
        }
    }

    fn push_event(&mut self, entry: LogEntry) {
        self.stats.ingest(&entry);
        self.last_routed = Some(RoutedMeta {
            provider: entry.provider.clone(),
            sent_model: entry.sent_model.clone(),
            tags: entry.tags.clone(),
        });
        self.events.push_front(entry);
        if self.events.len() > MAX_EVENTS {
            self.events.pop_back();
        }
        // Keep scroll pointing at the same event when new ones arrive at top.
        if self.scroll > 0 {
            self.scroll += 1;
        }
    }
}

// ── public entry point ────────────────────────────────────────────────────────

pub async fn run(base_url: &str) -> anyhow::Result<()> {
    let state = Arc::new(Mutex::new(AppState::new(base_url)));

    // Background task: read SSE and push events into shared state.
    {
        let state = state.clone();
        let url = format!("{}/dashboard/events", base_url.trim_end_matches('/'));
        tokio::spawn(async move { read_sse(url, state).await });
    }

    enable_raw_mode()?;
    let mut stdout = std::io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    terminal.clear()?;

    let result = event_loop(&mut terminal, &state).await;

    let _ = disable_raw_mode();
    let _ = execute!(terminal.backend_mut(), LeaveAlternateScreen);
    let _ = terminal.show_cursor();

    result
}

// ── SSE reader ────────────────────────────────────────────────────────────────

async fn read_sse(url: String, state: Arc<Mutex<AppState>>) {
    let client = reqwest::Client::new();
    let resp = match client
        .get(&url)
        .header("Accept", "text/event-stream")
        .send()
        .await
    {
        Ok(r) => {
            state.lock().unwrap().sse_status = "live".to_string();
            r
        }
        Err(e) => {
            state.lock().unwrap().sse_status = format!("error: {e}");
            return;
        }
    };

    let mut stream = resp.bytes_stream();
    let mut buf = String::new();

    while let Some(chunk) = stream.next().await {
        let chunk = match chunk {
            Ok(c) => c,
            Err(e) => {
                state.lock().unwrap().sse_status = format!("disconnected: {e}");
                return;
            }
        };
        buf.push_str(&String::from_utf8_lossy(&chunk));

        while let Some(nl) = buf.find('\n') {
            let line = buf[..nl].trim_end_matches('\r').to_owned();
            buf.drain(..=nl);
            if let Some(data) = line.strip_prefix("data: ") {
                if let Ok(entry) = serde_json::from_str::<LogEntry>(data) {
                    state.lock().unwrap().push_event(entry);
                }
            }
        }
    }

    state.lock().unwrap().sse_status = "disconnected".to_string();
}

// ── event loop ────────────────────────────────────────────────────────────────

async fn event_loop(
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    state: &Arc<Mutex<AppState>>,
) -> anyhow::Result<()> {
    let mut tick = tokio::time::interval(Duration::from_millis(50));

    loop {
        tick.tick().await;

        // Non-blocking drain of terminal events.
        while event::poll(Duration::ZERO)? {
            if let Event::Key(key) = event::read()? {
                if handle_key(key, state).await? {
                    return Ok(());
                }
            }
        }

        let s = state.lock().unwrap();
        terminal.draw(|f| render(f, &s))?;
    }
}

// ── input handling ────────────────────────────────────────────────────────────

/// Returns `true` if the app should quit.
async fn handle_key(
    key: event::KeyEvent,
    state: &Arc<Mutex<AppState>>,
) -> anyhow::Result<bool> {
    // Ctrl-C always quits.
    if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
        return Ok(true);
    }

    let mut s = state.lock().unwrap();

    match s.focus {
        Focus::Feed => match key.code {
            KeyCode::Char('q') => return Ok(true),
            KeyCode::Tab | KeyCode::Char('i') => s.focus = Focus::Chat,
            KeyCode::Up | KeyCode::Char('k') => {
                let max = s.events.len().saturating_sub(1);
                if s.scroll < max {
                    s.scroll += 1;
                }
            }
            KeyCode::Down | KeyCode::Char('j') => {
                s.scroll = s.scroll.saturating_sub(1);
            }
            _ => {}
        },
        Focus::Chat => match key.code {
            KeyCode::Esc => {
                s.input.clear();
                s.focus = Focus::Feed;
            }
            KeyCode::Tab => {
                s.focus = Focus::Feed;
            }
            KeyCode::Backspace => {
                s.input.pop();
            }
            KeyCode::Enter => {
                if s.input.is_empty() || s.sending {
                    // nothing to do
                } else if let Some(name) = s.input.strip_prefix(":model ") {
                    s.model = name.trim().to_string();
                    s.input.clear();
                } else {
                    let msg = std::mem::take(&mut s.input);
                    let model = s.model.clone();
                    let url = s.base_url.clone();
                    s.sending = true;
                    drop(s); // release lock before async work

                    let state2 = state.clone();
                    tokio::spawn(async move {
                        let result = send_chat(&url, &model, &msg).await;
                        let mut st = state2.lock().unwrap();
                        st.sending = false;
                        st.last_response = Some(match result {
                            Ok(text) => text,
                            Err(e) => format!("error: {e}"),
                        });
                    });
                    return Ok(false);
                }
            }
            KeyCode::Char(c) => s.input.push(c),
            _ => {}
        },
    }

    Ok(false)
}

async fn send_chat(base_url: &str, model: &str, message: &str) -> anyhow::Result<String> {
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{}/v1/chat/completions", base_url.trim_end_matches('/')))
        .json(&serde_json::json!({
            "model": model,
            "messages": [{"role": "user", "content": message}]
        }))
        .send()
        .await?
        .json::<serde_json::Value>()
        .await?;

    if let Some(content) = resp["choices"][0]["message"]["content"].as_str() {
        return Ok(content.to_string());
    }
    if let Some(err) = resp["error"]["message"].as_str() {
        return Ok(format!("error: {err}"));
    }
    Ok("(empty response)".to_string())
}

// ── rendering ─────────────────────────────────────────────────────────────────

fn render(f: &mut Frame, s: &AppState) {
    let outer = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
        .split(f.area());

    let bottom = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(35), Constraint::Percentage(65)])
        .split(outer[1]);

    render_feed(f, s, outer[0]);
    render_stats(f, s, bottom[0]);
    render_chat(f, s, bottom[1]);
}

fn render_feed(f: &mut Frame, s: &AppState, area: ratatui::layout::Rect) {
    let focused = s.focus == Focus::Feed;
    let border_style = focus_border(focused);

    let title = format!(
        " Pipeline  {}  {} events ",
        s.sse_status,
        s.events.len()
    );
    let hint = if focused {
        " ↑/↓ scroll  Tab/i → chat  q quit "
    } else {
        " Tab/i to focus "
    };

    let block = Block::default()
        .title(title)
        .title_bottom(hint)
        .borders(Borders::ALL)
        .border_style(border_style);

    let inner = block.inner(area);
    f.render_widget(block, area);

    let items: Vec<ListItem> = s
        .events
        .iter()
        .skip(s.scroll)
        .map(entry_to_list_item)
        .collect();

    f.render_widget(List::new(items), inner);
}

fn entry_to_list_item(e: &LogEntry) -> ListItem<'static> {
    let secs = (e.ts_ms / 1000) as u64;
    let time = format!("{:02}:{:02}:{:02}", (secs / 3600) % 24, (secs / 60) % 60, secs % 60);

    // header: time  provider  model [→ rewrite]  [tags]  Nms
    let mut header: Vec<Span<'static>> = vec![
        Span::styled(time, Style::default().fg(Color::DarkGray)),
        Span::raw("  "),
        Span::styled(e.provider.clone(), Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        Span::raw("  "),
    ];

    if e.requested_model != e.sent_model {
        header.push(Span::styled(e.requested_model.clone(), Style::default().fg(Color::DarkGray)));
        header.push(Span::styled(" → ", Style::default().fg(Color::DarkGray)));
        header.push(Span::styled(e.sent_model.clone(), Style::default().fg(Color::Blue)));
    } else {
        header.push(Span::styled(e.sent_model.clone(), Style::default().fg(Color::Blue)));
    }

    for tag in &e.tags {
        header.push(Span::raw("  "));
        header.push(Span::styled(format!("[{tag}]"), Style::default().fg(Color::Yellow)));
    }
    for plugin in &e.plugins {
        header.push(Span::raw("  "));
        header.push(Span::styled(format!("({plugin})"), Style::default().fg(Color::Magenta)));
    }

    let dur_color = if e.error.is_some() { Color::Red } else { Color::DarkGray };
    header.push(Span::styled(
        format!("  {}ms", e.duration_ms),
        Style::default().fg(dur_color),
    ));

    let last_user = e
        .messages
        .iter()
        .rev()
        .find(|m| m.role == Role::User)
        .map(|m| m.content.replace('\n', " "))
        .unwrap_or_default();

    let body = if let Some(err) = &e.error {
        Line::from(vec![
            Span::styled("  error: ", Style::default().fg(Color::DarkGray)),
            Span::styled(trunc(err, 90), Style::default().fg(Color::Red)),
        ])
    } else if let Some(resp) = &e.response {
        Line::from(vec![
            Span::styled("  reply: ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                trunc(resp.content.replace('\n', " ").as_str(), 90),
                Style::default().fg(Color::White),
            ),
        ])
    } else {
        Line::from("")
    };

    ListItem::new(vec![
        Line::from(header),
        Line::from(vec![
            Span::styled("  prompt:", Style::default().fg(Color::DarkGray)),
            Span::raw(" "),
            Span::raw(trunc(&last_user, 90)),
        ]),
        body,
        Line::from(""),
    ])
}

fn render_stats(f: &mut Frame, s: &AppState, area: ratatui::layout::Rect) {
    let mut lines: Vec<Line> = vec![
        Line::from(vec![
            Span::styled(
                s.stats.total.to_string(),
                Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
            ),
            Span::styled(" requests", Style::default().fg(Color::DarkGray)),
        ]),
        Line::from(vec![
            Span::styled(
                s.stats.errors.to_string(),
                Style::default().fg(if s.stats.errors > 0 { Color::Red } else { Color::DarkGray }),
            ),
            Span::styled(" errors", Style::default().fg(Color::DarkGray)),
        ]),
        Line::from(vec![
            Span::styled(
                format!("{}ms", s.stats.avg_latency_ms()),
                Style::default().fg(Color::White),
            ),
            Span::styled(" avg", Style::default().fg(Color::DarkGray)),
        ]),
    ];

    if !s.stats.by_provider.is_empty() {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "providers",
            Style::default().fg(Color::DarkGray).add_modifier(Modifier::ITALIC),
        )));
        let mut providers: Vec<(&String, &u32)> = s.stats.by_provider.iter().collect();
        providers.sort_by(|a, b| b.1.cmp(a.1));
        for (name, count) in providers {
            let pct = if s.stats.total > 0 { count * 100 / s.stats.total } else { 0 };
            let ago = s.stats.last_ok.get(name)
                .map(|&ts| ago_str(ts))
                .unwrap_or_else(|| "never".to_string());
            let ago_color = if ago == "never" { Color::Red } else { Color::DarkGray };
            lines.push(Line::from(vec![
                Span::styled(format!("  {name}"), Style::default().fg(Color::Cyan)),
                Span::styled(format!("  {count} ({pct}%)"), Style::default().fg(Color::DarkGray)),
                Span::styled(format!("  last {ago}"), Style::default().fg(ago_color)),
            ]));
        }
    }

    if !s.stats.by_tag.is_empty() {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "tags",
            Style::default().fg(Color::DarkGray).add_modifier(Modifier::ITALIC),
        )));
        let mut tags: Vec<(&String, &u32)> = s.stats.by_tag.iter().collect();
        tags.sort_by(|a, b| b.1.cmp(a.1));
        for (tag, count) in tags {
            lines.push(Line::from(vec![
                Span::styled(format!("  [{tag}]"), Style::default().fg(Color::Yellow)),
                Span::styled(format!("  {count}"), Style::default().fg(Color::DarkGray)),
            ]));
        }
    }

    f.render_widget(
        Paragraph::new(lines).block(
            Block::default()
                .title(" Stats ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::DarkGray)),
        ),
        area,
    );
}

fn render_chat(f: &mut Frame, s: &AppState, area: ratatui::layout::Rect) {
    let focused = s.focus == Focus::Chat;
    let border_style = focus_border(focused);
    let hint = if focused {
        " Enter=send  Esc=back  :model <name>=change "
    } else {
        " Tab/i to focus "
    };

    let block = Block::default()
        .title(format!(" Chat  model: {} ", s.model))
        .title_bottom(hint)
        .borders(Borders::ALL)
        .border_style(border_style);

    let inner = block.inner(area);
    f.render_widget(block, area);

    // Split: response fills the space above, input is fixed 3 lines at bottom.
    let split = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(1), Constraint::Length(3)])
        .split(inner);

    // Response pane — title shows routing metadata from last SSE event.
    let route_title = if let Some(meta) = &s.last_routed {
        let mut t = format!(" {} / {} ", meta.provider, meta.sent_model);
        for tag in &meta.tags {
            t.push_str(&format!("[{tag}] "));
        }
        t
    } else {
        " response ".to_string()
    };

    let resp_text = if s.sending {
        "sending…".to_string()
    } else {
        s.last_response
            .clone()
            .unwrap_or_else(|| "(no messages yet — type below and press Enter)".to_string())
    };
    f.render_widget(
        Paragraph::new(resp_text)
            .style(Style::default().fg(if s.sending { Color::DarkGray } else { Color::White }))
            .wrap(Wrap { trim: false })
            .block(
                Block::default()
                    .title(route_title)
                    .borders(Borders::BOTTOM)
                    .border_style(Style::default().fg(Color::DarkGray)),
            ),
        split[0],
    );

    // Input field — show a block cursor when focused
    let cursor = if focused && !s.sending { "█" } else { "" };
    f.render_widget(
        Paragraph::new(format!("{}{}", s.input, cursor))
            .style(Style::default().fg(Color::White))
            .block(
                Block::default()
                    .title(" ▸ ")
                    .borders(Borders::ALL)
                    .border_style(if focused {
                        Style::default().fg(Color::Cyan)
                    } else {
                        Style::default().fg(Color::DarkGray)
                    }),
            ),
        split[1],
    );
}

// ── helpers ───────────────────────────────────────────────────────────────────

fn focus_border(focused: bool) -> Style {
    if focused {
        Style::default().fg(Color::Cyan)
    } else {
        Style::default().fg(Color::DarkGray)
    }
}

fn trunc(s: &str, max: usize) -> String {
    let mut chars = s.chars();
    let head: String = chars.by_ref().take(max).collect();
    if chars.next().is_some() { head + "…" } else { head }
}

fn ago_str(ts_ms: u128) -> String {
    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let secs = (now_ms.saturating_sub(ts_ms) / 1000) as u64;
    if secs < 60 {
        format!("{secs}s ago")
    } else if secs < 3600 {
        format!("{}m ago", secs / 60)
    } else {
        format!("{}h ago", secs / 3600)
    }
}
