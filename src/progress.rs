//! SuperNormal-style terminal progress, driven only by CPU counters and time.
//! No tensor reads, device synchronization, or inference scheduling changes.
use std::{
    io::{self, IsTerminal, Write},
    sync::{Mutex, OnceLock},
    time::{Duration, Instant},
};

struct Stage {
    id: u64,
    label: String,
    total: usize,
    done: usize,
    unit: &'static str,
    started: Instant,
}
struct State {
    enabled: bool,
    terminal: bool,
    columns: usize,
    width: usize,
    last_draw: Instant,
    next_id: u64,
    view: Option<(u64, String, Instant)>,
    block: Option<(usize, usize)>,
    stages: Vec<Stage>,
}
static STATE: OnceLock<Mutex<State>> = OnceLock::new();

fn state() -> &'static Mutex<State> {
    STATE.get_or_init(|| {
        Mutex::new(State {
            enabled: true,
            terminal: io::stderr().is_terminal(),
            columns: std::env::var("COLUMNS")
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(120)
                .clamp(60, 240),
            width: 0,
            last_draw: Instant::now() - Duration::from_secs(30),
            next_id: 0,
            view: None,
            block: None,
            stages: Vec::new(),
        })
    })
}

pub fn configure(enabled: bool) {
    if let Ok(mut s) = state().lock() {
        s.enabled = enabled;
    }
    // A CPU-only heartbeat keeps elapsed time visible during lazy compilation,
    // tuning, CPU positional embeddings, and long device calls. Spawn only once.
    static HEARTBEAT: OnceLock<()> = OnceLock::new();
    if enabled {
        HEARTBEAT.get_or_init(|| {
            let _ = std::thread::Builder::new()
                .name("lino-progress".into())
                .spawn(|| {
                    loop {
                        std::thread::sleep(Duration::from_millis(250));
                        if let Ok(mut s) = state().lock() {
                            draw(&mut s, false);
                        }
                    }
                });
        });
    }
}

pub struct View {
    id: u64,
    finished: bool,
}
pub fn view(label: &str, index: usize, total: usize) -> View {
    let mut s = state().lock().expect("progress state");
    clear(&mut s);
    s.next_id += 1;
    let id = s.next_id;
    s.view = Some((
        id,
        format!("View {index}/{total} {}", clean(label)),
        Instant::now(),
    ));
    s.block = None;
    s.stages.clear();
    s.last_draw = Instant::now() - Duration::from_secs(30);
    draw(&mut s, true);
    View {
        id,
        finished: false,
    }
}
impl View {
    pub fn block(&self, current: usize, total: usize) {
        if let Ok(mut s) = state().lock() {
            if s.view.as_ref().is_some_and(|v| v.0 == self.id) {
                s.block = Some((current, total));
                draw(&mut s, false);
            }
        }
    }
    pub fn finish(mut self) {
        self.end("completed");
        self.finished = true;
    }
    fn end(&self, outcome: &str) {
        if let Ok(mut s) = state().lock() {
            if let Some((id, label, started)) = &s.view {
                if *id != self.id {
                    return;
                }
                let text = format!(
                    "{label} {outcome} | elapsed {}",
                    clock(started.elapsed().as_secs_f64())
                );
                clear(&mut s);
                if s.enabled {
                    let _ = writeln!(io::stderr().lock(), "{text}");
                }
                s.view = None;
                s.stages.clear();
                s.block = None;
            }
        }
    }
}
impl Drop for View {
    fn drop(&mut self) {
        if !self.finished {
            self.end("stopped");
        }
    }
}

/// Nested stages let attention report real query-block completion while keeping
/// the encoder branch/stage visible. Counters describe work, not GPU utilization.
pub struct Phase {
    id: Option<u64>,
}
pub fn phase(label: impl Into<String>, total: usize, unit: &'static str) -> Phase {
    let mut s = state().lock().expect("progress state");
    if !s.enabled || s.view.is_none() {
        return Phase { id: None };
    }
    s.next_id += 1;
    let id = s.next_id;
    s.stages.push(Stage {
        id,
        label: clean(&label.into()),
        total,
        done: 0,
        unit,
        started: Instant::now(),
    });
    draw(&mut s, false);
    Phase { id: Some(id) }
}
impl Phase {
    pub fn set(&self, done: usize) {
        let Some(id) = self.id else {
            return;
        };
        if let Ok(mut s) = state().lock() {
            if let Some(stage) = s.stages.iter_mut().find(|p| p.id == id) {
                stage.done = done.min(stage.total);
            }
            draw(&mut s, false);
        }
    }
}
impl Drop for Phase {
    fn drop(&mut self) {
        let Some(id) = self.id else {
            return;
        };
        if let Ok(mut s) = state().lock() {
            if let Some(index) = s.stages.iter().position(|p| p.id == id) {
                s.stages.truncate(index);
            }
            draw(&mut s, false);
        }
    }
}

/// Explicit messages must not collide with the live stderr line.
pub fn message(text: String) {
    if let Ok(mut s) = state().lock() {
        clear(&mut s);
        let _ = writeln!(io::stderr().lock(), "{text}");
        draw(&mut s, false);
    } else {
        eprintln!("{text}");
    }
}

fn draw(s: &mut State, force: bool) {
    if !s.enabled {
        return;
    }
    let Some((_, label, started)) = &s.view else {
        return;
    };
    let interval = if s.terminal {
        Duration::from_millis(250)
    } else {
        Duration::from_secs(15)
    };
    if !force && s.last_draw.elapsed() < interval {
        return;
    }
    let prefix = match s.block {
        Some((i, n)) => format!("{label} block {i}/{n}"),
        None => label.clone(),
    };
    let context = match (s.stages.first(), s.stages.last()) {
        (Some(first), Some(last)) if first.id != last.id => format!(
            "{} {}/{} {} / {}",
            first.label, first.done, first.total, first.unit, last.label
        ),
        (Some(first), _) => first.label.clone(),
        _ => String::new(),
    };
    let elapsed = clock(started.elapsed().as_secs_f64());
    let body = if let Some(stage) = s.stages.last() {
        let fraction = stage.done as f64 / stage.total.max(1) as f64;
        let filled = (fraction * 8.) as usize;
        // Hide premature ETA until at least two actual work units completed.
        let eta = if stage.done >= 2 && stage.done < stage.total {
            let seconds = stage.started.elapsed().as_secs_f64() * (stage.total - stage.done) as f64
                / stage.done as f64;
            format!("~{}", clock(seconds))
        } else if stage.done == stage.total {
            "00:00".into()
        } else {
            "--:--".into()
        };
        let unit = if s.terminal {
            match stage.unit {
                "query blocks" => "q",
                "channels" => "ch",
                "pixels" => "px",
                "layers" => "lyr",
                "batches" => "bat",
                other => other,
            }
        } else {
            stage.unit
        };
        let text = format!(
            "[{}{}] {:3.0}% {}/{} {} | elapsed {elapsed} | stage ETA {eta}",
            "=".repeat(filled),
            "-".repeat(8 - filled),
            fraction * 100.,
            stage.done,
            stage.total,
            unit
        );
        if s.terminal {
            text.replace(" | elapsed ", " | ")
                .replace(" | stage ETA ", " ETA ")
        } else {
            text
        }
    } else {
        format!("starting | elapsed {elapsed}")
    };
    let text = if s.terminal {
        let remaining = s.columns.saturating_sub(body.chars().count() + 7);
        let context = shorten(&context, (remaining / 2).min(40));
        let available = remaining.saturating_sub(context.chars().count());
        shorten(
            &format!("{} | {context} | {body}", shorten(&prefix, available)),
            s.columns - 1,
        )
    } else {
        format!("{prefix} | {context} | {body}")
    };
    let mut stderr = io::stderr().lock();
    if s.terminal {
        let width = text.chars().count();
        let _ = write!(
            stderr,
            "\r{text}{}",
            " ".repeat(s.width.saturating_sub(width))
        );
        let _ = stderr.flush();
        s.width = width;
    } else {
        let _ = writeln!(stderr, "{text}");
    }
    s.last_draw = Instant::now();
}
fn clear(s: &mut State) {
    if s.terminal && s.width > 0 {
        let mut stderr = io::stderr().lock();
        let _ = write!(stderr, "\r{}\r", " ".repeat(s.width));
        let _ = stderr.flush();
        s.width = 0;
    }
}
fn clean(s: &str) -> String {
    s.chars().filter(|c| !c.is_control()).collect()
}
fn shorten(s: &str, limit: usize) -> String {
    if s.chars().count() <= limit {
        s.into()
    } else {
        format!(
            "{}~",
            s.chars().take(limit.saturating_sub(1)).collect::<String>()
        )
    }
}
fn clock(seconds: f64) -> String {
    let n = seconds.max(0.) as u64;
    if n >= 3600 {
        format!("{}:{:02}:{:02}", n / 3600, n / 60 % 60, n % 60)
    } else {
        format!("{:02}:{:02}", n / 60, n % 60)
    }
}
