//! Process logging to stderr, driven only by CPU timestamps.
//! No interactive progress bars, background threads, or hot-loop mutex contention.
use std::{
    io::{self, Write},
    sync::atomic::{AtomicBool, Ordering},
    time::Instant,
};

static ENABLED: AtomicBool = AtomicBool::new(true);

pub fn configure(enabled: bool) {
    ENABLED.store(enabled, Ordering::Relaxed);
}

pub struct View {
    label: String,
    started: Instant,
    finished: bool,
}

pub fn view(label: &str, index: usize, total: usize) -> View {
    let tag = format!("View {index}/{total} {}", clean(label));
    message(format!("{tag} started"));
    View {
        label: tag,
        started: Instant::now(),
        finished: false,
    }
}

impl View {
    pub fn block(&self, current: usize, total: usize) {
        message(format!(
            "{} | starting block {current}/{total} (elapsed {})",
            self.label,
            clock(self.started.elapsed().as_secs_f64())
        ));
    }

    pub fn finish(mut self) {
        self.end("completed");
        self.finished = true;
    }

    fn end(&self, outcome: &str) {
        message(format!(
            "{} {outcome} | elapsed {}",
            self.label,
            clock(self.started.elapsed().as_secs_f64())
        ));
    }
}

impl Drop for View {
    fn drop(&mut self) {
        if !self.finished {
            self.end("stopped");
        }
    }
}

pub struct Phase;

#[inline]
pub fn phase(_label: impl Into<String>, _total: usize, _unit: &'static str) -> Phase {
    Phase
}

impl Phase {
    #[inline]
    pub fn set(&self, _done: usize) {}
}

pub fn message(text: String) {
    if ENABLED.load(Ordering::Relaxed) {
        let _ = writeln!(io::stderr().lock(), "{text}");
    }
}

fn clean(s: &str) -> String {
    if s.bytes().all(|b| b >= 0x20 && b != 0x7f) {
        s.to_owned()
    } else {
        s.chars().filter(|c| !c.is_control()).collect()
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
