//! Observable persistence; no driver overrides, warmup replay or hidden model passes.
use anyhow::{Result, ensure};
use cubecl::config::{CubeClRuntimeConfig, RuntimeConfig, cache::CacheConfig};
use cubecl_environment::persistence::{Access, Database, Insertion, Origin};
use std::{
    collections::HashSet,
    path::{Path, PathBuf},
    sync::{Mutex, OnceLock},
};

pub struct KernelCache {
    enabled: bool,
    root: Option<PathBuf>,
    db: Option<Database>,
    previous_probe: bool,
    error: Option<String>,
    initial_entries: u64,
}
struct CacheLogger;
impl log::Log for CacheLogger {
    fn enabled(&self, m: &log::Metadata<'_>) -> bool {
        m.level() <= log::Level::Warn
            && (m.target().starts_with("cubecl") || m.target().starts_with("turso"))
    }
    fn log(&self, r: &log::Record<'_>) {
        if !self.enabled(r.metadata()) {
            return;
        }
        static SEEN: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
        let message: String = format!("{}: {}", r.target(), r.args())
            .chars()
            .take(2048)
            .collect();
        if let Ok(mut seen) = SEEN.get_or_init(|| Mutex::new(HashSet::new())).lock() {
            if seen.len() < 32 && seen.insert(message.clone()) {
                eprintln!("[cache warning] {message}");
            }
        }
    }
    fn flush(&self) {}
}
fn default_root() -> PathBuf {
    if let Some(p) = std::env::var_os("LINO_CACHE_DIR") {
        return PathBuf::from(p);
    }
    if let Some(p) = std::env::var_os("LOCALAPPDATA") {
        return PathBuf::from(p).join("lino-rs").join("kernel-cache");
    }
    if let Some(p) = std::env::var_os("XDG_CACHE_HOME") {
        return PathBuf::from(p).join("lino-rs");
    }
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(Path::to_path_buf))
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".lino-kernels")
}
pub fn configure(root: Option<&Path>, enabled: bool) -> Result<KernelCache> {
    static LOGGER: CacheLogger = CacheLogger;
    if enabled && log::set_logger(&LOGGER).is_ok() {
        log::set_max_level(log::LevelFilter::Warn);
    }
    let mut config = CubeClRuntimeConfig::from_current_dir().override_from_env();
    config.environment.records.level = cubecl::records::RecordLevel::Off;
    config.environment.records.keep_sessions = Some(0);
    let mut info = KernelCache {
        enabled,
        root: None,
        db: None,
        previous_probe: false,
        error: None,
        initial_entries: 0,
    };
    if enabled {
        let root = root.map(Path::to_path_buf).unwrap_or_else(default_root);
        let root = if root.is_absolute() {
            root
        } else {
            std::env::current_dir()?.join(root)
        };
        std::fs::create_dir_all(&root)?;
        config.environment.path = CacheConfig::Directory(root.join("runtime"));
        config.environment.name = format!("lino-v1-{}", env!("LINO_REVISION"));
        config.compilation.cache = true;
        config.autotune.disable_cache = false;
        config.throughput.disable_cache = false;
        info.root = Some(root);
    } else {
        let exe = std::env::current_exe()?;
        ensure!(exe.is_file(), "Cannot select process-local runtime storage");
        config.environment.path = CacheConfig::Directory(exe);
        config.compilation.cache = false;
        config.autotune.disable_cache = true;
        // Throughput has no memory-only switch: the impossible root forces its RAM fallback.
        config.throughput.disable_cache = false;
    }
    ensure!(
        CubeClRuntimeConfig::try_set(config),
        "Runtime already initialized"
    );
    if enabled {
        let path = cubecl::environment::path();
        let probe = (|| -> Result<(Database, bool), String> {
            let db = Database::open_at(&path, Access::ReadWrite)?;
            let ns = "lino/persistence-probe/v1";
            let previous = db
                .get(ns, b"probe")
                .as_ref()
                .is_some_and(|v| &v[..] == b"ok");
            if !previous {
                match db.insert(ns, b"probe", b"ok", Origin::Local) {
                    Insertion::Stored => {}
                    Insertion::Conflict(v) if &v[..] == b"ok" => {}
                    other => return Err(format!("Cache write probe failed: {other:?}")),
                }
            }
            if !db
                .get(ns, b"probe")
                .as_ref()
                .is_some_and(|v| &v[..] == b"ok")
            {
                return Err("Cache read-after-write probe failed".into());
            }
            Ok((db, previous))
        })();
        match probe {
            Ok((db, previous)) => {
                info.initial_entries = db.summary().iter().map(|r| r.entries).sum();
                info.previous_probe = previous;
                info.db = Some(db);
                crate::progress::message(format!(
                    "Kernel cache verified: {} (previous-process probe: {previous}, {} entries)",
                    path.display(),
                    info.initial_entries
                ));
            }
            Err(error) => {
                eprintln!(
                    "Kernel cache probe failed: {error}. Cross-process reuse is not confirmed; runtime fallback may be used."
                );
                info.error = Some(error);
            }
        }
    }
    Ok(info)
}
impl KernelCache {
    pub fn report(&self) -> serde_json::Value {
        let mut compiled = 0u64;
        let mut tuned = 0u64;
        let mut entries = 0u64;
        let mut bytes = 0u64;
        if let Some(db) = &self.db {
            for r in db.summary() {
                entries += r.entries;
                bytes += r.bytes;
                if ["vulkan/", "cuda/", "wgpu/", "dx12/"]
                    .iter()
                    .any(|p| r.namespace.starts_with(*p))
                {
                    compiled += r.entries;
                }
                if r.namespace.starts_with("autotune/") {
                    tuned += r.entries;
                }
            }
        }
        serde_json::json!({"enabled":self.enabled,"root":self.root,"revision_namespace":format!("lino-v1-{}",env!("LINO_REVISION")),"read_write_probe_ok":self.db.is_some(),"previous_process_probe":self.previous_probe,"probe_error":self.error,"initial_entries":self.initial_entries,"current_entries":entries,"compiled_artifact_entries":compiled,"autotune_entries":tuned,"payload_bytes":bytes,"note":"Stored entry counts are not individual cache-hit counts; driver pipeline creation may remain process-local"})
    }
}
