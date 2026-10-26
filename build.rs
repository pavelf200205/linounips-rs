use std::{path::Path, process::Command};

fn main() {
    // Embed the revision without changing the inference runtime.
    let git = |args: &[&str]| -> Option<String> {
        let output = Command::new("git").args(args).output().ok()?;
        output
            .status
            .success()
            .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
    };
    for name in ["HEAD", "index", "packed-refs"] {
        if let Some(path) = git(&["rev-parse", "--git-path", name]) {
            if Path::new(&path).exists() {
                println!("cargo:rerun-if-changed={path}");
            }
        }
    }
    if let Some(reference) = git(&["symbolic-ref", "-q", "HEAD"]) {
        if let Some(path) = git(&["rev-parse", "--git-path", &reference]) {
            println!("cargo:rerun-if-changed={path}");
        }
    }
    let mut revision =
        git(&["rev-parse", "--short=12", "HEAD"]).unwrap_or_else(|| "unknown".into());
    if git(&["status", "--porcelain", "--untracked-files=no"]).is_some_and(|s| !s.is_empty()) {
        revision.push_str("-dirty");
    }
    println!("cargo:rustc-env=LINO_REVISION={revision}");
    for path in [
        "src",
        "Cargo.toml",
        "Cargo.lock",
        "build.rs",
        "tools/build_release.ps1",
    ] {
        println!("cargo:rerun-if-changed={path}");
    }
}
