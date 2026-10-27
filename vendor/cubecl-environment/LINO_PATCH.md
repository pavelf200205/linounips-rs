# Local Windows persistence repair

Upstream cubecl-environment 0.11.0, MIT OR Apache-2.0, from the installed Cargo
registry source. The only source modification is in src/persistence/turso.rs:
disable Turso's experimental multiprocess WAL on Windows because default
WindowsIO does not implement its shared-WAL coordination. Normal Windows file
locking remains active. Other platforms keep upstream behavior.

This follows the same fix in the sibling unimsps-rs checkout; it does not copy
its additional checkpoint/write-freezing modifications. The patch is local to
this repository through Cargo.toml. No global registry files were modified.
Use separate cache roots for simultaneous Windows processes.
