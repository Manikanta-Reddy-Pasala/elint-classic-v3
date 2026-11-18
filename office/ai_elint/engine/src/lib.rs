//! Engine library — exposes modules for integration tests.
//!
//! The public API here (`read_batch`, `insert_batch`, `table_count`,
//! `pg_dsn`, `emitter_id`, `Pulse` fields, `Config` fields) is consumed
//! by the `elint_engine` binary and external callers. The bin target
//! has its own `mod` copies of the same source files, so the lib's
//! copies are flagged as dead code by the compiler. They are not dead —
//! they are the library's public surface.
#![allow(dead_code)]

#[allow(dead_code, unused_imports, unused_variables, unused_mut)]
pub mod band;
#[allow(dead_code, unused_imports, unused_variables, unused_mut)]
pub mod ch_read;
#[allow(dead_code, unused_imports, unused_variables, unused_mut)]
pub mod config;
