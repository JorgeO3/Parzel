#![allow(unused)]

//! Internal Prelude
//!
//! Import this module via `use crate::prelude::*;` to get access to
//! core types, memory allocation tools, and the standardized Error system.

// Re-export crate Error and Result types
pub use crate::error::Error;

// Re-exports from alloc for convenient use without `std`
pub use alloc::boxed::Box;
pub use alloc::string::{String, ToString};
pub use alloc::vec;
pub use alloc::vec::Vec; // Para la macro vec![]

pub use core::option::Option;

// Alias for format! macro (Using alloc::format as std::format is unavailable in no_std)
pub use alloc::format as f;

// Generic Wrapper tuple struct for newtype pattern
pub struct W<T>(pub T);

// Common traits for safe casting and conversions
pub use core::convert::{TryFrom, TryInto};

pub type Result<T> = core::result::Result<T, Error>;

