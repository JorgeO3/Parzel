//! # Parzel - Hypersonic Search Engine (WASM Kernel)
//!
//! A high-performance, zero-allocation search engine designed for WebAssembly.
//!
//! ## Architecture: "Unsafe Shell, Safe Core"
//!
//! - **FFI boundary**: Minimal unsafe code for pointer handling
//! - **Core logic**: Fully safe Rust operating on validated slices
//! - **SIMD acceleration**: Platform-specific optimizations for bitmap scanning
//!
//! ## Modules
//!
//! - [`data`]: Index structures and binary format parsing
//! - [`engine`]: High-level search orchestration
//! - [`iter`]: Posting list iterators
//! - [`simd`]: SIMD-accelerated bitmap operations
//! - [`tokenizer`]: Query tokenization
//! - [`utils`]: Safe memory access abstractions

// Use no_std only for WASM production builds
#![cfg_attr(all(target_arch = "wasm32", not(test)), no_std)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::similar_names)]
#![allow(clippy::collapsible_if)]
#![allow(clippy::cast_possible_truncation)]

extern crate alloc;

// ============================================================================
// Allocator & Panic Handler (WASM production only)
// ============================================================================

#[cfg(all(feature = "lol_alloc", target_arch = "wasm32", not(test)))]
#[global_allocator]
static ALLOCATOR: lol_alloc::LockedAllocator<lol_alloc::FreeListAllocator> =
    lol_alloc::LockedAllocator::new(lol_alloc::FreeListAllocator::new());

#[cfg(all(target_arch = "wasm32", not(test)))]
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    core::arch::wasm32::unreachable()
}

// ============================================================================
// Module Declarations
// ============================================================================

mod data;
mod engine;
mod iter;
/// SIMD-accelerated operations (public for benchmarking).
pub mod simd;
mod tokenizer;
mod utils;

// ============================================================================
// Public API Re-exports
// ============================================================================

pub use data::HypersonicIndex;
pub use engine::{search, MAX_RESULTS};
pub use iter::{PostingIterator, SCRATCH_SIZE};
pub use tokenizer::{Tokenizer, MAX_TOKENS};

// ============================================================================
// FFI Boundary (WASM Exports)
// ============================================================================

use core::ptr;
use core::slice;

/// Static buffer for search results, accessible from JavaScript.
static mut RESULTS_BUFFER: [u32; MAX_RESULTS] = [0; MAX_RESULTS];

/// Performs a search and returns the number of results.
///
/// # Safety
///
/// Caller must ensure:
/// - `index_ptr` points to a valid byte array of at least `index_len` bytes
/// - `query_ptr` points to valid UTF-8 data of at least `query_len` bytes
/// - Both pointers remain valid for the duration of the call
/// - No other thread accesses `RESULTS_BUFFER` during or after this call
///
/// # Returns
///
/// Number of document IDs written to the results buffer.
/// Use `get_results_ptr` to access the results.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ffi_search(
    index_ptr: *const u8,
    index_len: usize,
    query_ptr: *const u8,
    query_len: usize,
) -> u32 {
    if index_ptr.is_null() || query_ptr.is_null() {
        return 0;
    }

    // SAFETY: Caller guarantees valid pointers and lengths
    let index_slice = unsafe { slice::from_raw_parts(index_ptr, index_len) };
    let query_slice = unsafe { slice::from_raw_parts(query_ptr, query_len) };

    let Ok(query) = core::str::from_utf8(query_slice) else {
        return 0;
    };

    // SAFETY: We have exclusive access in single-threaded WASM
    let results_slice = unsafe {
        let ptr = ptr::addr_of_mut!(RESULTS_BUFFER);
        slice::from_raw_parts_mut(ptr.cast::<u32>(), MAX_RESULTS)
    };

    #[allow(clippy::cast_possible_truncation)]
    let result_count = engine::search(index_slice, query, results_slice) as u32;
    result_count
}

/// Returns a pointer to the results buffer.
///
/// # Safety
///
/// The returned pointer is valid until the next call to `ffi_search`.
/// Caller must not write to this memory.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ffi_get_results_ptr() -> *const u32 {
    // SAFETY: We're returning a pointer to static memory
    ptr::addr_of!(RESULTS_BUFFER).cast()
}

/// Initializes the search index. Currently a no-op placeholder.
///
/// # Safety
///
/// This function is safe to call with any arguments.
/// Future versions may require valid pointer and length.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ffi_init_index(_ptr: *const u8, _len: usize) {
    // Reserved for future use (e.g., preloading index into WASM memory)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    /// Helper to build test indices.
    struct IndexBuilder {
        data: Vec<u8>,
    }

    impl IndexBuilder {
        fn new(num_docs: u32) -> Self {
            let mut data = vec![0u8; 20];
            // Magic "HYP0"
            data[0..4].copy_from_slice(&0x4859_5030u32.to_le_bytes());
            // Number of documents
            data[4..8].copy_from_slice(&num_docs.to_le_bytes());
            // Postings offset
            data[16..20].copy_from_slice(&20u32.to_le_bytes());
            Self { data }
        }

        fn add_bitmap_posting(&mut self, docs: &[u32]) -> usize {
            let offset = self.data.len() - 20; // Relative to postings_base

            // Type byte
            self.data.push(0);

            // Calculate bitmap size
            let max_doc = docs.iter().max().copied().unwrap_or(0);
            let needed_bytes = ((max_doc / 8) + 1) as usize;
            let len_bytes = (needed_bytes + 15) & !15;

            // Length field
            self.data
                .extend_from_slice(&(len_bytes as u32).to_le_bytes());

            // Bitmap data
            let start_idx = self.data.len();
            self.data.resize(self.data.len() + len_bytes, 0);

            for &doc in docs {
                let byte_pos = (doc / 8) as usize;
                let bit_pos = doc % 8;
                self.data[start_idx + byte_pos] |= 1 << bit_pos;
            }

            offset
        }

        fn build(self) -> Vec<u8> {
            self.data
        }
    }

    #[test]
    fn index_builder_creates_valid_index() {
        let mut builder = IndexBuilder::new(100);
        builder.add_bitmap_posting(&[10, 20, 30]);
        let data = builder.build();

        let index = HypersonicIndex::new(&data);
        assert!(index.is_some());
    }

    #[test]
    fn search_finds_documents() {
        let mut builder = IndexBuilder::new(100);
        builder.add_bitmap_posting(&[10, 20, 30]);
        let data = builder.build();

        let index = HypersonicIndex::new(&data).unwrap();

        let mut scratch = [0u32; SCRATCH_SIZE];
        let mut iter = PostingIterator::new(&index, 0, &mut scratch);

        assert_eq!(iter.advance(0, 0), Some(10));
        assert_eq!(iter.advance(11, 0), Some(20));
        assert_eq!(iter.advance(21, 0), Some(30));
        assert_eq!(iter.advance(31, 0), None);
    }

    #[test]
    fn empty_query_returns_zero() {
        let builder = IndexBuilder::new(100);
        let data = builder.build();
        let mut results = [0u32; 10];

        assert_eq!(search(&data, "", &mut results), 0);
    }

    #[test]
    fn invalid_index_returns_zero() {
        let mut results = [0u32; 10];

        assert_eq!(search(&[], "test", &mut results), 0);
        assert_eq!(search(&[0u8; 5], "test", &mut results), 0);
    }
}

#[cfg(test)]
#[cfg(target_arch = "wasm32")]
mod wasm_tests {
    extern crate wasm_bindgen_test;

    use super::*;
    use alloc::vec;
    use wasm_bindgen_test::*;

    wasm_bindgen_test_configure!(run_in_node_experimental);

    #[wasm_bindgen_test]
    fn wasm_search_basic() {
        // Build a minimal test index
        let mut data = vec![0u8; 48];
        data[0..4].copy_from_slice(&0x4859_5030u32.to_le_bytes()); // Magic
        data[16..20].copy_from_slice(&20u32.to_le_bytes()); // Postings offset

        // Add bitmap posting at offset 0
        data[20] = 0; // Type = bitmap
        data[21..25].copy_from_slice(&16u32.to_le_bytes()); // Length
        data[26] = 0b00000100; // Doc 10 (byte 1, bit 2)

        let mut scratch = [0u32; SCRATCH_SIZE];
        let index = HypersonicIndex::new(&data).unwrap();
        let mut iter = PostingIterator::new(&index, 0, &mut scratch);

        // Should find doc at position 10
        let result = iter.advance(0, 0);
        assert!(result.is_some());
    }
}
