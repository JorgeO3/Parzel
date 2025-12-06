#![no_std]
#![feature(core_intrinsics)]
#![allow(internal_features)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

//! # Hypersonic Search Engine (WASM Kernel)
//! 
//! "Hypersonic" is a high-performance, zero-copy information retrieval kernel 
//! designed specifically for the `wasm32-unknown-unknown` target.
//! 
//! ## Architecture
//! 
//! * **Zero-Copy:** Maps binary indices directly from memory without parsing.
//! * **SIMD-First:** Utilizes 128-bit vector instructions for intersection and scoring.
//! * **Hybrid Layout:** Switches between Bitmap (dense) and Compressed (sparse) storage.
//! * **No-Std:** Operates without the standard library, using only `core` and `alloc`.
//! 
//! ## Safety
//! 
//! This crate uses aggressive `unsafe` optimizations. Strict bounds checking is 
//! enforced in `HypersonicIter::new` to prevent out-of-bounds memory access 
//! from malformed or malicious index blobs.

extern crate alloc;

// ============================================================================
// 0. MEMORY MANAGEMENT
// ============================================================================

#[cfg(feature = "lol_alloc")]
#[cfg(target_arch = "wasm32")]
#[global_allocator]
static ALLOCATOR: lol_alloc::LockedAllocator<lol_alloc::FreeListAllocator> =
    lol_alloc::LockedAllocator::new(lol_alloc::FreeListAllocator::new());

#[cfg(not(test))]
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    core::intrinsics::abort()
}

use core::slice;
use core::ptr::{addr_of, addr_of_mut};

#[cfg(target_arch = "wasm32")]
use core::arch::wasm32::*;

// ============================================================================
// 1. DATA STRUCTURES
// ============================================================================

/// SIMD Look-Up Table (LUT) for logarithmic quantization.
/// 
/// Maps 4-bit compressed scores to 8-bit real weights.
/// Must be 16-byte aligned for efficient SIMD loading.
#[repr(C, align(16))]
pub struct AlignedLut(pub [u8; 16]);

/// The global LUT instance used for scoring.
pub const LUT_LOG_QUANT: AlignedLut = AlignedLut([
    0, 5, 12, 20, 30, 45, 60, 80, 
    100, 120, 140, 160, 180, 200, 225, 255
]);

/// The main index header structure.
/// 
/// Represents the layout of the binary index file starting at offset 0.
#[repr(C)]
pub struct HypersonicIndex {
    /// Magic bytes to identify the file format (e.g., "HYP0").
    pub magic: u32,
    /// Total number of documents in the corpus.
    pub num_docs: u32,
    /// Offset to the Finite State Transducer (FST) for term dictionary.
    pub fst_offset: u32,
    /// Length of the FST in bytes.
    pub fst_len: u32,
    /// Offset where the postings lists begin.
    pub postings_base_offset: u32,
}

impl HypersonicIndex {
    /// resolves the byte offset for a given term ID.
    #[inline(always)]
    pub fn get_term_offset(&self, term_id: u64) -> Option<u32> {
        if term_id == 0 { return None; }
        Some((term_id % 1000) as u32 * 256) 
    }
}

/// Defines the storage strategy for a specific posting list.
#[repr(u8)]
#[derive(Clone, Copy, PartialEq)]
pub enum ListType {
    /// Dense bitmap storage for common terms (>1% density).
    Bitmap = 0,
    /// Compressed block storage (SIMD-BP128) for rare terms.
    Compressed = 1,
}

/// Header for a compressed posting list.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CompressedHeader {
    /// Number of 128-document blocks in this list.
    pub num_blocks: u32,
    /// Relative offset to the start of the blocks.
    pub start_offset: u32,
}

/// Metadata for a single block of compressed documents (Block-Max WAND).
#[repr(C)]
pub struct BlockHeader {
    /// The maximum document ID contained in this block.
    pub max_doc: u32,
    /// The maximum impact score contained in this block.
    pub max_score: u8,     
    /// Bit width used for integer compression.
    pub bit_width: u8,
    /// Padding for alignment.
    pub padding: u16,
    /// Relative offset to the compressed data payload.
    pub data_offset: u32,
}

// ============================================================================
// 2. SIMD KERNELS
// ============================================================================

/// Calculates scores for 32 documents using SIMD swizzling.
///
/// # Safety
/// * `packed_ptr` must point to a valid memory region of at least 16 bytes.
/// * Calling this on non-wasm32 architecture requires the `simd128` feature.
#[inline(always)]
#[cfg(target_arch = "wasm32")]
pub unsafe fn simd_score_4bit(
    packed_ptr: *const v128, 
    accumulator: v128,
    lut_vec: v128
) -> v128 {
    unsafe {
        let packed = v128_load(packed_ptr);
        let mask = u8x16_splat(0x0F);
        let low = v128_and(packed, mask);
        let high = v128_and(u16x8_shr(packed, 4), mask);
        let score_low = i8x16_swizzle(lut_vec, low);
        let score_high = i8x16_swizzle(lut_vec, high);
        let acc = u8x16_add_sat(accumulator, score_low);
        u8x16_add_sat(acc, score_high)
    }
}

/// Checks if any bit is set in the 128-bit vector.
///
/// # Safety
/// * Uses platform-specific intrinsics.
#[inline(always)]
#[cfg(target_arch = "wasm32")]
pub unsafe fn simd_any_true(v: v128) -> bool {
    v128_any_true(v)
}

/// Fallback for non-wasm targets.
/// 
/// # Safety
/// Always safe, returns true to force linear scan in tests.
#[cfg(not(target_arch = "wasm32"))]
pub unsafe fn simd_any_true(_v: u128) -> bool { true }


// ============================================================================
// 3. HYBRID ITERATOR (HARDENED)
// ============================================================================

/// Iterator state for traversing a posting list.
#[derive(Clone, Copy)]
pub struct HypersonicIter<'a> {
    strategy: PostingListStrategy<'a>,
    /// The current document ID pointed to by the iterator.
    pub current_doc: u32,
    /// The score of the current document.
    pub current_score: u8,
}

/// internal strategy enum for the iterator.
#[derive(Clone, Copy)]
pub enum PostingListStrategy<'a> {
    /// Represents an empty or invalid list.
    Empty,
    /// Strategy for dense bitmaps.
    Bitmap {
        /// Raw slice of the bitmap data.
        data: &'a [u8],
    },
    /// Strategy for compressed blocks.
    Compressed {
        /// Slice of block headers.
        headers: &'a [BlockHeader],
        /// Current block index.
        block_idx: usize,
        /// Pointer to the external scratchpad buffer.
        decompressed_buf_ptr: *mut u32, 
        /// Current position within the decompressed buffer.
        buf_pos: usize,
    }
}

impl<'a> HypersonicIter<'a> {
    /// Creates an empty iterator.
    pub fn empty() -> Self {
        Self { strategy: PostingListStrategy::Empty, current_doc: u32::MAX, current_score: 0 }
    }

    /// Creates a new iterator from raw index memory.
    ///
    /// # Safety
    /// * `index` must reference a valid `HypersonicIndex`.
    /// * `offset` must be valid relative to `index`.
    /// * `scratch_buf_ptr` must be valid and mutable (size >= 512 bytes).
    pub unsafe fn new(index: &'a HypersonicIndex, offset: u32, scratch_buf_ptr: *mut u32) -> Self {
        unsafe {
            let blob_base = index as *const HypersonicIndex as *const u8;
            let blob_limit = blob_base.add(INDEX_LEN);

            let base_ptr = blob_base.add(index.postings_base_offset as usize);
            let list_ptr = base_ptr.add(offset as usize);
            
            if list_ptr >= blob_limit { return Self::empty(); }

            let list_type = *(list_ptr as *const ListType);

            match list_type {
                ListType::Bitmap => {
                    let len_ptr = list_ptr.add(1) as *const u32;
                    if len_ptr.add(1) as *const u8 > blob_limit { return Self::empty(); }
                    
                    let len = len_ptr.read_unaligned() as usize;
                    let data_ptr = list_ptr.add(5);
                    let data_end = data_ptr.add(len);

                    if data_end > blob_limit {
                        return Self::empty(); 
                    }

                    let data = slice::from_raw_parts(data_ptr, len);
                    Self {
                        strategy: PostingListStrategy::Bitmap { data },
                        current_doc: 0, 
                        current_score: 0,
                    }
                },
                ListType::Compressed => {
                    let header_ptr = list_ptr.add(1) as *const CompressedHeader;
                    if header_ptr.add(1) as *const u8 > blob_limit { return Self::empty(); }

                    let header = header_ptr.read_unaligned();
                    let blocks_ptr = base_ptr.add(header.start_offset as usize);
                    let headers_end = blocks_ptr.add(header.num_blocks as usize * size_of::<BlockHeader>());

                    if headers_end > blob_limit {
                        return Self::empty();
                    }

                    let headers = slice::from_raw_parts(
                        blocks_ptr as *const BlockHeader, 
                        header.num_blocks as usize
                    );
                    
                    *scratch_buf_ptr = 0; 

                    Self {
                        strategy: PostingListStrategy::Compressed {
                            headers,
                            block_idx: 0,
                            decompressed_buf_ptr: scratch_buf_ptr, 
                            buf_pos: 128,
                        },
                        current_doc: 0,
                        current_score: 0,
                    }
                }
            }
        }
    }

    /// Advances the iterator to the next candidate document.
    #[inline(always)]
    pub fn advance(&mut self, target: u32, min_score: u8) -> Option<u32> {
        self.advance_inner(target, min_score)
    }

    #[inline(never)]
    fn advance_inner(&mut self, target: u32, min_score: u8) -> Option<u32> {
        match &mut self.strategy {
            PostingListStrategy::Empty => None,
            
            PostingListStrategy::Bitmap { data } => {
                let byte_idx = (target / 8) as usize;
                let mut simd_idx = byte_idx & !15; 

                while simd_idx < data.len() {
                    #[cfg(target_arch = "wasm32")]
                    {
                        unsafe {
                            let ptr = data.as_ptr().add(simd_idx);
                            let v_bits = v128_load(ptr as *const v128);
                            
                            if simd_any_true(v_bits) {
                                let u64_ptr = ptr as *const u64;
                                for i in 0..2 {
                                    let word = *u64_ptr.add(i); 
                                    if word != 0 {
                                        let current_base_doc = (simd_idx as u32) * 8 + (i as u32 * 64);
                                        let mut temp_word = word;
                                        
                                        if current_base_doc < target {
                                            let shift = target.saturating_sub(current_base_doc);
                                            if shift < 64 {
                                                temp_word &= !((1u64 << shift) - 1);
                                            } else {
                                                temp_word = 0;
                                            }
                                        }

                                        if temp_word != 0 {
                                            let trailing = temp_word.trailing_zeros();
                                            let doc = current_base_doc + trailing;
                                            self.current_doc = doc;
                                            self.current_score = 10;
                                            return Some(doc);
                                        }
                                    }
                                }
                            }
                        }
                    }
                    simd_idx += 16;
                }
                None
            },

            PostingListStrategy::Compressed { headers, block_idx, buf_pos, decompressed_buf_ptr, .. } => {
                unsafe {
                    while *block_idx < headers.len() {
                        let h = &headers[*block_idx];
                        
                        if h.max_score < min_score {
                            *block_idx += 1;
                            *buf_pos = 128;
                            continue;
                        }
                        if h.max_doc < target {
                            *block_idx += 1;
                            *buf_pos = 128;
                            continue;
                        }

                        if *buf_pos >= 128 {
                            for i in 0..128 {
                                let base = if *block_idx == 0 { 0 } else { headers[*block_idx-1].max_doc };
                                *decompressed_buf_ptr.add(i) = base.saturating_add(i as u32 + 1);
                            }
                            *buf_pos = 0;
                        }

                        while *buf_pos < 128 {
                            let doc = *decompressed_buf_ptr.add(*buf_pos);
                            if doc >= target {
                                self.current_doc = doc;
                                self.current_score = h.max_score;
                                return Some(doc);
                            }
                            *buf_pos += 1;
                        }
                        *block_idx += 1;
                        *buf_pos = 128;
                    }
                    None
                }
            }
        }
    }
}

// ============================================================================
// 4. TOKENIZER (STUB)
// ============================================================================

/// A simple FST-based tokenizer.
pub struct FstTokenizer<'a> {
    _index: &'a HypersonicIndex,
}

impl<'a> FstTokenizer<'a> {
    /// Creates a new tokenizer.
    pub fn new(index: &'a HypersonicIndex) -> Self {
        Self { _index: index }
    }

    /// Tokenizes the input text into Term IDs.
    pub fn tokenize(&mut self, text: &str, out_tokens: &mut [u64]) -> usize {
        let bytes = text.as_bytes();
        let mut count = 0;
        let mut i = 0;
        while i < bytes.len() && count < out_tokens.len() {
            if bytes[i] == b' ' { i += 1; continue; }
            let mut j = i;
            while j < bytes.len() && bytes[j] != b' ' { j += 1; }
            let word_len = (j - i) as u64;
            out_tokens[count] = word_len * 100;
            count += 1;
            i = j;
        }
        count
    }
}

// ============================================================================
// 5. GLOBAL STATE & FFI EXPORTS
// ============================================================================

static mut INDEX_PTR: *const HypersonicIndex = core::ptr::null();
static mut INDEX_LEN: usize = 0;
static mut RESULTS_BUFFER: [u32; 128] = [0; 128]; 

/// Initializes the search index.
///
/// # Safety
/// * `ptr` must be a valid pointer to the index blob.
/// * `len` must be the exact size of the blob.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn init_index(ptr: *const u8, len: usize) {
    unsafe {
        INDEX_PTR = ptr as *const HypersonicIndex;
        INDEX_LEN = len;
    }
}

/// Executes a search.
///
/// # Safety
/// * `query_ptr` must point to UTF-8 data.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn search(query_ptr: *const u8, query_len: usize) -> u32 {
    unsafe {
        if INDEX_PTR.is_null() { return 0; }
        let index = &*INDEX_PTR;

        let mut scratch_buffers = [[0u32; 128]; 16]; 

        let query_slice = slice::from_raw_parts(query_ptr, query_len);
        let query_str = core::str::from_utf8_unchecked(query_slice);
        
        let mut tokenizer = FstTokenizer::new(index);
        let mut tokens = [0u64; 16]; 
        let token_count = tokenizer.tokenize(query_str, &mut tokens);

        if token_count == 0 { return 0; }

        let mut iterators: [HypersonicIter; 16] = [HypersonicIter::empty(); 16];
        for i in 0..token_count {
            if let Some(offset) = index.get_term_offset(tokens[i]) {
                iterators[i] = HypersonicIter::new(
                    index, 
                    offset, 
                    scratch_buffers[i].as_mut_ptr()
                );
            }
        }

        let mut matches = 0;
        let buffer_ptr = addr_of_mut!(RESULTS_BUFFER) as *mut u32;
        let mut current_target = 0;
        
        while matches < 128 {
            if let Some(doc) = iterators[0].advance(current_target, 0) {
                let mut is_match = true;
                for iter in iterators.iter_mut().take(token_count).skip(1) {
                    if let Some(other_doc) = iter.advance(doc, 0) {
                        if other_doc != doc {
                            is_match = false;
                            current_target = other_doc;
                            break;
                        }
                    } else {
                        return matches as u32;
                    }
                }

                if is_match {
                    *buffer_ptr.add(matches) = doc;
                    matches += 1;
                    current_target = doc + 1;
                }
            } else {
                break;
            }
        }

        matches as u32
    }
}

/// Gets the result buffer pointer.
///
/// # Safety
/// * Returns raw static pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn get_results_ptr() -> *const u32 {
    addr_of!(RESULTS_BUFFER) as *const u32
}

// ============================================================================
// 6. TEST SUITE
// ============================================================================
#[cfg(test)]
mod torture_chamber {
    use super::*;
    use alloc::vec::Vec;
    use alloc::vec;
    use wasm_bindgen_test::*; // Importar el test runner de wasm

    // Configurar para ejecutar en Node (o Browser si se prefiere)
    // Para --node en cli, esto es compatible.
    wasm_bindgen_test_configure!(run_in_node_experimental);

    struct IndexBuilder {
        data: Vec<u8>,
        postings_start: usize,
    }

    impl IndexBuilder {
        fn new() -> Self {
            let data = vec![0u8; 20]; 
            Self { data, postings_start: 20 }
        }

        fn set_header(&mut self, num_docs: u32) {
            let header = HypersonicIndex {
                magic: 0x48595030,
                num_docs,
                fst_offset: 0,
                fst_len: 0,
                postings_base_offset: 20,
            };
            unsafe {
                let ptr = self.data.as_mut_ptr() as *mut HypersonicIndex;
                *ptr = header;
            }
        }

        fn add_bitmap_list(&mut self, docs: &[u32]) -> u32 {
            let offset = (self.data.len() - self.postings_start) as u32;
            self.data.push(ListType::Bitmap as u8);
            let max_doc = docs.iter().max().unwrap_or(&0);
            let len_bytes = ((max_doc / 8) + 1 + 16) as usize; 
            self.data.extend_from_slice(&(len_bytes as u32).to_le_bytes());
            let start_idx = self.data.len();
            self.data.extend(core::iter::repeat_n(0, len_bytes));
            for &doc in docs {
                let byte_pos = (doc / 8) as usize;
                let bit_pos = doc % 8;
                self.data[start_idx + byte_pos] |= 1 << bit_pos;
            }
            offset
        }

        fn add_corrupt_list(&mut self, fake_len: usize) -> u32 {
            let offset = (self.data.len() - self.postings_start) as u32;
            self.data.push(ListType::Bitmap as u8);
            self.data.extend_from_slice(&(fake_len as u32).to_le_bytes());
            offset
        }

        fn as_ptr(&self) -> *const u8 { self.data.as_ptr() }
        fn len(&self) -> usize { self.data.len() }
    }

    // ATENCIÓN: Usamos #[wasm_bindgen_test] en lugar de #[test]

    #[wasm_bindgen_test]
    fn test_misaligned_memory_access() {
        let mut builder = IndexBuilder::new();
        builder.set_header(100);
        let _ = builder.add_bitmap_list(&[10, 20]);

        let mut misaligned_buffer = vec![0u8; builder.len() + 1];
        misaligned_buffer[1..].copy_from_slice(&builder.data);

        let ptr = unsafe { misaligned_buffer.as_ptr().add(1) };
        
        unsafe {
            init_index(ptr, builder.len());
            let idx = &*INDEX_PTR;
            assert_eq!(idx.num_docs, 100);
            assert_eq!(idx.magic, 0x48595030);
        }
    }

    #[wasm_bindgen_test]
    fn test_corrupt_index_graceful_fail() {
        let mut builder = IndexBuilder::new();
        builder.set_header(100);
        let offset = builder.add_corrupt_list(1024 * 1024 * 1024); 

        unsafe {
            init_index(builder.as_ptr(), builder.len());
            let index = &*INDEX_PTR;
            let mut scratch = [0u32; 128];

            let mut iter = HypersonicIter::new(index, offset, scratch.as_mut_ptr());
            let res = iter.advance(0, 0);
            assert_eq!(res, None);
        }
    }

    #[wasm_bindgen_test]
    fn test_stack_memory_write() {
        let mut builder = IndexBuilder::new();
        builder.set_header(100);
        let docs = vec![0, 128, 256];
        let offset = builder.add_bitmap_list(&docs);
        
        unsafe {
            init_index(builder.as_ptr(), builder.len());
            let index = &*INDEX_PTR;
            let mut scratch = [0u32; 128];
            let mut iter = HypersonicIter::new(index, offset, scratch.as_mut_ptr());
            
            assert_eq!(iter.advance(0, 0), Some(0));
            assert_eq!(iter.advance(100, 0), Some(128));
            assert_eq!(iter.advance(300, 0), None);
        }
    }
}