//! Posting list iterators for document retrieval.
//!
//! Provides efficient iterators over both bitmap and compressed
//! posting lists with SIMD acceleration where available.

// In WASM32, usize == u32, so truncation casts are safe
#![allow(clippy::cast_possible_truncation)]

use core::mem::size_of;

use crate::data::{BlockHeader, HypersonicIndex, PostingListType};
use crate::simd::{is_empty, scan_chunk, ChunkBits};
use crate::utils::ByteSliceExt;

/// Size of the decompression scratch buffer.
pub const SCRATCH_SIZE: usize = 128;

/// Iterator over a posting list, yielding document IDs matching search criteria.
///
/// Supports both bitmap and compressed posting list formats with
/// score-based early termination for top-k queries.
pub struct PostingIterator<'a> {
    strategy: Strategy<'a>,
    current_doc: u32,
    current_score: u8,
}

/// Internal strategy for different posting list formats.
enum Strategy<'a> {
    Empty,
    Bitmap(BitmapState<'a>),
    Compressed(CompressedState<'a>),
}

struct BitmapState<'a> {
    data: &'a [u8],
    chunk_idx: usize,
}

struct CompressedState<'a> {
    headers: &'a [BlockHeader],
    block_idx: usize,
    scratch: &'a mut [u32; SCRATCH_SIZE],
    buf_pos: usize,
}

impl<'a> PostingIterator<'a> {
    /// Creates an empty iterator that yields no documents.
    #[inline]
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            strategy: Strategy::Empty,
            current_doc: u32::MAX,
            current_score: 0,
        }
    }

    /// Creates an iterator for a posting list at the given offset.
    ///
    /// # Arguments
    /// * `index` - The index to read from
    /// * `term_offset` - Offset relative to `postings_base`
    /// * `scratch` - Scratch buffer for decompression
    ///
    /// # Returns
    /// An iterator positioned before the first document.
    pub fn new(
        index: &HypersonicIndex<'a>,
        term_offset: usize,
        scratch: &'a mut [u32; SCRATCH_SIZE],
    ) -> Self {
        let list_start = index.postings_base() + term_offset;
        let data = index.data();

        let Some(list_type_byte) = data.read_u8(list_start) else {
            return Self::empty();
        };

        let Ok(list_type) = PostingListType::try_from(list_type_byte) else {
            return Self::empty();
        };

        match list_type {
            PostingListType::Bitmap => Self::new_bitmap(data, list_start),
            PostingListType::Compressed => Self::new_compressed(index, list_start, scratch),
        }
    }

    fn new_bitmap(data: &'a [u8], list_start: usize) -> Self {
        let Some(len) = data.read_u32_le(list_start + 1) else {
            return Self::empty();
        };

        let Some(bitmap_data) = data.sub_slice(list_start + 5, len as usize) else {
            return Self::empty();
        };

        Self {
            strategy: Strategy::Bitmap(BitmapState {
                data: bitmap_data,
                chunk_idx: 0,
            }),
            current_doc: 0,
            current_score: 0,
        }
    }

    fn new_compressed(
        index: &HypersonicIndex<'a>,
        list_start: usize,
        scratch: &'a mut [u32; SCRATCH_SIZE],
    ) -> Self {
        let data = index.data();

        let Some(num_blocks) = data.read_u32_le(list_start + 1) else {
            return Self::empty();
        };

        let Some(start_offset) = data.read_u32_le(list_start + 5) else {
            return Self::empty();
        };

        let abs_headers_start = index.postings_base() + start_offset as usize;
        let headers_len = num_blocks as usize * size_of::<BlockHeader>();

        if abs_headers_start + headers_len > data.len() {
            return Self::empty();
        }

        // SAFETY: We validated bounds above.
        // Note: BlockHeader is repr(C) and packed appropriately.
        // In WASM32 this cast is safe as alignment is handled by the index format.
        #[allow(clippy::cast_ptr_alignment)]
        let headers = unsafe {
            core::slice::from_raw_parts(
                data.as_ptr().add(abs_headers_start).cast::<BlockHeader>(),
                num_blocks as usize,
            )
        };

        Self {
            strategy: Strategy::Compressed(CompressedState {
                headers,
                block_idx: 0,
                scratch,
                buf_pos: SCRATCH_SIZE,
            }),
            current_doc: 0,
            current_score: 0,
        }
    }

    /// Returns the current document ID.
    #[inline]
    #[must_use]
    pub const fn doc(&self) -> u32 {
        self.current_doc
    }

    /// Returns the score of the current document.
    #[inline]
    #[must_use]
    pub const fn score(&self) -> u8 {
        self.current_score
    }

    /// Advances to the first document >= `target` with score >= `min_score`.
    ///
    /// # Arguments
    /// * `target` - Minimum document ID to return
    /// * `min_score` - Minimum score threshold (for early termination)
    ///
    /// # Returns
    /// * `Some(doc_id)` - Found a matching document
    /// * `None` - No more matching documents
    #[inline]
    pub fn advance(&mut self, target: u32, min_score: u8) -> Option<u32> {
        match &mut self.strategy {
            Strategy::Empty => None,
            Strategy::Bitmap(state) => {
                let result = advance_bitmap(state, target);
                if let Some((doc, score)) = result {
                    self.current_doc = doc;
                    self.current_score = score;
                    Some(doc)
                } else {
                    None
                }
            }
            Strategy::Compressed(state) => {
                let result = advance_compressed(state, target, min_score);
                if let Some((doc, score)) = result {
                    self.current_doc = doc;
                    self.current_score = score;
                    Some(doc)
                } else {
                    None
                }
            }
        }
    }
}

/// Advances through a bitmap posting list.
fn advance_bitmap(state: &mut BitmapState<'_>, target: u32) -> Option<(u32, u8)> {
    let byte_idx = (target / 8) as usize;
    state.chunk_idx = byte_idx & !15; // Align to 16-byte boundary

    while state.chunk_idx < state.data.len() {
        // Try SIMD path for 16-byte chunks
        if let Some(chunk) = state.data.sub_slice(state.chunk_idx, 16) {
            if let Some(bits) = scan_chunk(chunk) {
                let base_doc = (state.chunk_idx as u32) * 8;

                // Skip chunk if all docs are below target
                if base_doc + 128 <= target {
                    state.chunk_idx += 16;
                    continue;
                }

                // All zeros - skip
                if is_empty(bits) {
                    state.chunk_idx += 16;
                    continue;
                }

                // Scan for matching document using CTZ optimization
                if let Some(doc) = find_set_bit_fast(bits, base_doc, target) {
                    state.chunk_idx = ((doc / 8) as usize) & !15;
                    return Some((doc, 10)); // Fixed score for bitmap entries
                }

                state.chunk_idx += 16;
                continue;
            }
        }

        // Fallback for partial chunks
        if let Some((doc, score)) = scan_partial_chunk(state, target) {
            return Some((doc, score));
        }

        break;
    }

    None
}

/// Finds the first set bit >= target using CTZ (count trailing zeros).
///
/// This is O(1) per found bit instead of O(128) scanning.
/// Works on the (lo, hi) pair to avoid slow u128 operations in WASM32.
#[inline]
fn find_set_bit_fast(bits: ChunkBits, base_doc: u32, target: u32) -> Option<u32> {
    let (mut lo, mut hi) = bits;
    let relative_target = target.saturating_sub(base_doc);

    // Process low 64 bits (docs 0-63 relative to chunk)
    if lo != 0 && relative_target < 64 {
        // Mask out bits below target
        let mask = if relative_target > 0 {
            !((1u64 << relative_target) - 1)
        } else {
            u64::MAX
        };
        lo &= mask;

        if lo != 0 {
            let bit_pos = lo.trailing_zeros();
            let doc = base_doc + bit_pos;
            return Some(doc);
        }
    }

    // Process high 64 bits (docs 64-127 relative to chunk)
    if hi != 0 {
        // If target is in high part, mask out bits below
        let high_relative = relative_target.saturating_sub(64);
        if high_relative < 64 {
            let mask = if high_relative > 0 {
                !((1u64 << high_relative) - 1)
            } else {
                u64::MAX
            };
            hi &= mask;
        }

        if hi != 0 {
            let bit_pos = hi.trailing_zeros();
            let doc = base_doc + 64 + bit_pos;
            return Some(doc);
        }
    }

    None
}

/// Scans remaining bytes when less than 16 bytes are available.
fn scan_partial_chunk(state: &mut BitmapState<'_>, target: u32) -> Option<(u32, u8)> {
    let remaining = state.data.len() - state.chunk_idx;

    for i in 0..remaining {
        let mut byte = state.data[state.chunk_idx + i];
        if byte == 0 {
            continue;
        }

        let doc_base = ((state.chunk_idx + i) as u32) * 8;

        // Skip entire byte if all its docs are below target
        if doc_base + 8 <= target {
            continue;
        }

        // Mask out bits below target within this byte
        let relative = target.saturating_sub(doc_base);
        if relative < 8 {
            byte &= !((1u8 << relative) - 1);
        }

        if byte != 0 {
            let bit_pos = byte.trailing_zeros();
            let doc = doc_base + bit_pos;
            return Some((doc, 10));
        }
    }

    None
}

/// Advances through a compressed posting list.
fn advance_compressed(
    state: &mut CompressedState<'_>,
    target: u32,
    min_score: u8,
) -> Option<(u32, u8)> {
    while state.block_idx < state.headers.len() {
        let header = &state.headers[state.block_idx];

        // Skip blocks that can't contain our target
        if header.max_score < min_score || header.max_doc < target {
            state.block_idx += 1;
            state.buf_pos = SCRATCH_SIZE;
            continue;
        }

        // Decompress block if needed
        if state.buf_pos >= SCRATCH_SIZE {
            decompress_block(state);
        }

        // Scan buffer for matching document
        while state.buf_pos < SCRATCH_SIZE {
            let doc = state.scratch[state.buf_pos];
            if doc >= target {
                return Some((doc, header.max_score));
            }
            state.buf_pos += 1;
        }

        state.block_idx += 1;
        state.buf_pos = SCRATCH_SIZE;
    }

    None
}

/// Decompresses a block into the scratch buffer.
fn decompress_block(state: &mut CompressedState<'_>) {
    let base = if state.block_idx == 0 {
        0
    } else {
        state.headers[state.block_idx - 1].max_doc
    };

    // Generate sequential doc IDs (simplified decompression)
    for (i, slot) in state.scratch.iter_mut().enumerate() {
        *slot = base.saturating_add(i as u32 + 1);
    }

    state.buf_pos = 0;
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    fn make_test_index(postings: &[u8]) -> Vec<u8> {
        let mut data = vec![0u8; 20];
        // Magic "HYP0"
        data[0..4].copy_from_slice(&0x4859_5030u32.to_le_bytes());
        // Postings offset
        data[16..20].copy_from_slice(&20u32.to_le_bytes());
        data.extend_from_slice(postings);
        data
    }

    fn make_bitmap_posting(docs: &[u32]) -> Vec<u8> {
        let max_doc = docs.iter().max().copied().unwrap_or(0);
        let needed_bytes = ((max_doc / 8) + 1) as usize;
        let len_bytes = (needed_bytes + 15) & !15;

        let mut result = vec![0u8]; // Type = 0 (bitmap)
        result.extend_from_slice(&(len_bytes as u32).to_le_bytes());

        let mut bitmap = vec![0u8; len_bytes];
        for &doc in docs {
            let byte_pos = (doc / 8) as usize;
            let bit_pos = doc % 8;
            bitmap[byte_pos] |= 1 << bit_pos;
        }

        result.extend_from_slice(&bitmap);
        result
    }

    #[test]
    fn empty_iterator() {
        let iter = PostingIterator::empty();
        assert_eq!(iter.doc(), u32::MAX);
        assert_eq!(iter.score(), 0);
    }

    #[test]
    fn bitmap_basic_advance() {
        let postings = make_bitmap_posting(&[10, 20, 30]);
        let index_data = make_test_index(&postings);
        let index = HypersonicIndex::new(&index_data).unwrap();

        let mut scratch = [0u32; SCRATCH_SIZE];
        let mut iter = PostingIterator::new(&index, 0, &mut scratch);

        assert_eq!(iter.advance(0, 0), Some(10));
        assert_eq!(iter.advance(11, 0), Some(20));
        assert_eq!(iter.advance(25, 0), Some(30));
        assert_eq!(iter.advance(31, 0), None);
    }

    #[test]
    fn bitmap_exact_target() {
        let postings = make_bitmap_posting(&[10, 20, 30]);
        let index_data = make_test_index(&postings);
        let index = HypersonicIndex::new(&index_data).unwrap();

        let mut scratch = [0u32; SCRATCH_SIZE];
        let mut iter = PostingIterator::new(&index, 0, &mut scratch);

        assert_eq!(iter.advance(20, 0), Some(20));
    }
}
