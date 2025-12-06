//! High-level search engine logic.
//!
//! Provides the main search interface that orchestrates tokenization,
//! posting list iteration, and result collection.

use crate::data::HypersonicIndex;
use crate::iter::{PostingIterator, SCRATCH_SIZE};
use crate::tokenizer::{Tokenizer, MAX_TOKENS};

/// Maximum number of results to return from a search.
pub const MAX_RESULTS: usize = 128;

/// Performs a conjunctive (AND) search over the index.
///
/// Returns documents that match ALL query terms, using document-at-a-time
/// intersection with galloping.
///
/// # Arguments
/// * `index_data` - Raw index bytes
/// * `query` - Search query string
/// * `results` - Output buffer for matching document IDs
///
/// # Returns
/// Number of results written to the output buffer.
///
/// # Algorithm
/// 1. Tokenize query into term IDs
/// 2. Create posting list iterators for each term
/// 3. Perform intersection using document-at-a-time strategy
/// 4. Return matching document IDs
pub fn search(index_data: &[u8], query: &str, results: &mut [u32]) -> usize {
    // Parse index
    let Some(index) = HypersonicIndex::new(index_data) else {
        return 0;
    };

    // Tokenize query
    let tokenizer = Tokenizer::new();
    let mut tokens = [0u64; MAX_TOKENS];
    let token_count = tokenizer.tokenize(query, &mut tokens);

    if token_count == 0 {
        return 0;
    }

    // Allocate scratch buffers for decompression
    let mut scratch_buffers = [[0u32; SCRATCH_SIZE]; MAX_TOKENS];

    // Create iterators for each term
    let mut iterators: [PostingIterator<'_>; MAX_TOKENS] =
        core::array::from_fn(|_| PostingIterator::empty());

    let mut active_count = 0;
    let mut scratch_iter = scratch_buffers.iter_mut();

    for &token in tokens.iter().take(token_count) {
        if let Some(offset) = index.term_offset(token) {
            if let Some(scratch) = scratch_iter.next() {
                iterators[active_count] = PostingIterator::new(&index, offset, scratch);
                active_count += 1;
            }
        }
    }

    if active_count == 0 {
        return 0;
    }

    // Perform intersection
    intersect(&mut iterators[..active_count], results)
}

/// Intersects multiple posting lists using document-at-a-time strategy.
fn intersect(iterators: &mut [PostingIterator<'_>], results: &mut [u32]) -> usize {
    if iterators.is_empty() {
        return 0;
    }

    let max_results = results.len();
    let mut matches = 0;
    let mut target = 0u32;

    loop {
        // Find first document >= target in the lead iterator
        let Some(candidate) = iterators[0].advance(target, 0) else {
            break;
        };

        // Check if all other iterators contain this document
        let mut is_match = true;

        for iter in iterators.iter_mut().skip(1) {
            match iter.advance(candidate, 0) {
                Some(doc) if doc == candidate => {}
                Some(doc) => {
                    // This iterator jumped ahead - update target and retry
                    is_match = false;
                    target = doc;
                    break;
                }
                None => {
                    // Iterator exhausted - no more results
                    return matches;
                }
            }
        }

        if is_match {
            results[matches] = candidate;
            matches += 1;
            target = candidate.saturating_add(1);

            if matches >= max_results {
                break;
            }
        }
    }

    matches
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    fn make_test_index_with_bitmap(docs: &[u32]) -> Vec<u8> {
        let mut data = vec![0u8; 20];
        // Magic "HYP0"
        data[0..4].copy_from_slice(&0x4859_5030u32.to_le_bytes());
        // Postings offset at byte 20
        data[16..20].copy_from_slice(&20u32.to_le_bytes());

        // Bitmap posting list
        data.push(0); // Type = bitmap

        let max_doc = docs.iter().max().copied().unwrap_or(0);
        let needed_bytes = ((max_doc / 8) + 1) as usize;
        let len_bytes = (needed_bytes + 15) & !15;

        data.extend_from_slice(&(len_bytes as u32).to_le_bytes());

        let bitmap_start = data.len();
        data.resize(data.len() + len_bytes, 0);

        for &doc in docs {
            let byte_pos = (doc / 8) as usize;
            let bit_pos = doc % 8;
            data[bitmap_start + byte_pos] |= 1 << bit_pos;
        }

        data
    }

    #[test]
    fn empty_query() {
        let index = make_test_index_with_bitmap(&[1, 2, 3]);
        let mut results = [0u32; 10];
        assert_eq!(search(&index, "", &mut results), 0);
    }

    #[test]
    fn invalid_index() {
        let mut results = [0u32; 10];
        assert_eq!(search(&[], "test", &mut results), 0);
        assert_eq!(search(&[0u8; 10], "test", &mut results), 0);
    }

    #[test]
    fn intersect_empty() {
        let mut iters: [PostingIterator<'_>; 0] = [];
        let mut results = [0u32; 10];
        assert_eq!(intersect(&mut iters, &mut results), 0);
    }
}
