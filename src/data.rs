//! Index data structures and parsing logic.
//!
//! Defines the binary format for the Hypersonic index and provides
//! safe parsing routines with proper validation.

use crate::utils::ByteSliceExt;

/// Magic number identifying a valid Hypersonic index: "HYP0" in little-endian.
const MAGIC: u32 = 0x4859_5030;

/// Minimum header size required for a valid index.
const MIN_HEADER_SIZE: usize = 20;

/// Parsed view into an immutable Hypersonic index.
///
/// This struct provides zero-copy access to index data, validating
/// the format on construction.
#[derive(Clone, Copy, Debug)]
pub struct HypersonicIndex<'a> {
    data: &'a [u8],
    postings_base: usize,
}

impl<'a> HypersonicIndex<'a> {
    /// Creates a new index view from raw bytes.
    ///
    /// # Arguments
    /// * `data` - Raw index bytes with header and postings
    ///
    /// # Returns
    /// * `Some(index)` - Valid index structure
    /// * `None` - Invalid magic, insufficient size, or corrupt offset
    #[must_use]
    pub fn new(data: &'a [u8]) -> Option<Self> {
        if data.len() < MIN_HEADER_SIZE {
            return None;
        }

        let magic = data.read_u32_le(0)?;
        if magic != MAGIC {
            return None;
        }

        let postings_offset = data.read_u32_le(16)? as usize;
        if postings_offset > data.len() {
            return None;
        }

        Some(Self {
            data,
            postings_base: postings_offset,
        })
    }

    /// Returns the raw index data.
    #[inline]
    #[must_use]
    pub fn data(&self) -> &'a [u8] {
        self.data
    }

    /// Returns the base offset for postings lists.
    #[inline]
    #[must_use]
    pub fn postings_base(&self) -> usize {
        self.postings_base
    }

    /// Computes the posting list offset for a given term ID.
    ///
    /// # Returns
    /// * `Some(offset)` - Relative offset from `postings_base`
    /// * `None` - Invalid term ID (e.g., zero)
    #[inline]
    #[must_use]
    pub fn term_offset(&self, term_id: u64) -> Option<usize> {
        if term_id == 0 {
            return None;
        }
        // Simple hash-based offset calculation
        Some((term_id % 1000) as usize * 256)
    }
}

/// Header for a compressed postings list.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
#[allow(dead_code)] // Used by FFI consumers
pub struct CompressedHeader {
    /// Number of blocks in the posting list.
    pub num_blocks: u32,
    /// Offset to the block headers.
    pub start_offset: u32,
}

/// Header for a single compressed block within a postings list.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct BlockHeader {
    /// Maximum document ID in this block.
    pub max_doc: u32,
    /// Maximum score in this block (for early termination).
    pub max_score: u8,
    /// Bit width for delta encoding.
    pub bit_width: u8,
    /// Padding for alignment.
    pub _padding: u16,
    /// Offset to compressed data from block headers start.
    pub data_offset: u32,
}

/// Posting list type discriminant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum PostingListType {
    Bitmap = 0,
    Compressed = 1,
}

impl TryFrom<u8> for PostingListType {
    type Error = ();

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Bitmap),
            1 => Ok(Self::Compressed),
            _ => Err(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn make_valid_header() -> alloc::vec::Vec<u8> {
        let mut data = vec![0u8; 20];
        // Magic "HYP0"
        data[0..4].copy_from_slice(&MAGIC.to_le_bytes());
        // Postings offset at byte 16
        data[16..20].copy_from_slice(&20u32.to_le_bytes());
        data
    }

    #[test]
    fn valid_index() {
        let data = make_valid_header();
        let index = HypersonicIndex::new(&data).unwrap();
        assert_eq!(index.postings_base(), 20);
    }

    #[test]
    fn invalid_magic() {
        let mut data = make_valid_header();
        data[0] = 0xFF;
        assert!(HypersonicIndex::new(&data).is_none());
    }

    #[test]
    fn too_small() {
        let data = [0u8; 19];
        assert!(HypersonicIndex::new(&data).is_none());
    }

    #[test]
    fn invalid_offset() {
        let mut data = make_valid_header();
        // Offset beyond data length
        data[16..20].copy_from_slice(&100u32.to_le_bytes());
        assert!(HypersonicIndex::new(&data).is_none());
    }

    #[test]
    fn term_offset_zero() {
        let data = make_valid_header();
        let index = HypersonicIndex::new(&data).unwrap();
        assert!(index.term_offset(0).is_none());
    }

    #[test]
    fn term_offset_valid() {
        let data = make_valid_header();
        let index = HypersonicIndex::new(&data).unwrap();
        assert_eq!(index.term_offset(1), Some(256));
        assert_eq!(index.term_offset(1001), Some(256)); // 1001 % 1000 = 1
    }

    #[test]
    fn posting_list_type_conversion() {
        assert_eq!(PostingListType::try_from(0), Ok(PostingListType::Bitmap));
        assert_eq!(PostingListType::try_from(1), Ok(PostingListType::Compressed));
        assert!(PostingListType::try_from(2).is_err());
    }
}
