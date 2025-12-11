//! Index data structures and parsing logic.
//!
//! Defines the binary format for the Hypersonic index and provides
//! safe parsing routines with proper validation.

use crate::utils::ByteSliceExt;
use fst::Map; // Importamos el Mapa FST

/// Magic number identifying a valid Hypersonic index: "HYP0" in little-endian.
const MAGIC: u32 = 0x4859_5030;

/// Minimum header size required for a valid index.
/// 20 bytes original header + FST data overhead (variable but checked).
const MIN_HEADER_SIZE: usize = 28; // Updated to account for FST offset/len fields

/// Parsed view into an immutable Hypersonic index.
///
/// This struct provides zero-copy access to index data, validating
/// the format on construction.
#[derive(Clone, Debug)]
pub struct HypersonicIndex<'a> {
    data: &'a [u8],
    postings_base: usize,
    term_map: Map<&'a [u8]>,
    doc_norms: &'a [u8],
    avg_field_len: f32,
    num_docs: u32,
}

impl<'a> HypersonicIndex<'a> {
    /// Creates a new index view from raw bytes.
    ///
    /// # Arguments
    /// * `data` - Raw index bytes with header, FST, and postings.
    ///
    /// # Returns
    /// * `Some(index)` - Valid index structure.
    /// * `None` - Invalid magic, insufficient size, corrupt FST, or bad offsets.
    #[must_use]
    pub fn new(data: &'a [u8]) -> Option<Self> {
        if data.len() < MIN_HEADER_SIZE {
            return None;
        }

        // 1. Validar Magic (0-4)
        let magic = data.read_u32_le(0)?;
        if magic != 0x4859_5030 {
            return None;
        }

        // 2. Leer Metadata
        // [8..12] num_docs
        let num_docs = data.read_u32_le(8)?;
        // [12..16] avg_field_len (f32)
        let avg_field_len_bytes = data.sub_slice(12, 4)?.try_into().ok()?;
        let avg_field_len = f32::from_le_bytes(avg_field_len_bytes);
        // [16..20] Postings Base Offset
        let postings_offset = data.read_u32_le(16)? as usize;
        // [20..24] FST Size (bytes)
        let fst_size = data.read_u32_le(20)? as usize;
        // [24..28] Doc Norms Offset
        let norms_offset = data.read_u32_le(24)? as usize;

        // 3. Bounds Check y Slicing (Zero-Copy)
        let fst_start = MIN_HEADER_SIZE;
        let fst_end = fst_start + fst_size;

        // CRÍTICO: El array de normas debe existir y tener el tamaño correcto (1 byte por doc).
        let norms_end = norms_offset.checked_add(num_docs as usize)?;

        // Verificación de límites.
        if fst_end > data.len() || norms_end > data.len() {
            return None;
        }

        // 4. Inicializar FST Map
        let fst_bytes = &data[fst_start..fst_end];
        let term_map = Map::new(fst_bytes).ok()?;

        // 5. Slice para Doc Norms
        let doc_norms = &data[norms_offset..norms_end];

        Some(Self {
            data,
            postings_base: postings_offset,
            term_map,
            doc_norms,
            avg_field_len,
            num_docs,
        })
    }

    /// Returns the raw index data.
    #[inline]
    pub const fn data(&self) -> &'a [u8] {
        self.data
    }

    /// Returns the base offset for postings lists.
    #[inline]
    pub const fn postings_base(&self) -> usize {
        self.postings_base
    }

    /// Computes the posting list offset for a given term.
    ///
    /// Uses the internal FST to map the string term to a relative offset.
    ///
    /// # Arguments
    /// * `term` - The search term (e.g., "apple").
    ///
    /// # Returns
    /// * `Some(offset)` - Relative offset from `postings_base` if the term exists.
    /// * `None` - Term not found in the index.
    #[inline]
    pub fn get_term_offset(&self, term: &str) -> Option<usize> {
        self.term_map.get(term).and_then(|val| val.try_into().ok())
    }

    /// Returns the document norm for a given document ID.
    ///
    /// # Arguments
    /// * `doc_id` - The document identifier.
    ///
    /// # Returns
    /// * `Some(norm)` - The normalization factor for the document if it exists.
    /// * `None` - Document ID out of bounds.
    #[inline]
    pub fn get_doc_norm(&self, doc_id: u32) -> Option<u8> {
        self.doc_norms.get(doc_id as usize).copied()
    }

    /// Returns the average field length across all documents.
    #[inline]
    pub const fn avg_field_len(&self) -> f32 {
        self.avg_field_len
    }

    /// Returns the total number of documents in the index.
    #[inline]
    pub const fn num_docs(&self) -> u32 {
        self.num_docs
    }

    /// Busca un término y devuelve sus metadatos completos.
    /// Asumimos que el FST devuelve un puntero a una estructura `[doc_freq(4B) | offset(4B)]`
    /// ubicada en una sección dedicada de "Term Metadata" en el binario.
    #[inline]
    pub fn get_term_info(&self, term: &str) -> Option<TermInfo> {
        let meta_offset = self.term_map.get(term)? as usize;

        // Leemos 8 bytes desde la posición indicada por el FST
        // (Nota: Esto requiere que tu IndexBuilder escriba los datos así)
        let doc_freq = self.data.read_u32_le(meta_offset)?;
        let relative_offset = self.data.read_u32_le(meta_offset + 4)? as usize;

        Some(TermInfo {
            doc_freq,
            postings_offset: self.postings_base + relative_offset,
        })
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

#[derive(Debug, Clone, Copy)]
pub struct TermInfo {
    pub doc_freq: u32, // NECESARIO para calcular IDF
    pub postings_offset: usize,
}

/// Header for a single compressed block within a postings list.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct BlockHeader {
    pub max_doc: u32,
    pub max_tf: u16,
    pub bit_width_doc: u8,
    pub bit_width_tf: u8,
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
    use fst::MapBuilder;

    // Helper to create a valid index with a real FST
    fn make_valid_index_with_fst() -> Vec<u8> {
        let mut data = vec![0u8; 24]; // Header size is now 24

        // 1. Header Fields
        data[0..4].copy_from_slice(&MAGIC.to_le_bytes());
        // Postings offset will be after the FST. We'll set it dynamically.

        // 2. Build a tiny FST
        let mut build = MapBuilder::memory();
        build.insert("foo", 100).unwrap();
        build.insert("test", 200).unwrap();
        let fst_bytes = build.into_inner().unwrap();
        let fst_len = fst_bytes.len();

        // 3. Write FST Size at offset 20
        data[20..24].copy_from_slice(&(fst_len as u32).to_le_bytes());

        // 4. Append FST
        data.extend_from_slice(&fst_bytes);

        // 5. Calculate and Write Postings Offset
        // Postings start right after FST
        let postings_start = 24 + fst_len;
        data[16..20].copy_from_slice(&(postings_start as u32).to_le_bytes());

        data
    }

    #[test]
    fn valid_index_construction() {
        let data = make_valid_index_with_fst();
        let index = HypersonicIndex::new(&data);
        assert!(index.is_some(), "Should accept valid buffer with FST");

        let idx = index.unwrap();
        // Check lookup
        assert_eq!(idx.get_term_offset("foo"), Some(100));
        assert_eq!(idx.get_term_offset("test"), Some(200));
        assert_eq!(idx.get_term_offset("bar"), None);
    }

    #[test]
    fn invalid_magic() {
        let mut data = make_valid_index_with_fst();
        data[0] = 0xFF; // Corrupt magic
        assert!(HypersonicIndex::new(&data).is_none());
    }

    #[test]
    fn invalid_fst_data() {
        let mut data = make_valid_index_with_fst();
        // Corrupt the FST bytes (start at 24)
        if data.len() > 24 {
            data[24] = 0xFF;
            data[25] = 0xFF;
        }
        // Map::new should fail validation
        assert!(HypersonicIndex::new(&data).is_none());
    }

    #[test]
    fn posting_list_type_conversion() {
        assert_eq!(PostingListType::try_from(0), Ok(PostingListType::Bitmap));
        assert_eq!(
            PostingListType::try_from(1),
            Ok(PostingListType::Compressed)
        );
        assert!(PostingListType::try_from(2).is_err());
    }
}
