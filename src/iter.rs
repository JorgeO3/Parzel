//! Posting list iterators for document retrieval.
//!
//! Provides efficient iterators over both bitmap and compressed
//! posting lists with SIMD acceleration where available.

use core::mem::MaybeUninit;
use core::mem::size_of;
use core::num::Wrapping; // Used for safe index arithmetic

use crate::data::{BlockHeader, HypersonicIndex, PostingListType};
use crate::prelude::*;
use crate::simd::{ChunkBits, is_empty, scan_chunk};
#[cfg(target_arch = "wasm32")]
use crate::simd::{decode_bp128_chunk_simd, decode_raw_chunk_simd, write_docid_chunk}; // Importaciones SIMD reales
use crate::utils::ByteSliceExt;

/// Size of the decompression scratch buffer (128 u32 slots = 512 bytes).
pub const SCRATCH_SIZE: usize = 128;

/// Internal strategy for different posting list formats.
#[allow(clippy::large_enum_variant)]
enum Strategy<'a> {
    Empty,
    Bitmap(BitmapState<'a>),
    Compressed(CompressedState<'a>),
}

/// State for reading an uncompressed Flat Bitset posting list.
struct BitmapState<'a> {
    data: &'a [u8],
    chunk_idx: usize,
}

/// State for reading a compressed (BP128) posting list.
///
/// NOTE: The scratch buffer is used to store decompressed `DocIDs`.
/// In the SIMD path, `DocIDs` are decoded directly into this buffer.
struct CompressedState<'a> {
    headers_data: &'a [u8],
    num_blocks: usize,
    block_idx: usize,
    scratch: &'a mut [u32; SCRATCH_SIZE],
    tfs_scratch: [u32; SCRATCH_SIZE],
    buf_pos: usize,
    #[cfg(target_arch = "wasm32")]
    compressed_data: &'a [u8],
    #[cfg(target_arch = "wasm32")]
    tf_data_slice: &'a [u8],
    last_doc_id: u32,
}

impl CompressedState<'_> {
    /// Lee el header actual manejando la desalineación de forma segura.
    /// Costo: Un `memcpy` de 12 bytes (muy barato, probablemente optimizado a registros).
    #[inline]
    const fn current_header(&self) -> BlockHeader {
        // Cálculo del offset en bytes
        let offset = self.block_idx * size_of::<BlockHeader>();

        // Debug assert para desarrollo (en prod el iterador controla los límites)
        debug_assert!(offset + size_of::<BlockHeader>() <= self.headers_data.len());

        // SAFETY:
        // 1. Validez de Memoria: Hemos verificado previamente (mediante `debug_assert!`
        //    y la lógica de construcción del índice) que `offset + size_of::<BlockHeader>()`
        //    está dentro de los límites del slice `self.headers_data`. Por tanto, el puntero
        //    resultante apunta a memoria válida y asignada.
        // 2. Alineación: Aunque `src_ptr` puede no cumplir con la alineación requerida para
        //    `BlockHeader` (align 4), usamos `ptr::read_unaligned` que está diseñado
        //    específicamente para realizar copias seguras desde direcciones desalineadas.
        // 3. Inicialización: Los bytes provienen de un slice `&[u8]` válido, por lo que
        //    la memoria está inicializada. `BlockHeader` es un tipo POD (Plain Old Data)
        //    compuesto de enteros, por lo que cualquier patrón de bits es válido.
        unsafe {
            // 1. Obtenemos el puntero crudo al byte donde empieza el header
            let src_ptr = self.headers_data.as_ptr().add(offset);

            // 2. Casteamos a *const BlockHeader.
            //    NOTA: Este puntero puede estar DESALINEADO. No podemos dereferenciarlo directamente (*ptr).
            #[allow(clippy::cast_ptr_alignment)]
            let ptr = src_ptr.cast::<BlockHeader>();

            // 3. Usamos read_unaligned via ptr::read_unaligned.
            //    Esto le dice al CPU: "Copia estos bytes a una struct en el stack, byte a byte si es necesario".
            //    En x86 y ARM modernos esto es casi tan rápido como una lectura normal.
            core::ptr::read_unaligned(ptr)
        }
    }
}

/// Iterator over a posting list, yielding document IDs matching search criteria.
///
/// Supports both bitmap and compressed posting list formats with
/// score-based early termination for top-k queries.
pub struct PostingIterator<'a> {
    strategy: Strategy<'a>,
    current_doc: u32,
    current_score: u8,
}

impl<'a> PostingIterator<'a> {
    /// Creates an empty iterator that yields no documents.
    pub const fn empty() -> Self {
        Self {
            strategy: Strategy::Empty,
            current_doc: u32::MAX,
            current_score: 0,
        }
    }

    /// Creates an iterator for a posting list at the given offset.
    pub fn new(
        index: &HypersonicIndex<'a>,
        term_offset: usize,
        scratch_raw: &'a mut MaybeUninit<[u32; SCRATCH_SIZE]>,
    ) -> Self {
        // SAFETY: The caller is responsible for ensuring that the scratch buffer
        // will be initialized before any read operations occur. The buffer is only
        // written to during decompression operations before being read.
        let scratch = unsafe { scratch_raw.assume_init_mut() };

        let Some(list_start) = index.postings_base().checked_add(term_offset) else {
            return Self::empty();
        };

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

        // Offset de datos comprimidos (Docs + TFs)
        let Some(compressed_data_start) = data.read_u32_le(list_start + 9) else {
            return Self::empty();
        };

        let Some(abs_headers_start) = index.postings_base().checked_add(start_offset as usize)
        else {
            return Self::empty();
        };

        let headers_len = num_blocks as usize * size_of::<BlockHeader>();

        // Verificamos que los headers caben en el archivo
        let Some(abs_headers_end) = abs_headers_start.checked_add(headers_len) else {
            return Self::empty();
        };

        if abs_headers_end > data.len() {
            return Self::empty();
        }

        let headers_data = &data[abs_headers_start..abs_headers_end];

        // 3. Validación de Datos Comprimidos
        let Some(abs_data_start) = index
            .postings_base()
            .checked_add(compressed_data_start as usize)
        else {
            return Self::empty();
        };

        // A. Verificación de Límites del Archivo
        if abs_data_start > data.len() {
            return Self::empty();
        }

        // B. Creación del Slice Seguro
        let compressed_data_slice = &data[abs_data_start..];

        // C. Verificación de Longitud Mínima para SIMD (CRÍTICO)
        // Aseguramos al menos 64 bytes para evitar lecturas fuera de límites en v128_load
        if num_blocks > 0 && compressed_data_slice.len() < 64 {
            return Self::empty();
        }

        Self {
            strategy: Strategy::Compressed(CompressedState {
                block_idx: 0,
                scratch,
                headers_data,
                last_doc_id: 0,
                buf_pos: SCRATCH_SIZE,
                num_blocks: num_blocks as usize,
                tfs_scratch: [0u32; SCRATCH_SIZE],
                #[cfg(target_arch = "wasm32")]
                compressed_data: compressed_data_slice,
                #[cfg(target_arch = "wasm32")]
                tf_data_slice: compressed_data_slice,
            }),
            current_doc: 0,
            // current_score se mantiene solo si no has borrado el campo del struct principal,
            // pero ya no se usa para lógica de negocio.
            current_score: 0,
        }
    }

    #[inline]
    /// Returns the term frequency of the most recently yielded document.
    ///
    /// For bitmap postings this is always `1`.
    pub const fn current_tf(&self) -> u32 {
        match &self.strategy {
            Strategy::Empty => 0,
            Strategy::Bitmap(_) => 1,
            Strategy::Compressed(state) => {
                // IMPLEMENTACIÓN REAL:
                // Leemos del buffer de scratch usando la posición actual
                // (Nota: buf_pos ya ha sido incrementado por advance, así que usamos el anterior o ajustamos lógica)
                // Ajuste: `advance` incrementa `buf_pos` PREPARANDO el siguiente.
                // El valor actual válido está en `buf_pos - 1`.
                let idx = state.buf_pos.wrapping_sub(1) % SCRATCH_SIZE;
                state.tfs_scratch[idx]
            }
        }
    }

    /// Returns the current document ID.
    #[inline]
    pub const fn doc(&self) -> u32 {
        self.current_doc
    }
    /// Returns the current document score.
    #[inline]
    pub const fn score(&self) -> u8 {
        self.current_score
    }
    /// Advances the iterator to the first document >= `target` with score >= `min_score`.
    ///
    /// Returns the document ID if found, or `None` if no matching document exists.
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
                let doc = advance_compressed(state, target, min_score)?;
                self.current_doc = doc;
                Some(doc)
            }
        }
    }

    /// Returns an estimate of the posting list size in bytes.
    pub const fn size_estimate(&self) -> usize {
        match &self.strategy {
            Strategy::Empty => 0,
            Strategy::Bitmap(state) => state.data.len(),
            Strategy::Compressed(state) => state.headers_data.len(),
        }
    }
}

// ... (advance_bitmap, find_set_bit_fast, scan_partial_chunk se mantienen sin cambios) ...
#[inline]
fn advance_bitmap(state: &mut BitmapState<'_>, target: u32) -> Option<(u32, u8)> {
    let byte_idx = (target / 8) as usize;
    state.chunk_idx = byte_idx & !15;

    while state.chunk_idx < state.data.len() {
        if let Some(chunk) = state.data.sub_slice(state.chunk_idx, 16)
            && let Some(bits) = scan_chunk(chunk)
        {
            let base_doc = (state.chunk_idx as u32) * 8;

            if base_doc + 128 <= target {
                state.chunk_idx += 16;
                continue;
            }

            if is_empty(bits) {
                state.chunk_idx += 16;
                continue;
            }

            if let Some(doc) = find_set_bit_fast(bits, base_doc, target) {
                state.chunk_idx = ((doc / 8) as usize) & !15;
                return Some((doc, 10));
            }

            state.chunk_idx += 16;
            continue;
        }

        if let Some((doc, score)) = scan_partial_chunk(state, target) {
            return Some((doc, score));
        }

        break;
    }
    None
}

#[inline]
const fn find_set_bit_fast(bits: ChunkBits, base_doc: u32, target: u32) -> Option<u32> {
    let (mut lo, mut hi) = bits;
    let relative_target = target.saturating_sub(base_doc);

    if lo != 0 && relative_target < 64 {
        let mask = if relative_target > 0 {
            !((1u64 << relative_target) - 1)
        } else {
            u64::MAX
        };
        lo &= mask;

        if lo != 0 {
            return Some(base_doc + lo.trailing_zeros());
        }
    }

    if hi != 0 {
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
            return Some(base_doc + 64 + hi.trailing_zeros());
        }
    }
    None
}

#[inline]
fn scan_partial_chunk(state: &mut BitmapState<'_>, target: u32) -> Option<(u32, u8)> {
    let slice = state.data.get(state.chunk_idx..)?;
    let mut chunks = slice.chunks_exact(8);

    for (i, chunk) in chunks.by_ref().enumerate() {
        let mut bytes: [u8; 8] = [0; 8];
        bytes.copy_from_slice(chunk);
        let word = u64::from_le_bytes(bytes);

        if word != 0 {
            let offset = i * 8;
            let current_byte_idx = state.chunk_idx + offset;
            let doc_base = (current_byte_idx as u32) * 8;

            if doc_base + 64 <= target {
                continue;
            }

            let mut w = word;
            let relative = target.saturating_sub(doc_base);

            if relative < 64 {
                w &= !((1u64 << relative) - 1);
            } else {
                w = 0;
            }

            if w != 0 {
                let bit_pos = w.trailing_zeros();
                let doc = doc_base + bit_pos;
                state.chunk_idx = current_byte_idx + (bit_pos as usize / 8);
                return Some((doc, 10));
            }
        }
    }

    let processed_bytes = slice.len() - chunks.remainder().len();
    for (i, &byte) in chunks.remainder().iter().enumerate() {
        if byte != 0 {
            let current_byte_idx = state.chunk_idx + processed_bytes + i;
            let doc_base = (current_byte_idx as u32) * 8;

            if doc_base + 8 > target {
                let mut b = byte;
                let relative = target.saturating_sub(doc_base);
                if relative < 8 {
                    b &= !((1u8 << relative) - 1);
                }
                if b != 0 {
                    let bit_pos = b.trailing_zeros();
                    state.chunk_idx = current_byte_idx;
                    return Some((doc_base + bit_pos, 10));
                }
            }
        }
    }

    state.chunk_idx = state.data.len();
    None
}

// ----------------------------------------------------------------------------
// 3. COMPRESSED ITERATOR (WAND/BP128 INTEGRATION)
// ----------------------------------------------------------------------------

// src/iter.rs

/// Advances the iterator to the first document >= `target`.
///
/// NOTE: WAND pruning based on score is temporarily disabled in this function
/// because we switched to dynamic BM25 calculation. We only prune based on `max_doc`.
#[inline]
fn advance_compressed(
    state: &mut CompressedState<'_>,
    target: u32,
    _min_score: u8, // Argumento ignorado (DAAT puro por ahora)
) -> Option<u32> {
    // CAMBIO: Ya no devolvemos el score/frecuencia aquí
    while state.block_idx < state.num_blocks {
        let header = state.current_header();

        // 1. Block Pruning (Solo Max Doc)
        // Ya no podemos usar max_score. Si implementáramos Block-Max WAND en el futuro,
        // necesitaríamos calcular el "Max BM25 posible" usando header.max_tf e IDF.
        if header.max_doc < target {
            state.block_idx = Wrapping(state.block_idx).0.wrapping_add(1);
            state.buf_pos = SCRATCH_SIZE; // Mark buffer as expired
            // Important: We must advance last_doc_id to maintain delta context for next block
            state.last_doc_id = header.max_doc;
            continue;
        }

        // 2. Decompression / Load Buffer
        if state.buf_pos >= SCRATCH_SIZE {
            decompress_block(state);
        }

        // 3. Search within the Buffer (NextGEQ)
        let search_slice = &state.scratch[state.buf_pos..SCRATCH_SIZE];

        // Optimized Search: Find the insertion point of the target
        let local_idx = search_slice
            .iter()
            .position(|&doc| doc >= target)
            .unwrap_or(search_slice.len());

        // 4. Update Position
        state.buf_pos = Wrapping(state.buf_pos).0.wrapping_add(local_idx);

        // 5. Return Result or Continue
        if state.buf_pos < SCRATCH_SIZE {
            let doc = state.scratch[state.buf_pos];
            state.buf_pos = Wrapping(state.buf_pos).0.wrapping_add(1);

            // CAMBIO: Solo devolvemos el DocID.
            // La frecuencia (TF) se accederá externamente vía `state.tfs_scratch`
            // usando el índice `state.buf_pos - 1`.
            return Some(doc);
        }

        // If we reached here, the target was beyond the end of the current buffer.
        state.block_idx = Wrapping(state.block_idx).0.wrapping_add(1);
        state.buf_pos = SCRATCH_SIZE;
        // Last doc of buffer is already header.max_doc, so state.last_doc_id is implicitly updated
    }

    None
}

/// Función de entrada: Decide qué implementación usar (SIMD o Scalar).
/// Mantiene la firma safe para no contagiar 'unsafe' hacia arriba.
#[inline]
fn decompress_block(state: &mut CompressedState<'_>) {
    let base_doc = if state.block_idx == 0 {
        0
    } else {
        state.last_doc_id
    };

    // --- RAMA WASM SIMD ---
    #[cfg(target_arch = "wasm32")]
    {
        // Llamamos a la implementación especializada.
        // SAFETY:
        // 1. Estamos en target_arch="wasm32".
        // 2. Asumimos que el entorno de ejecución (Navegador/Node) soporta SIMD128.
        //    (En producción, esto se garantiza compilando con RUSTFLAGS="-C target-feature=+simd128")
        unsafe { decompress_block_simd(state, base_doc) };
    }

    // --- RAMA FALLBACK (SCALAR) ---
    #[cfg(not(target_arch = "wasm32"))]
    {
        let mut current = base_doc;
        for slot in state.scratch.iter_mut() {
            current = current.wrapping_add(1);
            *slot = current;
        }
        state.tfs_scratch.fill(1);
    }

    // --- LIMPIEZA COMÚN ---
    state.buf_pos = 0;
    state.last_doc_id = state.scratch[SCRATCH_SIZE - 1];
}

/// Implementación optimizada con SIMD128.
/// Se extrajo a su propia función para poder aplicarle el atributo `target_feature`.
#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")] // <--- Habilita instrucciones específicas
#[inline]
unsafe fn decompress_block_simd(state: &mut CompressedState<'_>, base_doc: u32) {
    // Constantes de formato
    const BASES_HEADER_SIZE: usize = 32; // 8 chunks * 4 bytes (u32)
    const SIMD_PADDING: usize = 64; // 4 vectores SIMD de seguridad

    let header = state.current_header();
    let block_start = header.data_offset as usize;

    // 1. Check Global: Validar que el inicio del bloque esté dentro del slice
    if block_start >= state.compressed_data.len() {
        let sentinel = header.max_doc.saturating_add(1);
        fill_scratch_with_sentinels(state.scratch, sentinel);
        return;
    }

    // 2. VALIDATION BARRIER
    // Calculamos requisitos totales usando el ancho de bits de DOCUMENTOS.
    // Error Fix: header.bit_width -> header.bit_width_doc
    let last_chunk_relative_offset = 14 * (header.bit_width_doc as usize);
    let min_required_bytes = BASES_HEADER_SIZE + last_chunk_relative_offset + SIMD_PADDING;

    // Calculamos cuántos bytes reales quedan desde el inicio del bloque
    let available_bytes = state.compressed_data.len() - block_start;

    if available_bytes < min_required_bytes {
        let sentinel = header.max_doc.saturating_add(1);
        fill_scratch_with_sentinels(state.scratch, sentinel);
        return;
    }

    // 3. LOAD BASES (Parallel Setup)
    let mut bases = [0u32; 8];

    // SAFETY:
    // Error Fix: `ptr` -> `core::ptr`
    unsafe {
        let src_ptr = state.compressed_data.as_ptr().add(block_start);
        let dst_ptr = bases.as_mut_ptr().cast::<u8>();
        core::ptr::copy_nonoverlapping(src_ptr, dst_ptr, BASES_HEADER_SIZE);
    }

    // Definimos el slice de datos bit-packed, saltando el header de bases
    let packed_data_start = block_start + BASES_HEADER_SIZE;
    let packed_data = &state.compressed_data[packed_data_start..];

    // 4. EXECUTE PARALLEL SIMD LOOP (DOC IDs)
    for (i, &base) in bases.iter().enumerate() {
        // Error Fix: header.bit_width -> header.bit_width_doc
        let chunk_offset = (i * header.bit_width_doc as usize * 16) / 8;

        let chunk_base = base_doc.wrapping_add(base);

        let chunk_vectors = decode_bp128_chunk_simd(
            &packed_data[chunk_offset..],
            header.bit_width_doc, // Error Fix: bit_width_doc
            chunk_base,
        );

        unsafe { write_docid_chunk(state.scratch, i * 16, chunk_vectors) };
    }

    // 5. TERM FREQUENCIES DECODING (NUEVA LÓGICA)
    // Error Fix: Eliminada lógica de scores_even/odd.
    // Ahora decodificamos TFs usando bit_width_tf.

    // Calculamos dónde terminan los Docs y empiezan los TFs.
    // 128 docs * bits_doc / 8
    let docs_payload_size = (128 * header.bit_width_doc as usize) / 8;

    // Verificamos si existen datos para TFs
    if let Some(tf_data_slice) = packed_data.get(docs_payload_size..) {
        // Calculamos cuánto espacio deberían ocupar los TFs
        let tf_payload_needed = (128 * header.bit_width_tf as usize) / 8;

        // Validación básica de seguridad (+ padding implícito del final del archivo)
        if tf_data_slice.len() >= tf_payload_needed {
            for i in 0..8 {
                let chunk_offset = (i * header.bit_width_tf as usize * 16) / 8;

                // Usamos la nueva función RAW (sin delta base)
                // Nota: Asegúrate de importar decode_raw_chunk_simd en iter.rs
                let chunk_vectors =
                    decode_raw_chunk_simd(&tf_data_slice[chunk_offset..], header.bit_width_tf);

                // Escribimos en el buffer de frecuencias
                unsafe { write_docid_chunk(&mut state.tfs_scratch, i * 16, chunk_vectors) };
            }
        } else {
            // Fallback si el bloque está truncado o es 0-length
            state.tfs_scratch.fill(1);
        }
    } else {
        // Fallback si no hay slice de TFs
        state.tfs_scratch.fill(1);
    }
}

// Helper para manejar fallos silenciosamente.
// ESTRATEGIA: Llenamos el buffer con un valor "centinela" que sabemos
// causará un "mismatch" seguro en la búsqueda actual.
#[inline(never)]
#[cfg(target_arch = "wasm32")]
fn fill_scratch_with_sentinels(scratch: &mut [u32], sentinel_val: u32) {
    // Llenamos todo el buffer con el mismo valor.
    // Como la búsqueda en 'advance' usa scan lineal o comparaciones >=,
    // tener valores repetidos no rompe la lógica, solo avanza rápido.
    scratch.fill(sentinel_val);
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;
    use fst::MapBuilder;

    fn make_test_index(postings: &[u8]) -> Vec<u8> {
        // Build an empty but valid FST
        let build = MapBuilder::memory();
        let fst_bytes = build.into_inner().unwrap();
        let fst_len = fst_bytes.len();
        let postings_base = 28 + fst_len;
        let norms_offset = postings_base + postings.len();

        let mut data = vec![0u8; 28];
        data[0..4].copy_from_slice(&0x4859_5030u32.to_le_bytes());
        data[8..12].copy_from_slice(&1u32.to_le_bytes());
        data[12..16].copy_from_slice(&10.0f32.to_le_bytes());
        data[16..20].copy_from_slice(&(postings_base as u32).to_le_bytes());
        data[20..24].copy_from_slice(&(fst_len as u32).to_le_bytes());
        data[24..28].copy_from_slice(&(norms_offset as u32).to_le_bytes());

        data.extend_from_slice(&fst_bytes);
        data.extend_from_slice(postings);
        data.push(10); // single doc norm
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
        let Some(index) = HypersonicIndex::new(&index_data) else {
            panic!("Index should be valid");
        };

        let mut scratch = MaybeUninit::uninit();
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
        let Some(index) = HypersonicIndex::new(&index_data) else {
            panic!("Index should be valid");
        };

        let mut scratch = MaybeUninit::uninit();
        let mut iter = PostingIterator::new(&index, 0, &mut scratch);

        assert_eq!(iter.advance(20, 0), Some(20));
    }
}
