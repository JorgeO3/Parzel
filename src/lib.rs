//! # Parzel - Hypersonic Search Engine (WASM Kernel) 🚀
//!
//! A high-performance, zero-allocation search engine designed for WebAssembly.
//!
//! ## Architecture: "Unsafe Shell, Safe Core"
//!
//! - **WASM FFI Boundary**: Handled by `wasm-bindgen` for safe data transfer.
//! - **Core Logic**: Fully safe Rust operating on validated slices.
//! - **SIMD Acceleration**: Platform-specific optimizations for bitmap scanning.
//!
//! ## Modules
//!
//! - [`data`]: Index structures and binary format parsing
//! - [`engine`]: High-level search orchestration
//! - [`error`]: Centralized error types and WASM exception bridge
//! - [`iter`]: Posting list iterators
//! - [`simd`]: SIMD-accelerated bitmap operations
//! - [`tokenizer`]: Query tokenization
//!
// Use no_std only for WASM production builds (ensures smaller binary size)
#![cfg_attr(all(target_arch = "wasm32", not(test)), no_std)]
#![cfg_attr(not(test), deny(unsafe_op_in_unsafe_fn))]
#![cfg_attr(not(test), warn(clippy::undocumented_unsafe_blocks))]
#![cfg_attr(not(test), warn(missing_docs))]
#![cfg_attr(not(test), warn(clippy::missing_const_for_fn))]
#![cfg_attr(not(test), warn(clippy::cast_possible_wrap))]
#![cfg_attr(not(test), warn(clippy::cast_precision_loss))]
#![cfg_attr(not(test), warn(clippy::cast_sign_loss))]
#![cfg_attr(not(test), warn(clippy::pedantic))]
#![cfg_attr(not(test), warn(clippy::unwrap_used))]
#![cfg_attr(not(test), warn(clippy::expect_used))]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::missing_errors_doc)]

extern crate alloc;

use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
use lol_alloc::{AssumeSingleThreaded, FreeListAllocator};

#[cfg(target_arch = "wasm32")]
#[global_allocator]
static ALLOCATOR: AssumeSingleThreaded<FreeListAllocator> =
    // SAFETY: This application is single threaded, so using AssumeSingleThreaded is allowed.
    unsafe { AssumeSingleThreaded::new(FreeListAllocator::new()) };

// ============================================================================
// 2. Allocator & Panic Handler (WASM production only)
// ============================================================================

// [Allocator and Panic Handler SNIPPED for brevity, but assumed to remain]

// ============================================================================
// 3. Module Declarations & Prelude
// ============================================================================

mod context;
mod data;
pub mod engine;
mod error;
mod iter;
mod prelude; // Brings in Result, Vec, String, f!, etc.
mod scoring;
pub mod simd;
mod tokenizer;
mod utils;

// Imports for the public layer
use crate::prelude::*;
use wasm_bindgen::JsError;

// ============================================================================
// 4. Public API Re-exports
// ============================================================================

pub use data::HypersonicIndex;
pub use engine::MAX_RESULTS;
pub use error::Error;
pub use iter::{PostingIterator, SCRATCH_SIZE};
pub use tokenizer::{MAX_TOKENS, Tokenizer}; // Export the central Error type

// ============================================================================
// 5. WASM BOUNDARY (Public API)
// ============================================================================

/// Searches the index. Returns matching document IDs as a JavaScript `Uint32Array`.
///
/// # Arguments
/// * `index_data`: Index bytes (`Uint8Array` from JS).
/// * `query`: Search query string.
///
/// # Returns
/// `Result<Box<[u32]>, JsError>`: Array of IDs on success; throws exception on error.
#[wasm_bindgen]
pub fn search(index_data: &[u8], query: &str) -> core::result::Result<Box<[u32]>, JsError> {
    // 1. Execute core logic which returns the internal Result<T, Error>.
    // 2. Map the internal error (Error) to the boundary error (JsError).
    internal_search(index_data, query).map_err(Into::into)
}

// --- Internal Engine Wrapper ---

fn internal_search(index_data: &[u8], query: &str) -> Result<Box<[u32]>> {
    // // 1. FAST VALIDATION
    if index_data.is_empty() {
        // // Empty buffer check.
        return Err("Index data cannot be empty".into());
    }
    if query.trim().is_empty() {
        // // Empty query means zero results.
        return Ok(Box::new([]));
    }

    // // 2. PREPARE OUTPUT BUFFER
    let mut results_buffer = [0u32; MAX_RESULTS];

    // // 3. CORE EXECUTION
    // // Call the engine and get the number of matches found.
    // // NOTE: engine::search must handle internal errors (like corrupted index)
    // // and should ideally return Result<usize> if those errors are recoverable.
    let count = engine::search(index_data, query, &mut results_buffer);

    // // 4. FINAL OUTPUT PACKAGING
    // // Take only the found matches (0..count) and box them for efficient JS transfer.
    // // This creates the exact sized Uint32Array in JavaScript.
    // let exact_matches = results_buffer[..count].to_vec().into_boxed_slice();
    let exact_matches = Box::from(&results_buffer[..count]);

    Ok(exact_matches)
}

// ============================================================================
// Tests
// ============================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;
    use core::mem::MaybeUninit;
    use fst::MapBuilder;
    use wasm_bindgen_test::*;

    // Configuración para correr tests en entorno Node.js
    wasm_bindgen_test_configure!(run_in_node_experimental);

    // ------------------------------------------------------------------------
    // 1. TEST INFRASTRUCTURE (Index Builder)
    // ------------------------------------------------------------------------

    /// Constructor auxiliar para generar índices binarios válidos en tests.
    /// Maneja la complejidad de crear el FST y alinear los offsets.
    struct IndexBuilder {
        // Pares de (Termino, Lista de Documentos)
        terms: Vec<(String, Vec<u32>)>,
    }

    impl IndexBuilder {
        fn new() -> Self {
            Self { terms: Vec::new() }
        }

        /// Agrega un término y su lista de posting (se guardará como Bitmap).
        fn add(mut self, term: &str, docs: &[u32]) -> Self {
            let mut sorted_docs = docs.to_vec();
            sorted_docs.sort_unstable();
            self.terms.push((term.to_string(), sorted_docs));
            self
        }

        /// Construye el blob binario final (Header + FST + Postings).
        fn build(mut self) -> Vec<u8> {
            // A. Ordenar términos alfabéticamente (Requisito de FST)
            self.terms.sort_by(|a, b| a.0.cmp(&b.0));

            // B. Generar Postings Data y calcular offsets para el FST
            let mut postings_data = Vec::new();
            let mut fst_inputs = Vec::new();

            for (term, docs) in self.terms {
                let relative_offset = postings_data.len() as u64;
                fst_inputs.push((term, relative_offset));

                // Escribir Posting (Formato Bitmap)
                // 1. Type Byte (0 = Bitmap)
                postings_data.push(0);

                // 2. Calcular tamaño del bitmap
                if docs.is_empty() {
                    postings_data.extend_from_slice(&0u32.to_le_bytes()); // Len 0
                } else {
                    let max_doc = *docs.last().unwrap();
                    let needed_bytes = ((max_doc / 8) + 1) as usize;
                    // Padding a 16 bytes (simulado para compatibilidad simple)
                    let len_bytes = (needed_bytes + 15) & !15;

                    postings_data.extend_from_slice(&(len_bytes as u32).to_le_bytes());

                    let start_idx = postings_data.len();
                    postings_data.resize(start_idx + len_bytes, 0);

                    for doc in docs {
                        let byte_pos = (doc / 8) as usize;
                        let bit_pos = doc % 8;
                        postings_data[start_idx + byte_pos] |= 1 << bit_pos;
                    }
                }
            }

            // C. Construir FST (Map Term -> Relative Offset)
            let mut map_build = MapBuilder::memory();
            for (term, offset) in fst_inputs {
                map_build.insert(term, offset).unwrap();
            }
            let fst_bytes = map_build.into_inner().unwrap();

            // D. Ensamblar el Archivo Completo
            let mut file = vec![0; 24];

            // [0..4] Magic "HYP0"
            file[0..4].copy_from_slice(&0x4859_5030u32.to_le_bytes());

            // Offsets
            let fst_len = fst_bytes.len();
            let postings_base = 24 + fst_len;

            // [16..20] Postings Base Offset
            file[16..20].copy_from_slice(&(postings_base as u32).to_le_bytes());
            // [20..24] FST Size
            file[20..24].copy_from_slice(&(fst_len as u32).to_le_bytes());

            // Append FST
            file.extend_from_slice(&fst_bytes);

            // Append Postings
            file.extend_from_slice(&postings_data);

            file
        }
    }

    // ------------------------------------------------------------------------
    // 2. HAPPY PATH TESTS (Funcionalidad Básica)
    // ------------------------------------------------------------------------

    #[wasm_bindgen_test]
    fn test_single_term_search() {
        // Index: "apple" -> [10, 20, 30]
        let data = IndexBuilder::new().add("apple", &[10, 20, 30]).build();

        let mut results = [0u32; 10];
        let count = engine::search(&data, "apple", &mut results);

        assert_eq!(count, 3);
        assert_eq!(results[0], 10);
        assert_eq!(results[1], 20);
        assert_eq!(results[2], 30);
    }

    #[wasm_bindgen_test]
    fn test_intersection_and_logic() {
        // Escenario: Búsqueda conjuntiva (AND)
        // "coche" -> [1, 5, 10, 20]
        // "rojo"  -> [5, 20, 30]
        // Query: "coche rojo" -> Debería retornar [5, 20]

        let data = IndexBuilder::new()
            .add("coche", &[1, 5, 10, 20])
            .add("rojo", &[5, 20, 30])
            .build();

        let mut results = [0u32; 10];
        // Nota: engine::search tokeniza por espacios
        let count = engine::search(&data, "coche rojo", &mut results);

        assert_eq!(count, 2, "Debería encontrar la intersección exacta");
        assert_eq!(results[0], 5);
        assert_eq!(results[1], 20);
    }

    #[wasm_bindgen_test]
    fn test_no_results_intersection() {
        // "gato" -> [1, 2]
        // "perro" -> [3, 4]
        // Query: "gato perro" -> []

        let data = IndexBuilder::new()
            .add("gato", &[1, 2])
            .add("perro", &[3, 4])
            .build();

        let mut results = [0u32; 10];
        let count = engine::search(&data, "gato perro", &mut results);
        assert_eq!(count, 0);
    }

    #[wasm_bindgen_test]
    fn test_term_not_found_is_ignored_loose_and() {
        // En tu implementación actual de `search` / `ctx.prepare`:
        // Si un término no existe en el índice, NO se agrega al array de iteradores.
        // Por lo tanto, actúa como si ese término no estuviera en la query (Loose AND).
        // Query: "gato unicornio" -> Busca solo "gato".

        let data = IndexBuilder::new().add("gato", &[10]).build();

        let mut results = [0u32; 10];
        let count = engine::search(&data, "gato unicornio", &mut results);

        assert_eq!(count, 1);
        assert_eq!(results[0], 10);
    }

    #[wasm_bindgen_test]
    fn test_score_retrieval() {
        let data = IndexBuilder::new().add("score_test", &[10, 20]).build();

        let index = HypersonicIndex::new(&data).unwrap();
        let mut scratch = MaybeUninit::uninit();
        let mut iter = PostingIterator::new(
            &index,
            index.get_term_offset("score_test").unwrap(),
            &mut scratch,
        );

        // Doc 10
        assert_eq!(iter.advance(0, 0), Some(10));
        assert!(iter.score() > 0, "El score no debería ser cero");
        // En tu implementación de Bitmap, vi que hardcodeaste el score a 10.
        assert_eq!(iter.score(), 10);
    }

    #[wasm_bindgen_test]
    fn test_boundary_crossing_large_list() {
        // Generamos 200 documentos consecutivos: 0, 1, 2 ... 199
        let docs: Vec<u32> = (0..200).collect();

        let data = IndexBuilder::new().add("denso", &docs).build();

        let index = HypersonicIndex::new(&data).unwrap();
        let mut scratch = MaybeUninit::uninit();
        let mut iter = PostingIterator::new(
            &index,
            index.get_term_offset("denso").unwrap(),
            &mut scratch,
        );

        // 1. Verificar el inicio
        assert_eq!(iter.advance(0, 0), Some(0));

        // 2. Saltar a un límite de bloque (ej: cerca de 128)
        assert_eq!(iter.advance(127, 0), Some(127));
        assert_eq!(iter.advance(128, 0), Some(128)); // Cruce de bloque crítico

        // 3. Verificar el final
        assert_eq!(iter.advance(199, 0), Some(199));
        assert_eq!(iter.advance(200, 0), None);
    }

    // ------------------------------------------------------------------------
    // 3. DESTRUCTIVE / SAFETY TESTS (Lo que faltaba)
    // ------------------------------------------------------------------------

    #[wasm_bindgen_test]
    fn test_safety_truncated_file() {
        // Simulamos que la descarga se cortó a la mitad.
        let valid_data = IndexBuilder::new().add("test", &[1]).build();

        // Cortamos arbitrariamente
        let truncated = &valid_data[..valid_data.len() / 2];
        let mut results = [0u32; 10];

        // El motor NO debe entrar en pánico (panic). Debe retornar 0 o manejarlo internamente.
        // HypersonicIndex::new debería fallar al validar offsets o FST.
        let count = engine::search(truncated, "test", &mut results);
        assert_eq!(count, 0);
    }

    #[wasm_bindgen_test]
    fn test_safety_corrupted_header_offsets() {
        let valid_data = IndexBuilder::new().add("test", &[1]).build();
        let mut corrupted = valid_data.clone();

        // Corrompemos el offset de postings para que apunte fuera de memoria (u32::MAX)
        // Offset 16..20
        corrupted[16..20].copy_from_slice(&u32::MAX.to_le_bytes());

        let mut results = [0u32; 10];
        // HypersonicIndex::new usa checked_add y validaciones de límites.
        // Esto verifica que esas validaciones funcionen y no intenten leer memoria inválida.
        let count = engine::search(&corrupted, "test", &mut results);
        assert_eq!(count, 0);
    }

    #[wasm_bindgen_test]
    fn test_safety_empty_inputs() {
        let mut results = [0u32; 10];

        // Caso 1: Index vacío
        assert_eq!(engine::search(&[], "test", &mut results), 0);

        // Caso 2: Query vacía
        let valid_data = IndexBuilder::new().add("test", &[1]).build();
        assert_eq!(engine::search(&valid_data, "", &mut results), 0);
        assert_eq!(engine::search(&valid_data, "   ", &mut results), 0);
    }

    // ------------------------------------------------------------------------
    // 4. UNIT TESTS FOR COMPONENTS
    // ------------------------------------------------------------------------

    #[test]
    fn unit_posting_iterator_advance() {
        // Prueba unitaria aislada de PostingIterator sin pasar por engine::search
        let data = IndexBuilder::new().add("x", &[10, 20, 30]).build();
        let index = HypersonicIndex::new(&data).expect("Index valid");

        let mut scratch = MaybeUninit::uninit();
        // offset 0 porque "x" es el primer término
        let offset = index.get_term_offset("x").expect("term found");

        let mut iter = PostingIterator::new(&index, offset, &mut scratch);

        assert_eq!(iter.advance(0, 0), Some(10));
        assert_eq!(iter.advance(15, 0), Some(20)); // Skip to >= 15
        assert_eq!(iter.advance(20, 0), Some(20)); // Exact match
        assert_eq!(iter.advance(21, 0), Some(30));
        assert_eq!(iter.advance(31, 0), None);
    }
}
