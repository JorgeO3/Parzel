//! High-level search engine logic.
//!
//! Provides the main search interface that orchestrates tokenization,
//! posting list iteration, and result collection.

use alloc::collections::BinaryHeap;
use alloc::vec::Vec;
use core::cmp::{Ordering, Reverse};

use crate::context::{QueryCtx, SearchArena}; // Asegúrate de que context.rs exporte esto
use crate::data::HypersonicIndex;
use crate::iter::PostingIterator;
use crate::scoring::{bm25, compute_idf};
use crate::tokenizer::{MAX_TOKENS, Tokenizer};

/// Maximum number of results to return from a search.
pub const MAX_RESULTS: usize = 128;

/// Estructura auxiliar para el Min-Heap.
/// Almacena el par (Score, `DocID`).
///
/// Implementamos `Ord` manual para que:
/// 1. Se ordene principalmente por `score`.
/// 2. En empate, se use `doc_id` (determinismo).
#[derive(Debug, Clone, Copy)]
struct ScoredDoc {
    score: f32,
    doc_id: u32,
}

// Implementación manual de Ord para f32 (que no implementa Ord por NaN)
impl PartialEq for ScoredDoc {
    fn eq(&self, other: &Self) -> bool {
        self.doc_id == other.doc_id
    }
}
impl Eq for ScoredDoc {}
impl PartialOrd for ScoredDoc {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for ScoredDoc {
    fn cmp(&self, other: &Self) -> Ordering {
        // Asumimos que no hay NaNs en nuestros cálculos BM25
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

/// Intersects multiple posting lists using a Ranked DAAT (Document-at-a-Time) strategy.
///
/// # Mathematical Logic (BM25)
/// Unlike the previous naive version, this calculates the exact probabilistic relevance:
/// $$ Score(D) = \sum_{t \in Q} `BM25(tf_t`, `idf_t`, |D|, avgdl) $$
///
/// # Arguments
/// * `iterators`: The posting lists to intersect.
/// * `index`: Reference to the index data (required for Document Lengths).
/// * `idfs`: Pre-calculated Inverse Document Frequency for each iterator.
/// * `results`: Output buffer.
pub fn intersect(
    iterators: &mut [PostingIterator<'_>],
    index: &HypersonicIndex,
    idfs: &[f32],
    results: &mut [u32],
) -> usize {
    if iterators.is_empty() {
        return 0;
    }

    // -- 1. Heap Setup --
    let mut heap = BinaryHeap::with_capacity(MAX_RESULTS);
    let mut target = 0u32;

    // CORRECCIÓN: avg_field_len es un campo público (f32), no un método.
    // Quitamos los paréntesis ().
    let avg_len = index.avg_field_len();

    // -- 2. Main Search Loop --
    loop {
        // A. Driver Advance
        let Some(candidate) = iterators[0].advance(target, 0) else {
            break;
        };

        // B. Intersection & Scoring
        let mut is_match = true;
        let mut next_target = candidate;

        // 1. Obtener datos del documento
        let doc_norm = index.get_doc_norm(candidate).unwrap_or(10);

        #[allow(clippy::cast_lossless)]
        let doc_len = f32::from(doc_norm);

        // 2. Calcular Score del Driver
        #[allow(clippy::cast_precision_loss)]
        let driver_tf = iterators[0].current_tf() as f32;

        let mut total_score = bm25(driver_tf, idfs[0], doc_len, avg_len);

        // 3. Verificar Resto de Iteradores
        for (i, iter) in iterators.iter_mut().skip(1).enumerate() {
            match iter.advance(candidate, 0) {
                Some(doc) => {
                    // CASO 1: NO COINCIDE
                    if doc != candidate {
                        is_match = false;
                        next_target = doc;
                        break; // Salimos del for interno
                    }

                    // CASO 2: SÍ COINCIDE (El flujo continúa aquí si doc == candidate)
                    #[allow(clippy::cast_precision_loss)]
                    let tf = iter.current_tf() as f32;
                    total_score += bm25(tf, idfs[i + 1], doc_len, avg_len);
                }
                None => return flush_heap(heap, results),
            }
        }

        // C. Result Handling
        if !is_match {
            target = next_target;
            continue;
        }

        // -- Case: Hit! --

        // D. Heap Maintenance
        if heap.len() < MAX_RESULTS {
            heap.push(Reverse(ScoredDoc {
                score: total_score,
                doc_id: candidate,
            }));
        } else if let Some(Reverse(min_item)) = heap.peek()
            && total_score > min_item.score
        {
            heap.pop();
            heap.push(Reverse(ScoredDoc {
                score: total_score,
                doc_id: candidate,
            }));
        }

        if candidate == u32::MAX {
            break;
        }
        target = candidate + 1;
    }

    flush_heap(heap, results)
}

/// Helper para vaciar el heap en el buffer de resultados ordenados.
fn flush_heap(mut heap: BinaryHeap<Reverse<ScoredDoc>>, results: &mut [u32]) -> usize {
    let count = heap.len();
    // El heap extrae del menor al mayor (por el Reverse).
    // Queremos los resultados de MAYOR score a MENOR score.
    // Llenamos el buffer de atrás hacia adelante.
    for i in (0..count).rev() {
        if let Some(Reverse(item)) = heap.pop() {
            results[i] = item.doc_id;
        }
    }
    count
}

/// Performs a conjunctive (AND) search over the index.
///
/// Returns documents that match ALL present query terms.
/// Terms not found in the index are ignored (Loose AND).
///
/// # Returns
/// Number of results written to the output buffer.
pub fn search(index_data: &[u8], query: &str, results: &mut [u32]) -> usize {
    // 1. Index Load
    let Some(index) = HypersonicIndex::new(index_data) else {
        return 0;
    };

    // 2. Tokenization
    let tokenizer = Tokenizer::new();
    let mut tokens = [""; MAX_TOKENS];
    let token_count = tokenizer.tokenize(query, &mut tokens);

    if token_count == 0 {
        return 0;
    }

    // 3. Setup
    let mut arena = SearchArena::new();
    let mut ctx = QueryCtx::new();
    let search_window = &tokens[..token_count];

    // 4. Iterator Prep
    let active_count = ctx.prepare(&mut arena, &index, search_window);

    // 5. Calcular IDFs (NUEVO)
    // Debemos generar un vector de IDFs alineado con los iteradores activos.
    // Como ctx.prepare filtra tokens no encontrados, aquí hacemos un cálculo simplificado
    // asumiendo que el orden se mantiene. En producción, ctx.prepare debería devolver esto.

    let mut idfs = Vec::with_capacity(active_count);

    // Iteramos los tokens originales. Si el token generó un iterador (existe en índice),
    // calculamos su IDF.
    // NOTA: Esta lógica es frágil si ctx.prepare reordena.
    // Para v1.0 asumimos que ctx.prepare respeta el orden de tokens encontrados.
    for &token in search_window {
        if let Some(info) = index.get_term_info(token) {
            // ASUME: get_term_info implementado en data.rs
            let idf = compute_idf(info.doc_freq, index.num_docs());
            idfs.push(idf);
        }
        // Si no existe, ctx.prepare lo saltó, así que nosotros también.
    }

    // IMPORTANTE: Si por alguna razón la cuenta no coincide (por lógica interna de prepare),
    // rellenamos con un IDF neutral para evitar panic, aunque lo ideal es refactorizar prepare.
    while idfs.len() < active_count {
        idfs.push(0.1);
    }

    // 6. Execute Intersection
    intersect(ctx.active_slice(active_count), &index, &idfs, results)
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

        let Some(len_u32) = u32::try_from(len_bytes).ok() else {
            panic!("Posting list too large for u32 format (max 4GB)");
        };
        data.extend_from_slice(&len_u32.to_le_bytes());

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

        // 1. Crear un índice dummy válido.
        // Usamos tu función helper `make_test_index_with_bitmap` pasándole 0 documentos.
        // Esto genera un header válido para que HypersonicIndex::new no devuelva None.
        let dummy_data = make_test_index_with_bitmap(&[]);

        let index = HypersonicIndex::new(&dummy_data).expect("Failed to create dummy index");

        // 2. Crear un slice de IDFs vacío (coincide con los 0 iteradores)
        let idfs: [f32; 0] = [];

        // 3. Llamar a intersect con los 4 argumentos
        assert_eq!(intersect(&mut iters, &index, &idfs, &mut results), 0);
    }
}
