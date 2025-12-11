//! Algoritmos de Ranking Probabilístico.
//! Implementación estándar de Okapi BM25.

#[inline]
#[allow(non_snake_case, clippy::cast_precision_loss)]
pub fn compute_idf(doc_freq: u32, total_docs: u32) -> f32 {
    // Fórmula IDF estándar: log(1 + (N - n + 0.5) / (n + 0.5))
    // Añadimos 1.0 para evitar scores negativos en corpus pequeños.
    let n = doc_freq as f32;
    let N = total_docs as f32;
    ((N + 1.0) / (n + 0.5)).ln()
}

/// Calcula la contribución de un término al score de un documento.
///
/// k1: Saturación del término (usualmente 1.2).
/// b:  Penalización por longitud (usualmente 0.75).
#[inline]
pub fn bm25(tf: f32, idf: f32, doc_len: f32, avg_len: f32) -> f32 {
    const K1: f32 = 1.2;
    const B: f32 = 0.75;

    // Reformular: idf * tf * (K1 + 1) / (tf + K1 * (1 - B + B * doc_len/avg_len))
    // = idf * tf * (K1 + 1) / (tf + K1 * (1 - B) + K1 * B * doc_len/avg_len)

    let norm_factor = K1 * B * doc_len / avg_len;
    let denominator = tf + K1 * (1.0 - B) + norm_factor;

    idf * tf * (K1 + 1.0) / denominator
}
