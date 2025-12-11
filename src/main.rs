use core::hint::black_box;
use std::time::Instant; // Necesario para Vec en no_std

// Importar desde la librería
use parzel::engine::search;

/// Número de iteraciones por defecto para la prueba de perfilado.
const DEFAULT_ITERATIONS: u64 = 5_000_000_000; // Reducido para un profiling más rápido y enfocado.

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let benchmark = args.get(1).map(|s| s.as_str()).unwrap_or("realistic");
    let iterations: u64 = args
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_ITERATIONS);

    println!("=== Parzel HPC Profiling Benchmark (Foco Único) ===");
    println!("Benchmark: {benchmark}");
    println!("Iterations: {iterations}");
    println!();

    match benchmark {
        "realistic" => bench_realistic_search(iterations),
        _ => {
            eprintln!("Benchmark desconocido. Solo 'realistic' disponible.");
            std::process::exit(1);
        }
    }
}

// ============================================================================
// BENCHMARK PRINCIPAL
// ============================================================================

/// Benchmark de escenario realista: 1000 documentos y 3 términos AND.
/// Simula una carga intensiva de intersección de posting lists, forzando la caché.
fn bench_realistic_search(iterations: u64) {
    println!("[Realistic Search (1k Docs, 3-Term AND)]");

    // Crear un índice con 1000 documentos y 3 términos.
    let index_data = build_realistic_index(1000, 3);
    let mut results = [0u32; 1024];

    // Consulta simulada que interseca los 3 términos densos.
    let query = "term1 term2 term3";

    let start = Instant::now();
    for _ in 0..iterations {
        let count = search(
            black_box(&index_data),
            black_box(query),
            black_box(&mut results),
        );
        black_box(count);
    }
    let elapsed = start.elapsed();

    println!("  {} realistic searches in {:?}", iterations, elapsed);
    println!(
        "  {:.2} ns/search (Latencia promedio por consulta)",
        elapsed.as_nanos() as f64 / iterations as f64
    );
    println!();
}

// ============================================================================
// UTILIDADES DE CONSTRUCCIÓN DE ÍNDICE
// ============================================================================

/// Construye un índice de prueba realista para N documentos.
/// Cada posting list tendrá aproximadamente el 50% de densidad.
fn build_realistic_index(num_docs: u32, num_terms: u32) -> Vec<u8> {
    let mut data = Vec::with_capacity(200_000);

    // Header (20 bytes)
    data.extend_from_slice(&0x4859_5030u32.to_le_bytes()); // Magic "HYP0"
    data.extend_from_slice(&1u32.to_le_bytes()); // Version
    data.extend_from_slice(&num_docs.to_le_bytes()); // Document count (1000)
    data.extend_from_slice(&num_terms.to_le_bytes()); // Term count (3)
    let postings_base_offset = data.len() as u32;
    data.extend_from_slice(&postings_base_offset.to_le_bytes()); // Postings offset (20)

    let mut current_offset = 0;

    for i in 1..=num_terms {
        // Asignación de offset usando la misma lógica hash: (term_id % 1000) * 256
        let term_id = i as u64;
        let hash_offset = (term_id % 1000) as usize * 256;

        // Alineación y Padding
        let padding_needed = hash_offset.saturating_sub(current_offset);
        data.extend(std::iter::repeat_n(0, padding_needed));

        // --- Generación de Posting List (Bitmap 50% de Densidad) ---

        // 1. Tipo: Bitmap (0x00)
        data.push(0x00);

        // 2. Docs Activos: 50% de densidad (simulando listas densas que fuerzan el SIMD)
        let docs: Vec<u32> = (0..num_docs).filter(|doc_id| doc_id % 2 == i % 2).collect();

        let max_doc = *docs.iter().max().unwrap_or(&0);
        let needed_bytes = ((max_doc / 8) + 1) as usize;
        // La clave: Asegurar alineación SIMD de 16 bytes.
        let len_bytes = (needed_bytes + 15) & !15;

        // 3. Longitud
        data.extend_from_slice(&(len_bytes as u32).to_le_bytes());

        // 4. Bitmap data
        let bitmap_start = data.len();
        data.resize(data.len() + len_bytes, 0);

        for &doc in &docs {
            let byte_pos = (doc / 8) as usize;
            let bit_pos = doc % 8;
            data[bitmap_start + byte_pos] |= 1 << bit_pos;
        }

        // El offset actual para la próxima iteración es la longitud total hasta ahora.
        current_offset = data.len() - postings_base_offset as usize;
    }

    data
}

/// No implementada en este módulo, pero necesaria para la compilación.
#[allow(dead_code)]
fn verify_index(data: &[u8]) -> bool {
    parzel::HypersonicIndex::new(data).is_some()
}
