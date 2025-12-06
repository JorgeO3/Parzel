//! Benchmark runner para profiling con perf/flamegraph
//!
//! Uso:
//!   cargo build --release
//!   perf record -g ./target/release/parzel_cli [benchmark] [iterations]
//!   perf report
//!
//! Benchmarks disponibles: simd, ctz, bitmap, skip, all

use std::hint::black_box;
use std::time::Instant;

// Importar desde la librería
use parzel::{search, HypersonicIndex};

const DEFAULT_ITERATIONS: u64 = 10_000_000;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    
    let benchmark = args.get(1).map(|s| s.as_str()).unwrap_or("all");
    let iterations: u64 = args
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_ITERATIONS);

    println!("=== Parzel Profiling Benchmark ===");
    println!("Benchmark: {benchmark}");
    println!("Iterations: {iterations}");
    println!();

    match benchmark {
        "simd" => bench_simd(iterations),
        "ctz" => bench_ctz(iterations),
        "bitmap" => bench_bitmap(iterations),
        "skip" => bench_skip(iterations),
        "search" => bench_search(iterations),
        "all" => {
            bench_simd(iterations);
            bench_ctz(iterations);
            bench_bitmap(iterations);
            bench_skip(iterations);
            bench_search(iterations);
        }
        _ => {
            eprintln!("Benchmark desconocido: {benchmark}");
            eprintln!("Disponibles: simd, ctz, bitmap, skip, search, all");
            std::process::exit(1);
        }
    }
}

/// Benchmark SIMD scan_chunk
fn bench_simd(iterations: u64) {
    println!("[SIMD scan_chunk]");
    
    // Datos de prueba: 128 bits (16 bytes)
    let all_zeros: [u8; 16] = [0; 16];
    let sparse: [u8; 16] = {
        let mut arr = [0u8; 16];
        arr[8] = 0x80; // bit 71 set
        arr
    };
    let dense: [u8; 16] = [0xFF; 16];

    let start = Instant::now();
    for _ in 0..iterations {
        black_box(parzel::simd::scan_chunk(black_box(&all_zeros)));
        black_box(parzel::simd::scan_chunk(black_box(&sparse)));
        black_box(parzel::simd::scan_chunk(black_box(&dense)));
    }
    let elapsed = start.elapsed();
    
    println!("  {} iterations in {:?}", iterations * 3, elapsed);
    println!("  {:.2} ns/op", elapsed.as_nanos() as f64 / (iterations * 3) as f64);
    println!();
}

/// Benchmark CTZ (trailing zeros) vs naive bitscan
fn bench_ctz(iterations: u64) {
    println!("[CTZ vs Naive Bitscan]");
    
    // Valores con diferentes patrones de bits
    let values: [u64; 4] = [
        0x8000_0000_0000_0000, // bit 63 (máximo trailing zeros)
        0x0000_0000_0000_0001, // bit 0 (mínimo trailing zeros)
        0x0000_0001_0000_0000, // bit 32 (medio)
        0xFFFF_FFFF_FFFF_FFFF, // todos los bits
    ];

    // CTZ optimizado
    let start = Instant::now();
    for _ in 0..iterations {
        for &v in &values {
            black_box(black_box(v).trailing_zeros());
        }
    }
    let ctz_elapsed = start.elapsed();

    // Naive bitscan (para comparación)
    let start = Instant::now();
    for _ in 0..iterations {
        for &v in &values {
            black_box(naive_find_first_set(black_box(v)));
        }
    }
    let naive_elapsed = start.elapsed();

    println!("  CTZ:   {} iterations in {:?}", iterations * 4, ctz_elapsed);
    println!("         {:.2} ns/op", ctz_elapsed.as_nanos() as f64 / (iterations * 4) as f64);
    println!("  Naive: {} iterations in {:?}", iterations * 4, naive_elapsed);
    println!("         {:.2} ns/op", naive_elapsed.as_nanos() as f64 / (iterations * 4) as f64);
    println!("  Speedup: {:.1}x", naive_elapsed.as_nanos() as f64 / ctz_elapsed.as_nanos() as f64);
    println!();
}

#[inline(never)]
fn naive_find_first_set(mut v: u64) -> u32 {
    for i in 0..64 {
        if v & 1 != 0 {
            return i;
        }
        v >>= 1;
    }
    64
}

/// Benchmark bitmap iteration
fn bench_bitmap(iterations: u64) {
    println!("[Bitmap Iteration]");
    
    // Crear bitmap de 1MB (~8M bits)
    let size = 1024 * 1024;
    let mut bitmap: Vec<u8> = vec![0; size];
    
    // Sparse: ~1% bits set
    for i in (0..size * 8).step_by(100) {
        bitmap[i / 8] |= 1 << (i % 8);
    }

    let start = Instant::now();
    for _ in 0..iterations / 1000 {
        let mut count = 0u64;
        for (i, &byte) in bitmap.iter().enumerate() {
            if byte != 0 {
                let mut b = byte;
                while b != 0 {
                    let bit = b.trailing_zeros();
                    count = count.wrapping_add((i * 8) as u64 + bit as u64);
                    b &= b - 1; // clear lowest set bit
                }
            }
        }
        black_box(count);
    }
    let elapsed = start.elapsed();
    
    println!("  {} iterations in {:?}", iterations / 1000, elapsed);
    println!("  {:.2} µs/iteration", elapsed.as_micros() as f64 / (iterations / 1000) as f64);
    println!();
}

/// Benchmark skip operations
fn bench_skip(iterations: u64) {
    println!("[Skip Operations]");
    
    // Simular skip list con búsqueda binaria
    let size = 100_000;
    let data: Vec<u32> = (0..size).map(|i| i * 10).collect();
    
    let targets: Vec<u32> = (0..1000).map(|i| i * 997 % (size * 10)).collect();

    let start = Instant::now();
    for _ in 0..iterations / 1000 {
        for &target in &targets {
            let idx = data.partition_point(|&x| x < target);
            black_box(idx);
        }
    }
    let elapsed = start.elapsed();
    
    println!("  {} searches in {:?}", (iterations / 1000) * 1000, elapsed);
    println!("  {:.2} ns/search", elapsed.as_nanos() as f64 / ((iterations / 1000) * 1000) as f64);
    println!();
}

/// Benchmark full search pipeline
fn bench_search(iterations: u64) {
    println!("[Full Search Pipeline]");
    
    // Crear índice válido mínimo
    let index_data = build_test_index();
    let mut results = [0u32; 1024];
    
    let start = Instant::now();
    for _ in 0..iterations {
        let count = search(black_box(&index_data), black_box("test"), black_box(&mut results));
        black_box(count);
    }
    let elapsed = start.elapsed();
    
    println!("  {} searches in {:?}", iterations, elapsed);
    println!("  {:.2} ns/search", elapsed.as_nanos() as f64 / iterations as f64);
    println!();
}

/// Construye un índice de prueba válido
fn build_test_index() -> Vec<u8> {
    let mut data = Vec::with_capacity(1024);
    
    // Magic "HPSR"
    data.extend_from_slice(b"HPSR");
    
    // Version (1)
    data.extend_from_slice(&1u32.to_le_bytes());
    
    // Document count (100)
    data.extend_from_slice(&100u32.to_le_bytes());
    
    // Term count (1)
    data.extend_from_slice(&1u32.to_le_bytes());
    
    // Term: "test" -> offset
    let term = b"test";
    data.push(term.len() as u8);
    data.extend_from_slice(term);
    
    // Offset to posting list (será el siguiente byte después del header)
    let posting_offset = data.len() as u32 + 4;
    data.extend_from_slice(&posting_offset.to_le_bytes());
    
    // Posting list header
    data.push(0x01); // type: array
    data.extend_from_slice(&10u32.to_le_bytes()); // length: 10 docs
    
    // Doc IDs
    for i in 0..10u32 {
        data.extend_from_slice(&(i * 10).to_le_bytes());
    }
    
    data
}

/// Verificar que el índice es válido
#[allow(dead_code)]
fn verify_index(data: &[u8]) -> bool {
    HypersonicIndex::new(data).is_some()
}
