//! Intensive benchmarks for the Parzel search engine.
//!
//! These benchmarks test the core algorithms in isolation to identify
//! performance bottlenecks before WASM deployment.
//!
//! # Running benchmarks
//! ```bash
//! # Standard benchmark
//! cargo bench --no-default-features
//!
//! # With flamegraph (requires cargo-flamegraph)
//! cargo flamegraph --bench search_bench --no-default-features -- --bench
//!
//! # View assembly (requires cargo-show-asm)
//! cargo asm --no-default-features --lib parzel::simd::scan_chunk
//! cargo asm --no-default-features --lib parzel::iter::find_set_bit_fast
//! ```

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::prelude::*;

// Re-export internals for benchmarking
use parzel::{search, HypersonicIndex, PostingIterator, SCRATCH_SIZE};

/// Generates a test index with bitmap posting lists.
struct TestIndexBuilder {
    data: Vec<u8>,
}

impl TestIndexBuilder {
    fn new() -> Self {
        let mut data = vec![0u8; 20];
        // Magic "HYP0"
        data[0..4].copy_from_slice(&0x4859_5030u32.to_le_bytes());
        // Postings offset
        data[16..20].copy_from_slice(&20u32.to_le_bytes());
        Self { data }
    }

    /// Adds a bitmap posting list with the given document IDs.
    fn add_bitmap(&mut self, docs: &[u32]) -> usize {
        let offset = self.data.len() - 20;

        // Type = 0 (bitmap)
        self.data.push(0);

        // Calculate bitmap size (16-byte aligned)
        let max_doc = docs.iter().max().copied().unwrap_or(0);
        let needed_bytes = ((max_doc / 8) + 1) as usize;
        let len_bytes = (needed_bytes + 15) & !15;

        // Length field
        self.data
            .extend_from_slice(&(len_bytes as u32).to_le_bytes());

        // Bitmap data
        let start_idx = self.data.len();
        self.data.resize(self.data.len() + len_bytes, 0);

        for &doc in docs {
            let byte_pos = (doc / 8) as usize;
            let bit_pos = doc % 8;
            self.data[start_idx + byte_pos] |= 1 << bit_pos;
        }

        offset
    }

    /// Generates random document IDs with given density.
    fn random_docs(max_doc: u32, density: f64, rng: &mut impl Rng) -> Vec<u32> {
        (0..max_doc)
            .filter(|_| rng.r#gen::<f64>() < density)
            .collect()
    }

    fn build(self) -> Vec<u8> {
        self.data
    }
}

// =============================================================================
// SIMD CHUNK SCANNING BENCHMARKS
// =============================================================================

fn bench_simd_scan(c: &mut Criterion) {
    use parzel::simd::{is_empty, scan_chunk};

    let mut group = c.benchmark_group("simd_scan_chunk");

    // All zeros (fast path)
    let zeros = [0u8; 16];
    group.bench_function("all_zeros", |b| {
        b.iter(|| {
            let bits = scan_chunk(black_box(&zeros)).unwrap();
            is_empty(bits)
        })
    });

    // Sparse (few bits set)
    let mut sparse = [0u8; 16];
    sparse[0] = 0x01;
    sparse[15] = 0x80;
    group.bench_function("sparse", |b| {
        b.iter(|| scan_chunk(black_box(&sparse)))
    });

    // Dense (many bits set)
    let dense = [0xFFu8; 16];
    group.bench_function("dense", |b| {
        b.iter(|| scan_chunk(black_box(&dense)))
    });

    group.finish();
}

// =============================================================================
// BITMAP ITERATION BENCHMARKS
// =============================================================================

fn bench_bitmap_advance(c: &mut Criterion) {
    let mut group = c.benchmark_group("bitmap_advance");
    let mut rng = StdRng::seed_from_u64(42);

    // Test different bitmap sizes
    for &num_docs in &[1_000, 10_000, 100_000, 1_000_000] {
        // Sparse bitmap (~1% density)
        let sparse_docs = TestIndexBuilder::random_docs(num_docs, 0.01, &mut rng);
        let mut builder = TestIndexBuilder::new();
        builder.add_bitmap(&sparse_docs);
        let sparse_index = builder.build();

        group.throughput(Throughput::Elements(sparse_docs.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("sparse_1pct", num_docs),
            &sparse_index,
            |b, index_data| {
                b.iter(|| {
                    let index = HypersonicIndex::new(index_data).unwrap();
                    let mut scratch = [0u32; SCRATCH_SIZE];
                    let mut iter = PostingIterator::new(&index, 0, &mut scratch);
                    let mut count = 0u32;
                    let mut target = 0;
                    while let Some(doc) = iter.advance(target, 0) {
                        count += 1;
                        target = doc + 1;
                    }
                    black_box(count)
                })
            },
        );

        // Dense bitmap (~50% density)
        let dense_docs = TestIndexBuilder::random_docs(num_docs, 0.5, &mut rng);
        let mut builder = TestIndexBuilder::new();
        builder.add_bitmap(&dense_docs);
        let dense_index = builder.build();

        group.throughput(Throughput::Elements(dense_docs.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("dense_50pct", num_docs),
            &dense_index,
            |b, index_data| {
                b.iter(|| {
                    let index = HypersonicIndex::new(index_data).unwrap();
                    let mut scratch = [0u32; SCRATCH_SIZE];
                    let mut iter = PostingIterator::new(&index, 0, &mut scratch);
                    let mut count = 0u32;
                    let mut target = 0;
                    while let Some(doc) = iter.advance(target, 0) {
                        count += 1;
                        target = doc + 1;
                    }
                    black_box(count)
                })
            },
        );
    }

    group.finish();
}

// =============================================================================
// GALLOPING/SKIP BENCHMARKS
// =============================================================================

fn bench_bitmap_skip(c: &mut Criterion) {
    let mut group = c.benchmark_group("bitmap_skip");
    let mut rng = StdRng::seed_from_u64(42);

    // Create a large bitmap
    let docs = TestIndexBuilder::random_docs(1_000_000, 0.1, &mut rng);
    let mut builder = TestIndexBuilder::new();
    builder.add_bitmap(&docs);
    let index_data = builder.build();

    // Benchmark skipping to random positions
    let skip_targets: Vec<u32> = (0..1000).map(|_| rng.r#gen_range(0..1_000_000)).collect();

    group.bench_function("random_skip_1M", |b| {
        b.iter(|| {
            let index = HypersonicIndex::new(&index_data).unwrap();
            let mut scratch = [0u32; SCRATCH_SIZE];
            let mut iter = PostingIterator::new(&index, 0, &mut scratch);
            let mut found = 0u32;
            for &target in &skip_targets {
                if iter.advance(target, 0).is_some() {
                    found += 1;
                }
            }
            black_box(found)
        })
    });

    // Benchmark sequential access (cache-friendly)
    group.bench_function("sequential_1M", |b| {
        b.iter(|| {
            let index = HypersonicIndex::new(&index_data).unwrap();
            let mut scratch = [0u32; SCRATCH_SIZE];
            let mut iter = PostingIterator::new(&index, 0, &mut scratch);
            let mut count = 0u32;
            let mut target = 0;
            while let Some(doc) = iter.advance(target, 0) {
                count += 1;
                target = doc + 1;
            }
            black_box(count)
        })
    });

    group.finish();
}

// =============================================================================
// FULL SEARCH BENCHMARKS
// =============================================================================

fn bench_full_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_search");

    // Create a minimal searchable index
    let mut builder = TestIndexBuilder::new();
    let docs: Vec<u32> = (0..10000).step_by(3).collect();
    builder.add_bitmap(&docs);
    let index_data = builder.build();

    group.bench_function("single_term", |b| {
        let mut results = [0u32; 128];
        b.iter(|| {
            let count = search(black_box(&index_data), black_box("test"), &mut results);
            black_box(count)
        })
    });

    group.finish();
}

// =============================================================================
// CTZ (COUNT TRAILING ZEROS) MICRO-BENCHMARK
// =============================================================================

fn bench_ctz_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("ctz_micro");
    let mut rng = StdRng::seed_from_u64(42);

    // Generate random u64 values with various bit patterns
    let sparse_vals: Vec<u64> = (0..1000)
        .map(|_| 1u64 << rng.r#gen_range(0..64))
        .collect();

    let dense_vals: Vec<u64> = (0..1000).map(|_| rng.r#gen::<u64>() | 1).collect();

    group.bench_function("trailing_zeros_sparse", |b| {
        b.iter(|| {
            let mut sum = 0u32;
            for &v in &sparse_vals {
                sum += black_box(v).trailing_zeros();
            }
            sum
        })
    });

    group.bench_function("trailing_zeros_dense", |b| {
        b.iter(|| {
            let mut sum = 0u32;
            for &v in &dense_vals {
                sum += black_box(v).trailing_zeros();
            }
            sum
        })
    });

    // Compare with naive bit-by-bit scan
    group.bench_function("naive_bitscan_sparse", |b| {
        b.iter(|| {
            let mut sum = 0u32;
            for &v in &sparse_vals {
                let v = black_box(v);
                for i in 0..64u32 {
                    if (v >> i) & 1 != 0 {
                        sum += i;
                        break;
                    }
                }
            }
            sum
        })
    });

    group.finish();
}

// =============================================================================
// MEMORY ACCESS PATTERN BENCHMARKS
// =============================================================================

fn bench_memory_access(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_access");

    // Large bitmap to test cache behavior
    let size = 1024 * 1024; // 1MB
    let data: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();

    // Sequential read
    group.throughput(Throughput::Bytes(size as u64));
    group.bench_function("sequential_1MB", |b| {
        b.iter(|| {
            let mut sum = 0u64;
            for chunk in data.chunks(16) {
                if chunk.len() >= 16 {
                    let lo = u64::from_le_bytes(chunk[0..8].try_into().unwrap());
                    let hi = u64::from_le_bytes(chunk[8..16].try_into().unwrap());
                    sum += lo.wrapping_add(hi);
                }
            }
            black_box(sum)
        })
    });

    // Random access (cache-unfriendly)
    let mut rng = StdRng::seed_from_u64(42);
    let random_offsets: Vec<usize> = (0..10000)
        .map(|_| (rng.r#gen_range(0..size / 16)) * 16)
        .collect();

    group.bench_function("random_10k_reads", |b| {
        b.iter(|| {
            let mut sum = 0u64;
            for &off in &random_offsets {
                let chunk = &data[off..off + 16];
                let lo = u64::from_le_bytes(chunk[0..8].try_into().unwrap());
                sum += lo;
            }
            black_box(sum)
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_simd_scan,
    bench_bitmap_advance,
    bench_bitmap_skip,
    bench_full_search,
    bench_ctz_operations,
    bench_memory_access,
);

criterion_main!(benches);
