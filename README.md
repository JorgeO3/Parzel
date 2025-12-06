# Parzel

A high-performance, zero-allocation search engine kernel designed for WebAssembly.

## Features

- **Minimal binary size**: ~7KB WASM output
- **Zero allocations**: Uses pre-allocated buffers for search results
- **SIMD acceleration**: Uses WASM SIMD for bitmap scanning
- **Safe core design**: "Unsafe shell, safe core" architecture
- **`no_std` compatible**: Works in WASM without standard library

## Architecture

```
┌─────────────────────────────────────────────┐
│              FFI Boundary                   │
│         (Minimal unsafe code)               │
├─────────────────────────────────────────────┤
│               Engine                        │
│    (Search orchestration, intersection)     │
├─────────────────────────────────────────────┤
│  Tokenizer  │  Iterator  │     SIMD         │
│             │ (Posting   │ (Bitmap scan)    │
│             │   lists)   │                  │
├─────────────────────────────────────────────┤
│      Data Structures  │     Utils           │
│   (Index, Headers)    │ (Byte slice ext)    │
└─────────────────────────────────────────────┘
```

## Building

### For WASM (production)

```bash
cargo build --release --target wasm32-unknown-unknown --lib
```

### For native testing

```bash
cargo test --no-default-features
```

## FFI API

The library exports three C-compatible functions for WASM:

```c
// Perform a search, returns number of results
uint32_t ffi_search(
    const uint8_t* index_ptr,
    size_t index_len,
    const uint8_t* query_ptr,
    size_t query_len
);

// Get pointer to results buffer (128 u32s)
const uint32_t* ffi_get_results_ptr();

// Initialize index (reserved for future use)
void ffi_init_index(const uint8_t* ptr, size_t len);
```

## Rust API

```rust
use parzel::{search, HypersonicIndex, PostingIterator, SCRATCH_SIZE};

// High-level search
let mut results = [0u32; 128];
let count = search(index_data, "query terms", &mut results);

// Low-level iteration
let index = HypersonicIndex::new(index_data).unwrap();
let mut scratch = [0u32; SCRATCH_SIZE];
let mut iter = PostingIterator::new(&index, term_offset, &mut scratch);

while let Some(doc_id) = iter.advance(0, 0) {
    println!("Found document: {}", doc_id);
}
```

## Index Format

The index uses a simple binary format:

| Offset | Size | Field |
|--------|------|-------|
| 0      | 4    | Magic ("HYP0" = 0x48595030) |
| 4      | 4    | Number of documents |
| 8      | 8    | Reserved |
| 16     | 4    | Postings offset |
| 20+    | var  | Posting lists |

### Posting List Formats

**Bitmap (type=0)**:
- 1 byte: type (0)
- 4 bytes: length
- N bytes: bitmap data (16-byte aligned)

**Compressed (type=1)**:
- 1 byte: type (1)
- 4 bytes: num_blocks
- 4 bytes: headers_offset
- Block headers + compressed data

## License

MIT OR Apache-2.0
