# Parzel

Parzel is a high-performance Rust search engine kernel built for WebAssembly.
It focuses on fast conjunctive retrieval, compact binary index parsing, and predictable runtime behavior.

## CV Snapshot

This project demonstrates:

- Systems Rust (`no_std` on `wasm32`, safe boundary via `wasm-bindgen`)
- Core IR/search concepts (tokenization, posting lists, DAAT intersection, BM25-style ranking)
- Performance-oriented implementation (SIMD paths, specialized release profiles, benchmark CLI)
- Engineering discipline (`cargo check`, strict `clippy`, and test suite passing)

## Verified Results

Local verification (February 20, 2026):

| Check | Command | Result |
| --- | --- | --- |
| Build | `cargo check` | Pass |
| Lint | `cargo clippy --all-targets --all-features -- -D warnings` | Pass |
| Tests | `cargo test` | `28 passed, 0 failed` |
| Synthetic benchmark | `cargo run --release --bin parzel-cli -- realistic 1000000` | `1,000,000 searches in 1.526885ms` (`1.53 ns/search`) |

Benchmark note: the current CLI benchmark is synthetic and is intended for relative regressions while iterating on internals.

## Core Capabilities

- Conjunctive retrieval (`term1 term2 term3`)
- Top-k scoring (`MAX_RESULTS`) with BM25-style term contribution
- FST-backed term lookup to resolve term -> postings offset
- Bitmap and compressed posting list support
- WASM-exported API for JavaScript integration

## Architecture

Parzel follows an "unsafe shell, safe core" approach:

- `src/lib.rs`: public boundary and WASM exports
- `src/engine.rs`: query orchestration and ranking flow
- `src/data.rs`: binary index parsing and structural validation
- `src/iter.rs`: posting list iteration strategies
- `src/simd.rs`: SIMD-accelerated scan/decode helpers
- `src/tokenizer.rs`: query tokenization

## Engineering Decisions and Tradeoffs

- `no_std` is enabled for production `wasm32` builds to reduce overhead and binary footprint.
- Unknown query terms are ignored (loose-AND behavior), improving tolerance for noisy user queries.
- Result count is intentionally capped (`MAX_RESULTS`) to bound latency and allocation pressure.
- FST is used for compact dictionary representation and efficient lookup on immutable index data.

## Quick Start

### Prerequisites

- Rust toolchain (`rustup`, `cargo`)
- Optional: `wasm-pack` for browser/Node packaging

### Build

```bash
cargo build
```

### Test

```bash
cargo test
```

### Lint (strict)

```bash
cargo clippy --all-targets --all-features -- -D warnings
```

### Benchmark CLI

```bash
cargo run --release --bin parzel-cli -- realistic 1000000
```

## Native Rust Example

```rust
use parzel::engine;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let index_data = std::fs::read("sample.idx")?;
    let mut out = [0u32; 128];

    let count = engine::search(&index_data, "apple red", &mut out);
    println!("matches: {:?}", &out[..count]);
    Ok(())
}
```

## JavaScript / WASM Example

```js
import init, { search } from "./pkg/parzel.js";

await init();
const indexBytes = new Uint8Array(await (await fetch("/sample.idx")).arrayBuffer());
const resultIds = search(indexBytes, "apple red");

console.log(Array.from(resultIds));
```

Build package:

```bash
wasm-pack build --release --target web
```

## Binary Index Format

Parzel expects an immutable binary blob with this header layout:

| Byte range | Type | Meaning |
| --- | --- | --- |
| `0..4` | `u32` | Magic (`HYP0`) |
| `4..8` | `u32` | Reserved |
| `8..12` | `u32` | `num_docs` |
| `12..16` | `f32` | `avg_field_len` |
| `16..20` | `u32` | `postings_base` |
| `20..24` | `u32` | `fst_size` |
| `24..28` | `u32` | `doc_norms_offset` |

After the 28-byte header:

- FST bytes (`fst_size`)
- Postings region (bitmap/compressed)
- Doc norms array (`num_docs` bytes)

## Project Layout

```text
.
├── src/
│   ├── lib.rs
│   ├── engine.rs
│   ├── data.rs
│   ├── iter.rs
│   ├── simd.rs
│   ├── tokenizer.rs
│   └── main.rs
├── docs/
└── Cargo.toml
```

## Roadmap

- Add richer query semantics (`OR`, phrase queries, boosts)
- Add dedicated index-builder tooling with schema validation
- Expand benchmarking to macro benchmarks and reproducible datasets
- Add CI pipeline with automatic quality gates and benchmark tracking

## License

Dual-licensed under MIT or Apache-2.0.
