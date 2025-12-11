//! SIMD-accelerated index processing functions.
//!
//! Provides platform-specific optimizations for scanning bitmaps, 4-bit score accumulation,
//! and binary packing (BP128) decompression, prioritizing register operations and
//! zero-copy semantics for maximum performance on WebAssembly.
//!
//! # Performance Note
//! Returns `(u64, u64)` instead of `u128` because WASM32 lacks native 128-bit
//! arithmetic support. Using `u128` causes slow emulation via compiler intrinsics.

use crate::prelude::*;

// ============================================================================
// WASM32 INTRINSICS (UNSIGNED PURITY)
// ============================================================================
#[rustfmt::skip]
#[cfg(target_arch = "wasm32")]
// We explicitly import the correct unsigned SIMD intrinsics to ensure correct behavior.
use core::arch::wasm32::{
    // DocID Intrinsics (u32)
    u32x4, u32x4_add, u32x4_ge, u32x4_splat, u32x4_extract_lane,
    // Score Intrinsics (u8)
    u8x16, u8x16_add_sat, u8x16_bitmask, u8x16_shr, u8x16_splat, u8x16_swizzle,
    // Base Types & Generic Ops
    v128, v128_and, v128_any_true, v128_load, v128_store,
};

/// Result of scanning a 16-byte chunk: (low 64 bits, high 64 bits).
///
/// We avoid `u128` because WASM32 emulates it slowly. This representation
/// allows efficient use of `trailing_zeros()` on native 64-bit values.
pub type ChunkBits = (u64, u64);

/// A tuple of four SIMD vectors representing a decoded BP128 chunk of document IDs.
/// (4 vectors * 4 lanes = 16 `DocIDs`).
#[cfg(target_arch = "wasm32")]
pub type DocIdChunk = (v128, v128, v128, v128);

#[rustfmt::skip]
/// Lookup table for scoring based on byte population counts or 4-bit quantization.
/// Maps a 4-bit input (0-15) to an 8-bit score weight (0-255).
#[cfg(target_arch = "wasm32")]
pub const SCORING_LUT: v128 = u8x16(
    0, 15, 30, 45, 
    60, 75, 90, 105, 
    120, 135, 150, 165, 
    180, 195, 210, 255,
);

// ============================================================================
// 1. BITMAP SCANNING FUNCTIONS
// ============================================================================

/// Scans a 16-byte chunk and returns it as a pair of 64-bit values.
///
/// Uses WASM SIMD intrinsics when available, with early-exit optimization
/// for all-zero chunks.
///
/// # Arguments
/// * `chunk` - A byte slice of at least 16 bytes.
///
/// # Returns
/// * `Some((0, 0))` - All 16 bytes are zero.
/// * `Some((lo, hi))` - The 128 bits split into low and high 64-bit parts.
/// * `None` - Chunk is too small (< 16 bytes).
#[inline]
#[must_use]
pub fn scan_chunk(chunk: &[u8]) -> Option<ChunkBits> {
    if chunk.len() < 16 {
        return None;
    }

    #[cfg(target_arch = "wasm32")]
    {
        // SAFETY: We verified chunk has at least 16 bytes.
        Some(scan_chunk_wasm(chunk))
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        Some(scan_chunk_fallback(chunk))
    }
}

#[inline]
#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
fn scan_chunk_wasm(chunk: &[u8]) -> ChunkBits {
    let ptr = chunk.as_ptr();

    // SAFETY:
    // 1. The caller guarantees `chunk` is at least 16 bytes.
    // 2. `v128_load` handles unaligned memory access in WASM.
    let v = unsafe { v128_load(ptr.cast()) };

    // Efficiently check if the entire vector is zero.
    if !v128_any_true(v) {
        return (0, 0);
    }

    // SAFETY:
    // 1. We verified the chunk size is sufficient.
    // 2. We use `read_unaligned` to safely handle potential misalignment of `u64` access.
    unsafe {
        let lo = ptr.cast::<u64>().read_unaligned();
        let hi = ptr.add(8).cast::<u64>().read_unaligned();
        (lo, hi)
    }
}

#[inline]
#[cfg(not(target_arch = "wasm32"))]
fn scan_chunk_fallback(chunk: &[u8]) -> ChunkBits {
    // Validation `chunk.len() >= 16` is guaranteed by `scan_chunk`.

    let mut lo_bytes: [u8; 8] = [0; 8];
    let mut hi_bytes: [u8; 8] = [0; 8];

    // Use copy_from_slice for optimization; compiler often elides bounds checks here.
    lo_bytes.copy_from_slice(&chunk[0..8]);
    hi_bytes.copy_from_slice(&chunk[8..16]);

    let lo = u64::from_le_bytes(lo_bytes);
    let hi = u64::from_le_bytes(hi_bytes);

    (lo, hi)
}

/// Checks if the chunk bits are all zero.
#[inline]
pub const fn is_empty(bits: ChunkBits) -> bool {
    bits.0 == 0 && bits.1 == 0
}

// ============================================================================
// 2. SCORING FUNCTIONS (SWIZZLE-LUT)
// ============================================================================

/// Expands 4-bit quantized scores to 8-bit scores using a LUT (swizzle)
/// and accumulates them into two separate vectors (Even/Odd) using saturating addition.
///
/// This avoids the overhead of shuffling bytes to a single accumulator.
///
/// # Arguments
/// * `quantized_scores_vector` - 128-bit vector containing 32 packed 4-bit scores.
/// * `acc_even` - Accumulator for even-indexed documents (nibbles 0-3).
/// * `acc_odd` - Accumulator for odd-indexed documents (nibbles 4-7).
/// * `lut` - The `SCORING_LUT` vector.
///
/// # Returns
/// A tuple `(acc_even, acc_odd)` with the updated accumulators.
///
/// # Safety
/// The caller must ensure that the function is only called on architectures supporting
/// the `simd128` target feature.
#[inline]
#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
pub unsafe fn accumulate_4bit_scores_simd(
    quantized_scores_vector: v128,
    acc_even: v128,
    acc_odd: v128,
    lut: v128,
) -> (v128, v128) {
    // 1. Decode Low Nibbles (Even Docs)
    // Mask 0x0F isolates the lower 4 bits.
    let mask_low = u8x16_splat(0xF);
    let low_nibbles = v128_and(quantized_scores_vector, mask_low);

    // 2. Decode High Nibbles (Odd Docs)
    // OPTIMIZATION: u8x16_shr performs a logical shift, filling high bits with zeros.
    // Therefore, an additional AND mask is NOT required here.
    let high_nibbles = u8x16_shr(quantized_scores_vector, 4);

    // 3. Expansion & Dequantization (Swizzle / LUT Lookup)
    // Maps the 4-bit index (0-15) to the 8-bit score from the LUT.
    let scores_even = u8x16_swizzle(lut, low_nibbles);
    let scores_odd = u8x16_swizzle(lut, high_nibbles);

    // 4. Saturating Accumulation
    // We return the expression directly to satisfy Clippy's `let_and_return` rule.
    (
        u8x16_add_sat(acc_even, scores_even),
        u8x16_add_sat(acc_odd, scores_odd),
    )
}

/// Decodes and accumulates 4-bit scores for an entire block (128 documents).
///
/// Reads 64 bytes of quantized scores, expands them using the Swizzle-LUT,
/// and writes the results into the split even/odd score buffers.
///
/// # Arguments
/// * `src_ptr` - Pointer to the start of the compressed scores (must be valid for 64 bytes).
/// * `dest_even` - Pointer to the start of the even scores buffer (must be valid for 64 bytes).
/// * `dest_odd` - Pointer to the start of the odd scores buffer (must be valid for 64 bytes).
///
/// # Safety
/// This function is marked `unsafe` because it dereferences raw pointers.
/// The caller MUST ensure that:
/// 1. `src_ptr`, `dest_even`, and `dest_odd` are valid for reads/writes of 64 bytes.
/// 2. The memory regions do not overlap in a way that breaks semantics.
/// 3. The `simd128` target feature is enabled.
#[inline]
#[allow(clippy::cast_ptr_alignment)]
#[target_feature(enable = "simd128")]
#[cfg(target_arch = "wasm32")]
pub unsafe fn decode_scores_block_simd(src_ptr: *const u8, dest_even: *mut u8, dest_odd: *mut u8) {
    // Init accumulators to zero for this block.
    // SAFETY: Safe constructor for zero vector.
    let v_zero = u8x16_splat(0);
    let lut = SCORING_LUT;

    // Cast raw byte pointers to vector pointers.
    // This cast is safe, but dereferencing resulting pointers requires alignment/validity checks (handled by v128_load).
    let src_v = src_ptr.cast::<v128>();
    let dst_even_v = dest_even.cast::<v128>();
    let dst_odd_v = dest_odd.cast::<v128>();

    // Process 128 scores (4 iterations * 32 scores per vector)
    for i in 0..4 {
        // 1. Load 32 quantized scores (16 bytes)
        // SAFETY: Caller guarantees `src_ptr` is valid for 64 bytes.
        // `v128_load` handles unaligned access.
        let v_quantized = unsafe { v128_load(src_v.add(i)) };

        // 2. Expand and Accumulate
        // SAFETY: We are inside a function with `#[target_feature(enable = "simd128")]`.
        let (acc_even, acc_odd) =
            unsafe { accumulate_4bit_scores_simd(v_quantized, v_zero, v_zero, lut) };

        // 3. Store results
        // SAFETY: Caller guarantees `dest_even` and `dest_odd` are valid for 64 bytes.
        unsafe { v128_store(dst_even_v.add(i), acc_even) };
        // SAFETY: Caller guarantees `dest_odd` is valid for 64 bytes.
        unsafe { v128_store(dst_odd_v.add(i), acc_odd) };
    }
}

// ============================================================================
// 3. BP128 DECODING (IN-REGISTER) - OPTIMIZED FOR SPEED
// ============================================================================

/// Extracts a u32 from bitstream at compile-time known position.
/// All parameters are const-propagated by LLVM when IDX and WIDTH are literals.
#[cfg(target_arch = "wasm32")]
macro_rules! extract {
    ($ptr:expr, $idx:literal, $width:literal) => {{
        // Compile-time constants - no runtime computation
        const BYTE_POS: usize = ($idx * $width) / 8;
        const BIT_REM: u32 = (($idx * $width) % 8) as u32;
        const MASK: u32 = (1u32 << $width) - 1;

        // SAFETY: Caller guarantees ptr valid for 64 bytes
        let raw = unsafe { $ptr.add(BYTE_POS).cast::<u32>().read_unaligned() };
        (raw >> BIT_REM) & MASK
    }};
}

/// Unpacks 16 values and assembles into SIMD vectors.
#[cfg(target_arch = "wasm32")]
macro_rules! unpack16 {
    ($ptr:expr, $base:expr, $w:literal) => {{
        (
            u32x4_add(
                u32x4(
                    extract!($ptr, 0, $w),
                    extract!($ptr, 1, $w),
                    extract!($ptr, 2, $w),
                    extract!($ptr, 3, $w),
                ),
                $base,
            ),
            u32x4_add(
                u32x4(
                    extract!($ptr, 4, $w),
                    extract!($ptr, 5, $w),
                    extract!($ptr, 6, $w),
                    extract!($ptr, 7, $w),
                ),
                $base,
            ),
            u32x4_add(
                u32x4(
                    extract!($ptr, 8, $w),
                    extract!($ptr, 9, $w),
                    extract!($ptr, 10, $w),
                    extract!($ptr, 11, $w),
                ),
                $base,
            ),
            u32x4_add(
                u32x4(
                    extract!($ptr, 12, $w),
                    extract!($ptr, 13, $w),
                    extract!($ptr, 14, $w),
                    extract!($ptr, 15, $w),
                ),
                $base,
            ),
        )
    }};
}

/// Generates optimized match dispatch for all bit widths.
#[cfg(target_arch = "wasm32")]
macro_rules! bp128_dispatch {
    ($ptr:expr, $width:expr, $base:expr) => {
        match $width {
            1 => unpack16!($ptr, $base, 1),
            2 => unpack16!($ptr, $base, 2),
            3 => unpack16!($ptr, $base, 3),
            4 => unpack16!($ptr, $base, 4),
            5 => unpack16!($ptr, $base, 5),
            6 => unpack16!($ptr, $base, 6),
            7 => unpack16!($ptr, $base, 7),
            8 => unpack16!($ptr, $base, 8),
            9 => unpack16!($ptr, $base, 9),
            10 => unpack16!($ptr, $base, 10),
            11 => unpack16!($ptr, $base, 11),
            12 => unpack16!($ptr, $base, 12),
            13 => unpack16!($ptr, $base, 13),
            14 => unpack16!($ptr, $base, 14),
            15 => unpack16!($ptr, $base, 15),
            16 => unpack16!($ptr, $base, 16),
            17 => unpack16!($ptr, $base, 17),
            18 => unpack16!($ptr, $base, 18),
            19 => unpack16!($ptr, $base, 19),
            20 => unpack16!($ptr, $base, 20),
            21 => unpack16!($ptr, $base, 21),
            22 => unpack16!($ptr, $base, 22),
            23 => unpack16!($ptr, $base, 23),
            24 => unpack16!($ptr, $base, 24),
            25 => unpack16!($ptr, $base, 25),
            26 => unpack16!($ptr, $base, 26),
            27 => unpack16!($ptr, $base, 27),
            28 => unpack16!($ptr, $base, 28),
            29 => unpack16!($ptr, $base, 29),
            30 => unpack16!($ptr, $base, 30),
            31 => unpack16!($ptr, $base, 31),
            // 32-bit: direct SIMD load (most efficient)
            // SAFETY: ptr is valid for 64 bytes (asserted at entry)
            32 => unsafe {
                (
                    u32x4_add(v128_load($ptr.cast()), $base),
                    u32x4_add(v128_load($ptr.add(16).cast()), $base),
                    u32x4_add(v128_load($ptr.add(32).cast()), $base),
                    u32x4_add(v128_load($ptr.add(48).cast()), $base),
                )
            },
            // Invalid: return base (delta = 0)
            _ => ($base, $base, $base, $base),
        }
    };
}

#[cfg(target_arch = "wasm32")]
macro_rules! unpack16_raw {
    ($ptr:expr, $w:literal) => {{
        (
            u32x4(
                extract!($ptr, 0, $w),
                extract!($ptr, 1, $w),
                extract!($ptr, 2, $w),
                extract!($ptr, 3, $w),
            ),
            u32x4(
                extract!($ptr, 4, $w),
                extract!($ptr, 5, $w),
                extract!($ptr, 6, $w),
                extract!($ptr, 7, $w),
            ),
            u32x4(
                extract!($ptr, 8, $w),
                extract!($ptr, 9, $w),
                extract!($ptr, 10, $w),
                extract!($ptr, 11, $w),
            ),
            u32x4(
                extract!($ptr, 12, $w),
                extract!($ptr, 13, $w),
                extract!($ptr, 14, $w),
                extract!($ptr, 15, $w),
            ),
        )
    }};
}

// Dispatcher para RAW (sin base)
#[cfg(target_arch = "wasm32")]
macro_rules! raw_dispatch {
    ($ptr:expr, $width:expr) => {
        match $width {
            1 => unpack16_raw!($ptr, 1),
            2 => unpack16_raw!($ptr, 2),
            3 => unpack16_raw!($ptr, 3),
            4 => unpack16_raw!($ptr, 4),
            5 => unpack16_raw!($ptr, 5),
            6 => unpack16_raw!($ptr, 6),
            7 => unpack16_raw!($ptr, 7),
            8 => unpack16_raw!($ptr, 8),
            9 => unpack16_raw!($ptr, 9),
            10 => unpack16_raw!($ptr, 10),
            11 => unpack16_raw!($ptr, 11),
            12 => unpack16_raw!($ptr, 12),
            13 => unpack16_raw!($ptr, 13),
            14 => unpack16_raw!($ptr, 14),
            15 => unpack16_raw!($ptr, 15),
            16 => unpack16_raw!($ptr, 16),
            17 => unpack16_raw!($ptr, 17),
            18 => unpack16_raw!($ptr, 18),
            19 => unpack16_raw!($ptr, 19),
            20 => unpack16_raw!($ptr, 20),
            21 => unpack16_raw!($ptr, 21),
            22 => unpack16_raw!($ptr, 22),
            23 => unpack16_raw!($ptr, 23),
            24 => unpack16_raw!($ptr, 24),
            25 => unpack16_raw!($ptr, 25),
            26 => unpack16_raw!($ptr, 26),
            27 => unpack16_raw!($ptr, 27),
            28 => unpack16_raw!($ptr, 28),
            29 => unpack16_raw!($ptr, 29),
            30 => unpack16_raw!($ptr, 30),
            31 => unpack16_raw!($ptr, 31),
            32 => unsafe {
                (
                    v128_load($ptr.cast()),
                    v128_load($ptr.add(16).cast()),
                    v128_load($ptr.add(32).cast()),
                    v128_load($ptr.add(48).cast()),
                )
            },
            _ => (
                u32x4_splat(1),
                u32x4_splat(1),
                u32x4_splat(1),
                u32x4_splat(1),
            ), // Fallback seguro
        }
    };
}

/// Decodifica un chunk de Frecuencias (Raw Integers) sin Delta Encoding.
#[inline]
#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
pub fn decode_raw_chunk_simd(compressed_data: &[u8], bit_width: u8) -> DocIdChunk {
    // Sin assert agresivo aquí para velocidad, el caller valida.
    let ptr = &raw const compressed_data[0];
    raw_dispatch!(ptr, bit_width)
}

/// Decodes a BP128-compressed chunk of document IDs using SIMD instructions.
///
/// Uses compile-time constant propagation for maximum performance.
/// Each bit width generates specialized inline code with zero runtime overhead.
///
/// # Performance
/// - All bit positions, masks, and offsets are compile-time constants
/// - LLVM generates optimal `br_table` for the match dispatch
/// - No function call overhead, no memory lookups for parameters
///
/// # Arguments
/// * `compressed_data` - Slice containing the compressed data. Must be at least 64 bytes.
/// * `bit_width` - Bit width of the compressed data (1-32).
/// * `base_doc_id` - The base document ID to add to each decoded delta.
///
/// # Returns
/// A `DocIdChunk` (tuple of 4 `v128` vectors) containing 16 decoded document IDs.
///
/// # Panics
/// Panics if `compressed_data.len() < 64`.
#[inline]
#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
pub fn decode_bp128_chunk_simd(
    compressed_data: &[u8],
    bit_width: u8,
    base_doc_id: u32,
) -> DocIdChunk {
    assert!(
        compressed_data.len() >= 64,
        "BP128 chunk must be at least 64 bytes"
    );

    let ptr = &raw const compressed_data[0];
    let v_base = u32x4_splat(base_doc_id);

    // LLVM converts this to an efficient br_table with inline code per branch
    bp128_dispatch!(ptr, bit_width, v_base)
}

// ============================================================================
// 4. SIMD UTILITIES
// ============================================================================

/// Compares 16 `DocIDs` against a Target `DocID` and returns a bitmask vector.
///
/// # Arguments
/// * `doc_vec` - A vector containing 4 `DocIDs` (u32).
/// * `target` - The target `DocID` to compare against.
///
/// # Returns
/// A `v128` mask where lanes are all-ones if `DocID >= Target`, else all-zeros.
///
/// # Safety
/// The caller must ensure that `simd128` target feature is enabled.
#[inline]
#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
pub unsafe fn find_geq_mask(doc_vec: v128, target: u32) -> v128 {
    let v_target = u32x4_splat(target);
    // Unsigned comparison: Greater-Than-Or-Equal
    u32x4_ge(doc_vec, v_target)
}

/// Converts a SIMD mask vector to a scalar bitmask (u16).
///
/// Extracts the most significant bit of each byte in the vector.
///
/// # Safety
/// The caller must ensure that `simd128` target feature is enabled.
#[inline]
#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
pub unsafe fn mask_to_scalar(mask: v128) -> u16 {
    // u8x16_bitmask is the correct intrinsic for byte-wise masks.
    u8x16_bitmask(mask)
}

/// Writes a chunk of 16 `DocIDs` (4 vectors) into a mutable `u32` buffer.
///
/// Abstracts pointer arithmetic and SIMD storage.
///
/// # Arguments
/// * `dest` - Mutable slice to write into.
/// * `index` - Starting index in the slice.
/// * `chunk` - The 16 `DocIDs` to write.
///
/// # Safety
/// This function is marked `unsafe` because it performs raw pointer arithmetic.
/// The caller MUST ensure that `dest` has sufficient space (at least 16 elements)
/// starting from `index`.
#[inline]
#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
#[allow(clippy::cast_ptr_alignment)]
pub unsafe fn write_docid_chunk(dest: &mut [u32], index: usize, chunk: DocIdChunk) {
    // Debug assertion to catch out-of-bounds in testing.
    debug_assert!(index + 16 <= dest.len());

    // Get raw pointer to the destination index.
    // &raw mut is safe to create, but casting and writing is unsafe.
    let ptr = (&raw mut dest[index]).cast::<v128>();

    // SAFETY:
    // 1. Caller guarantees `dest` bounds (checked via debug_assert).
    // 2. `v128_store` handles unaligned memory access in WASM (though `dest` is likely aligned).
    // 3. Pointer additions are within the bounds of the slice.
    unsafe {
        v128_store(ptr, chunk.0);
        v128_store(ptr.add(1), chunk.1);
        v128_store(ptr.add(2), chunk.2);
        v128_store(ptr.add(3), chunk.3);
    }
}

/// Extracts the last lane (lane 3) from a `DocIDs` vector.
///
/// Uses the optimized intrinsic instead of memory transmutation.
///
/// # Safety
/// The caller must ensure that `simd128` target feature is enabled.
#[inline]
#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
pub unsafe fn extract_last_lane(v: v128) -> u32 {
    // <3> indicates the lane index (0, 1, 2, 3).
    u32x4_extract_lane::<3>(v)
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scan_all_zeros() {
        let chunk = [0u8; 16];
        assert_eq!(scan_chunk(&chunk), Some((0, 0)));
    }

    #[test]
    fn scan_non_zero_low() {
        let mut chunk = [0u8; 16];
        chunk[0] = 1;
        let Some((lo, hi)) = scan_chunk(&chunk) else {
            panic!("Expected Some result");
        };
        assert_eq!(lo, 1);
        assert_eq!(hi, 0);
    }

    #[test]
    fn scan_non_zero_high() {
        let mut chunk = [0u8; 16];
        chunk[8] = 1;
        let Some((lo, hi)) = scan_chunk(&chunk) else {
            panic!("Expected Some result");
        };
        assert_eq!(lo, 0);
        assert_eq!(hi, 1);
    }

    #[test]
    fn scan_too_small() {
        let chunk = [0u8; 15];
        assert_eq!(scan_chunk(&chunk), None);
    }

    #[test]
    fn scan_high_bit() {
        let mut chunk = [0u8; 16];
        chunk[15] = 0x80;
        let Some((lo, hi)) = scan_chunk(&chunk) else {
            panic!("Expected Some result");
        };
        assert_eq!(lo, 0);
        assert_eq!(hi, 0x80u64 << 56);
    }

    #[test]
    fn scan_both_parts() {
        let mut chunk = [0u8; 16];
        chunk[0] = 0xFF;
        chunk[8] = 0xFF;
        let Some((lo, hi)) = scan_chunk(&chunk) else {
            panic!("Expected Some result");
        };
        assert_eq!(lo, 0xFF);
        assert_eq!(hi, 0xFF);
    }
}
