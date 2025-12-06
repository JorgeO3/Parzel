//! SIMD-accelerated bitmap scanning operations.
//!
//! Provides platform-specific optimizations for scanning bitmaps,
//! with fallback implementations for non-WASM targets.
//!
//! # Performance Note
//! Returns `(u64, u64)` instead of `u128` because WASM32 lacks native 128-bit
//! arithmetic support. Using `u128` causes slow emulation via compiler intrinsics.

#[cfg(target_arch = "wasm32")]
use core::arch::wasm32::{v128_any_true, v128_load};

/// Result of scanning a 16-byte chunk: (low 64 bits, high 64 bits).
///
/// We avoid `u128` because WASM32 emulates it slowly. This representation
/// allows efficient use of `trailing_zeros()` on native 64-bit values.
pub type ChunkBits = (u64, u64);

/// Scans a 16-byte chunk and returns it as a pair of 64-bit values.
///
/// Uses WASM SIMD intrinsics when available, with early-exit optimization
/// for all-zero chunks.
///
/// # Arguments
/// * `chunk` - A byte slice of at least 16 bytes
///
/// # Returns
/// * `Some((0, 0))` - All 16 bytes are zero
/// * `Some((lo, hi))` - The 128 bits split into low and high 64-bit parts
/// * `None` - Chunk is too small
#[inline(never)]
pub fn scan_chunk(chunk: &[u8]) -> Option<ChunkBits> {
    if chunk.len() < 16 {
        return None;
    }

    #[cfg(target_arch = "wasm32")]
    {
        // SAFETY: We verified chunk has at least 16 bytes
        Some(scan_chunk_wasm(chunk))
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        Some(scan_chunk_fallback(chunk))
    }
}

#[cfg(target_arch = "wasm32")]
#[inline]
fn scan_chunk_wasm(chunk: &[u8]) -> ChunkBits {
    let ptr = chunk.as_ptr();
    // SAFETY: Caller guarantees chunk.len() >= 16
    let v = unsafe { v128_load(ptr.cast()) };

    if !v128_any_true(v) {
        return (0, 0);
    }

    // SAFETY: Pointer is valid for 16 bytes
    unsafe {
        let lo = ptr.cast::<u64>().read_unaligned();
        let hi = ptr.add(8).cast::<u64>().read_unaligned();
        (lo, hi)
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[inline]
fn scan_chunk_fallback(chunk: &[u8]) -> ChunkBits {
    // SAFETY: We know chunk.len() >= 16 from caller
    let lo = u64::from_le_bytes(chunk[0..8].try_into().unwrap());
    let hi = u64::from_le_bytes(chunk[8..16].try_into().unwrap());
    (lo, hi)
}

/// Checks if the chunk bits are all zero.
#[inline]
pub const fn is_empty(bits: ChunkBits) -> bool {
    bits.0 == 0 && bits.1 == 0
}

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
        let (lo, hi) = scan_chunk(&chunk).unwrap();
        assert_eq!(lo, 1);
        assert_eq!(hi, 0);
    }

    #[test]
    fn scan_non_zero_high() {
        let mut chunk = [0u8; 16];
        chunk[8] = 1;
        let (lo, hi) = scan_chunk(&chunk).unwrap();
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
        let (lo, hi) = scan_chunk(&chunk).unwrap();
        assert_eq!(lo, 0);
        assert_eq!(hi, 0x80u64 << 56);
    }

    #[test]
    fn scan_both_parts() {
        let mut chunk = [0u8; 16];
        chunk[0] = 0xFF;
        chunk[8] = 0xFF;
        let (lo, hi) = scan_chunk(&chunk).unwrap();
        assert_eq!(lo, 0xFF);
        assert_eq!(hi, 0xFF);
    }
}
