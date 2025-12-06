//! Safe memory access abstractions for byte slices.
//!
//! Provides efficient, bounds-checked access patterns for reading
//! binary data without unnecessary allocations.

/// Extension trait for safe byte slice operations.
///
/// All methods perform bounds checking and return `None` on invalid access.
pub trait ByteSliceExt {
    /// Reads a little-endian `u32` at the given offset.
    fn read_u32_le(&self, offset: usize) -> Option<u32>;

    /// Reads a single byte at the given offset.
    fn read_u8(&self, offset: usize) -> Option<u8>;

    /// Returns a sub-slice starting at `offset` with length `len`.
    fn sub_slice(&self, offset: usize, len: usize) -> Option<&[u8]>;
}

impl ByteSliceExt for [u8] {
    #[inline]
    fn read_u32_le(&self, offset: usize) -> Option<u32> {
        self.get(offset..offset.checked_add(4)?)?
            .try_into()
            .ok()
            .map(u32::from_le_bytes)
    }

    #[inline]
    fn read_u8(&self, offset: usize) -> Option<u8> {
        self.get(offset).copied()
    }

    #[inline]
    fn sub_slice(&self, offset: usize, len: usize) -> Option<&[u8]> {
        let end = offset.checked_add(len)?;
        self.get(offset..end)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_u32_le_valid() {
        let data = [0x01, 0x02, 0x03, 0x04];
        assert_eq!(data.read_u32_le(0), Some(0x0403_0201));
    }

    #[test]
    fn read_u32_le_out_of_bounds() {
        let data = [0x01, 0x02, 0x03];
        assert_eq!(data.read_u32_le(0), None);
        assert_eq!(data.read_u32_le(1), None);
    }

    #[test]
    fn read_u8_valid() {
        let data = [0xAB, 0xCD];
        assert_eq!(data.read_u8(0), Some(0xAB));
        assert_eq!(data.read_u8(1), Some(0xCD));
        assert_eq!(data.read_u8(2), None);
    }

    #[test]
    fn sub_slice_valid() {
        let data = [1, 2, 3, 4, 5];
        assert_eq!(data.sub_slice(1, 3), Some(&[2, 3, 4][..]));
        assert_eq!(data.sub_slice(0, 5), Some(&data[..]));
    }

    #[test]
    fn sub_slice_overflow() {
        let data = [1, 2, 3];
        assert_eq!(data.sub_slice(usize::MAX, 1), None);
        assert_eq!(data.sub_slice(1, usize::MAX), None);
    }
}
