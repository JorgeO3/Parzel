//! Query tokenization for search processing.
//!
//! Provides a simple tokenizer that splits text on whitespace
//! and converts tokens to term IDs for index lookup.

/// Maximum number of tokens supported in a single query.
pub const MAX_TOKENS: usize = 16;

/// Tokenizes input text into term IDs.
///
/// This is a simple whitespace tokenizer suitable for basic search.
/// For production use, consider integrating a proper FST-based tokenizer.
#[derive(Debug, Default)]
pub struct Tokenizer {
    _private: (), // Prevent direct construction
}

impl Tokenizer {
    /// Creates a new tokenizer instance.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self { _private: () }
    }

    /// Tokenizes input text into an array of term IDs.
    ///
    /// # Arguments
    /// * `text` - Input text to tokenize
    /// * `out` - Output buffer for term IDs
    ///
    /// # Returns
    /// Number of tokens written to `out`
    ///
    /// # Algorithm
    /// - Splits on ASCII whitespace
    /// - Generates a simple hash as term ID based on word length
    /// - Skips empty tokens
    pub fn tokenize(&self, text: &str, out: &mut [u64]) -> usize {
        let bytes = text.as_bytes();
        let max_tokens = out.len();

        let mut count = 0;
        let mut start = 0;

        while start < bytes.len() && count < max_tokens {
            // Skip leading whitespace
            while start < bytes.len() && is_whitespace(bytes[start]) {
                start += 1;
            }

            if start >= bytes.len() {
                break;
            }

            // Find end of token
            let mut end = start;
            while end < bytes.len() && !is_whitespace(bytes[end]) {
                end += 1;
            }

            let token_len = end - start;
            if token_len > 0 {
                // Simple term ID: hash based on length and first byte
                let first_byte = u64::from(bytes[start]);
                out[count] = first_byte.wrapping_mul(100).wrapping_add(token_len as u64);
                count += 1;
            }

            start = end;
        }

        count
    }
}

/// Checks if a byte is ASCII whitespace.
#[inline]
const fn is_whitespace(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | b'\r')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input() {
        let tokenizer = Tokenizer::new();
        let mut out = [0u64; 4];
        assert_eq!(tokenizer.tokenize("", &mut out), 0);
    }

    #[test]
    fn whitespace_only() {
        let tokenizer = Tokenizer::new();
        let mut out = [0u64; 4];
        assert_eq!(tokenizer.tokenize("   \t\n  ", &mut out), 0);
    }

    #[test]
    fn single_word() {
        let tokenizer = Tokenizer::new();
        let mut out = [0u64; 4];
        let count = tokenizer.tokenize("hello", &mut out);
        assert_eq!(count, 1);
        assert_ne!(out[0], 0);
    }

    #[test]
    fn multiple_words() {
        let tokenizer = Tokenizer::new();
        let mut out = [0u64; 4];
        let count = tokenizer.tokenize("hello world", &mut out);
        assert_eq!(count, 2);
    }

    #[test]
    fn buffer_limit() {
        let tokenizer = Tokenizer::new();
        let mut out = [0u64; 2];
        let count = tokenizer.tokenize("one two three four", &mut out);
        assert_eq!(count, 2);
    }

    #[test]
    fn extra_whitespace() {
        let tokenizer = Tokenizer::new();
        let mut out = [0u64; 4];
        let count = tokenizer.tokenize("  hello   world  ", &mut out);
        assert_eq!(count, 2);
    }
}
