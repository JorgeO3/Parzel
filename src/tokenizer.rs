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
    pub const fn new() -> Self {
        Self { _private: () }
    }

    /// Tokenizes input text into an array of string slices.
    ///
    /// # Arguments
    /// * `text` - Input text to tokenize.
    /// * `out` - Output buffer to store the resulting string slices.
    ///
    /// # Returns
    /// Number of tokens written to `out`.
    ///
    /// # Algorithm
    /// - Uses `split_ascii_whitespace` iterator which handles leading/trailing spaces
    ///   and consecutive delimiters automatically.
    /// - Uses `zip` to pair tokens with the output buffer slots, ensuring we never
    ///   write out of bounds (eliminating bounds checks).
    pub fn tokenize<'a>(&self, text: &'a str, out: &mut [&'a str]) -> usize {
        // Iterator magic:
        // 1. split_ascii_whitespace(): Generates slices of words, skipping all spaces.
        // 2. take(out.len()): Ensures we stop if the text has more words than the buffer.
        // 3. zip(out.iter_mut()): Pairs each found token with a mutable slot in 'out'.
        //    Crucially, 'zip' stops when EITHER iterator is exhausted.

        let mut count = 0;

        for (token, slot) in text.split_ascii_whitespace().zip(out.iter_mut()) {
            *slot = token;
            count += 1;
        }

        count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input() {
        let tokenizer = Tokenizer::new();
        let mut out = [""; 4];
        assert_eq!(tokenizer.tokenize("", &mut out), 0);
    }

    #[test]
    fn whitespace_only() {
        let tokenizer = Tokenizer::new();
        let mut out = [""; 4];
        assert_eq!(tokenizer.tokenize("   \t\n  ", &mut out), 0);
    }

    #[test]
    fn single_word() {
        let tokenizer = Tokenizer::new();
        let mut out = [""; 4];
        let count = tokenizer.tokenize("hello", &mut out);
        assert_eq!(count, 1);
        assert_ne!(out[0], "");
    }

    #[test]
    fn multiple_words() {
        let tokenizer = Tokenizer::new();
        let mut out = [""; 4];
        let count = tokenizer.tokenize("hello world", &mut out);
        assert_eq!(count, 2);
    }

    #[test]
    fn buffer_limit() {
        let tokenizer = Tokenizer::new();
        let mut out = [""; 2];
        let count = tokenizer.tokenize("one two three four", &mut out);
        assert_eq!(count, 2);
    }

    #[test]
    fn extra_whitespace() {
        let tokenizer = Tokenizer::new();
        let mut out = [""; 4];
        let count = tokenizer.tokenize("  hello   world  ", &mut out);
        assert_eq!(count, 2);
    }
}
