//! Encodings: pre-tokenization, BPE, lookup by name, and truncation. No DuckDB types
//! here, so it's unit-testable.
//!
//! BPE dictionaries come from `bpe-openai`, but pre-tokenization is done here. Its
//! `Tokenizer` shares one regex cache across threads, which serializes them on the
//! cache pool's lock; here each thread has its own cache.

use bpe_openai::byte_pair_encoding::BytePairEncoding;
use regex_automata::{
    Anchored, Input,
    meta::{Cache, Regex},
};
use std::{cell::RefCell, sync::LazyLock};

pub const DEFAULT: &str = "o200k_base";

/// tiktoken's `encoding_for_model` table, restricted to the encodings we ship.
/// Exact names are checked before prefixes, as tiktoken does.
const MODELS: &[(&str, &str)] = &[
    ("o1", "o200k_base"),
    ("o3", "o200k_base"),
    ("o4-mini", "o200k_base"),
    ("gpt-5", "o200k_base"),
    ("gpt-4.1", "o200k_base"),
    ("gpt-4o", "o200k_base"),
    ("gpt-4", "cl100k_base"),
    ("gpt-3.5-turbo", "cl100k_base"),
    ("gpt-3.5", "cl100k_base"),
    ("gpt-35-turbo", "cl100k_base"),
    ("davinci-002", "cl100k_base"),
    ("babbage-002", "cl100k_base"),
    ("text-embedding-ada-002", "cl100k_base"),
    ("text-embedding-3-small", "cl100k_base"),
    ("text-embedding-3-large", "cl100k_base"),
];

const MODEL_PREFIXES: &[(&str, &str)] = &[
    ("o1-", "o200k_base"),
    ("o3-", "o200k_base"),
    ("o4-mini-", "o200k_base"),
    ("gpt-5", "o200k_base"),
    ("gpt-4.5-", "o200k_base"),
    ("gpt-4.1-", "o200k_base"),
    ("chatgpt-4o-", "o200k_base"),
    ("gpt-4o-", "o200k_base"),
    ("gpt-4-", "cl100k_base"),
    ("gpt-3.5-turbo-", "cl100k_base"),
    ("gpt-35-turbo-", "cl100k_base"),
    ("gpt-oss-", "o200k_harmony"),
    ("ft:gpt-4o", "o200k_base"),
    ("ft:gpt-4", "cl100k_base"),
    ("ft:gpt-3.5-turbo", "cl100k_base"),
    ("ft:davinci-002", "cl100k_base"),
    ("ft:babbage-002", "cl100k_base"),
];

pub struct Encoding {
    /// Index into the thread-local regex caches.
    id: usize,
    pub bpe: &'static BytePairEncoding,
    /// tiktoken's split regex, with its negative look-ahead `\s+(?!\S)` rewritten as
    /// pattern 1, `\s+\s`, whose last character is dropped from the match. From bpe-openai.
    pat: Regex,
}

/// The pattern whose last character is a look-ahead.
const LOOKAHEAD: usize = 1;

static O200K: LazyLock<Encoding> = LazyLock::new(|| {
    let pat1 = [
        "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
        "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
        "\\p{N}{1,3}",
        " ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*",
        "\\s*[\\r\\n]+",
        "\\s+$",
    ]
    .join("|");
    Encoding::new(0, &bpe_openai::o200k_base().bpe, &pat1)
});

static CL100K: LazyLock<Encoding> = LazyLock::new(|| {
    let pat1 = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+$";
    Encoding::new(1, &bpe_openai::cl100k_base().bpe, pat1)
});

thread_local! {
    static CACHES: [RefCell<Option<Cache>>; 2] = const { [RefCell::new(None), RefCell::new(None)] };
}

impl Encoding {
    fn new(id: usize, bpe: &'static BytePairEncoding, pat1: &str) -> Self {
        let pat = Regex::new_many(&[pat1, "\\s+\\s", "\\s+"]).expect("valid regex");
        Self { id, bpe, pat }
    }

    /// Calls `f` with each pre-tokenized piece, in order, until it returns false.
    fn for_each_piece(&self, text: &str, mut f: impl FnMut(&str) -> bool) {
        CACHES.with(|caches| {
            let mut cache = caches[self.id].borrow_mut();
            let cache = cache.get_or_insert_with(|| self.pat.create_cache());
            let mut start = 0;
            while start < text.len() {
                let input = Input::new(&text[start..]).anchored(Anchored::Yes);
                let Some(m) = self.pat.search_with(cache, &input) else { break };
                let mut end = start + m.end();
                if m.pattern().as_usize() == LOOKAHEAD {
                    end -= text[start..end].chars().next_back().map_or(0, char::len_utf8);
                }
                assert!(end > start, "pre-tokenizer made no progress");
                if !f(&text[start..end]) {
                    break;
                }
                start = end;
            }
        })
    }

    pub fn count(&self, text: &str) -> usize {
        let mut n = 0;
        self.for_each_piece(text, |piece| {
            n += self.bpe.count(piece.as_bytes());
            true
        });
        n
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut tokens = Vec::with_capacity(text.len() / 4);
        self.for_each_piece(text, |piece| {
            tokens.extend(self.bpe.encode_via_backtracking(piece.as_bytes()));
            true
        });
        tokens
    }

    /// The longest prefix of `text` that encodes to at most `max_tokens` tokens. The prefix
    /// ends on a character boundary, so no partial UTF-8 is produced. Only the pieces up
    /// to the cut are tokenized, so this is cheap for long texts and small limits.
    pub fn truncate<'a>(&self, text: &'a str, max_tokens: usize) -> &'a str {
        let mut used = 0;
        let mut end = 0;
        self.for_each_piece(text, |piece| {
            let n = self.bpe.count(piece.as_bytes());
            if used + n <= max_tokens {
                used += n;
                end += piece.len();
                return true;
            }
            let tokens = self.bpe.encode_via_backtracking(piece.as_bytes());
            end += tokens[..max_tokens - used].iter().map(|&t| self.bpe.token_len(t)).sum::<usize>();
            while !text.is_char_boundary(end) {
                end -= 1;
            }
            false
        });
        let mut out = &text[..end];
        // BPE isn't prefix-stable: re-encoding the cut text can merge differently.
        // Drop characters until it fits; this almost never runs more than once.
        while self.count(out) > max_tokens {
            let last = out.chars().next_back().map_or(0, char::len_utf8);
            out = &out[..out.len() - last];
        }
        out
    }
}

/// Resolves an encoding name (`o200k_base`) or a model name (`gpt-4o`).
pub fn resolve(name: &str) -> Result<&'static Encoding, String> {
    let encoding = MODELS
        .iter()
        .find(|(m, _)| *m == name)
        .or_else(|| MODEL_PREFIXES.iter().find(|(p, _)| name.starts_with(p)))
        .map_or(name, |(_, e)| e);
    match encoding {
        "o200k_base" => Ok(&O200K),
        // harmony only adds special tokens, which ordinary encoding never produces.
        "o200k_harmony" => Ok(&O200K),
        "cl100k_base" => Ok(&CL100K),
        "r50k_base" | "p50k_base" | "p50k_edit" | "gpt2" => Err(format!(
            "encoding '{name}' is not supported; use o200k_base or cl100k_base"
        )),
        _ => Err(format!(
            "unknown encoding or model '{name}'; use o200k_base, cl100k_base or a model name such as gpt-4o"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_encodings_and_models() {
        let o200k = &*O200K as *const _;
        let cl100k = &*CL100K as *const _;
        for (name, want) in [
            ("o200k_base", o200k),
            ("cl100k_base", cl100k),
            ("gpt-4o", o200k),
            ("gpt-4o-2024-05-13", o200k),
            ("gpt-5-mini", o200k),
            ("o3", o200k),
            ("gpt-oss-120b", o200k),
            ("gpt-4", cl100k),
            ("gpt-4-0314", cl100k),
            ("text-embedding-3-small", cl100k),
            ("ft:gpt-4o-mini:org::id", o200k),
        ] {
            assert_eq!(resolve(name).map(|t| t as *const _), Ok(want), "{name}");
        }
        assert!(resolve("p50k_base").err().unwrap().contains("not supported"));
        assert!(resolve("llama").err().unwrap().contains("unknown"));
    }

    #[test]
    fn truncates_to_token_limit() {
        let tok = &*O200K;
        let text = "The quick brown fox jumps over the lazy dog. 日本語のテキスト 🦀🦀🦀";
        let total = tok.count(text);
        for max in 0..=total + 2 {
            let cut = tok.truncate(text, max);
            assert!(text.starts_with(cut));
            assert!(tok.count(cut) <= max, "max {max}: {cut:?}");
            if max >= total {
                assert_eq!(cut, text);
            }
        }
        assert_eq!(tok.truncate("hello world", 1), "hello");
        assert_eq!(tok.truncate("", 5), "");
    }
}
