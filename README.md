# tiktoken: OpenAI token counts in SQL

A DuckDB extension for tokenizing text with OpenAI's encodings. The output is identical to tiktoken's `encode_ordinary`.

```sql
LOAD tiktoken;

-- how big is this corpus, and what will embedding it cost?
SELECT count(*), sum(tiktoken_count(body)) FROM read_parquet('docs/*.parquet');

-- which documents won't fit in the embedding model, and a version of each that does
SELECT id, tiktoken_count(body, 'text-embedding-3-small') AS tokens,
       tiktoken_truncate(body, 8191, 'text-embedding-3-small') AS body
FROM docs WHERE tokens > 8191;

SELECT tiktoken_encode('hello world');           -- [24912, 2375]
SELECT tiktoken_decode([24912, 2375]);           -- hello world
```

## Functions

Every function takes an optional last argument, `encoding`. It can be an encoding name (`o200k_base`, `cl100k_base`) or a model name, resolved the same way as tiktoken's `encoding_for_model`: `gpt-4o`, `gpt-5-mini`, `o3`, `gpt-4`, `text-embedding-3-small`, `ft:gpt-4o-mini:...`, and so on.
The default is `o200k_base`, which is used by GPT-4o and later models.
The encoding can be a column, so each row can use a different one.

| function | returns |
|---|---|
| `tiktoken_count(text [, encoding])` | `BIGINT`: the number of tokens |
| `tiktoken_encode(text [, encoding])` | `INTEGER[]`: the token ids |
| `tiktoken_decode(tokens [, encoding])` | `VARCHAR`. Bytes that aren't valid UTF-8 (a token list that ends mid-character) become U+FFFD, as in tiktoken. Unknown ids are an error. |
| `tiktoken_truncate(text, max_tokens [, encoding])` | `VARCHAR`: the longest prefix of `text` that has at most `max_tokens` tokens. It never cuts a character in half. Only the start of the text is tokenized, so it's cheap on long documents. |

A NULL argument gives NULL.

**Special tokens** such as `<|endoftext|>` are encoded as ordinary text (`encode_ordinary`), so a document containing that string doesn't affect the count or cause an error.
This is what you want for counting and embedding untrusted text. It doesn't count the few extra tokens a chat API adds for each message.

**Supported encodings:** `o200k_base` (and `o200k_harmony`, which only adds special tokens) and `cl100k_base`.
The older `p50k_base`, `r50k_base` and `gpt2` encodings aren't supported.

## Performance

Measured on an M-series laptop (4 performance and 4 efficiency cores) with 23 MB of mixed code and prose in 4,206 rows:

| | throughput |
|---|---|
| Python `tiktoken` (`encode_ordinary_batch`, 1 thread) | 12–17 MB/s |
| `tiktoken_count` / `tiktoken_encode`, per core | ~33 MB/s |
| `tiktoken_count` / `tiktoken_encode` | ~115 MB/s |
| `tiktoken_truncate(body, 512)` | ~350 MB/s |

Token ids matched Python tiktoken 0.14 for every row, for both encodings.

## Design notes

- **Uses GitHub's [`bpe`](https://github.com/github/rust-gems/tree/main/crates/bpe) for BPE.** It's linear-time and about 3× faster than `tiktoken-rs`. Its dictionaries are prebuilt at compile time, so they load in about 20 ms, but they account for most of the extension's ~50 MB size.
- **Pre-tokenization is done in this extension, not by `bpe-openai`.** `bpe-openai` shares one regex cache across all threads, so tokenizing in parallel only scaled 1.8× on 8 cores. Here each thread has its own cache. The regex patterns are copied from `bpe-openai`.
- **Large chunks are tokenized on a rayon pool.** DuckDB only parallelizes a scan across row groups (about 122k rows). Without the pool, a table of a few thousand long documents would be tokenized on one thread. A chunk with at least 64 KB of text is split across the pool, which has one thread per core and ignores DuckDB's `threads` setting.
- **Pinned to one DuckDB version.** duckdb-rs uses the unstable C API, so each build works with exactly one DuckDB version (`TARGET_DUCKDB_VERSION` in the Makefile).

## Development

You need Rust (the version is pinned in `rust-toolchain.toml`), Python 3 and Make.

```sh
git clone --recurse-submodules <repo>
make configure        # python venv with the matching duckdb, platform detection
make debug            # build/debug/tiktoken.duckdb_extension
make test             # test/sql/tiktoken.test
cargo test            # unit tests for encoding lookup and truncation
```

Before the first commit, set the version with `make extension_version EXTENSION_VERSION=v0.1.0`. Otherwise it's taken from `git describe`, and with no commits the error text becomes the version and corrupts the extension's metadata.
