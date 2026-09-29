//! `tiktoken`: OpenAI tokenization in DuckDB. Output is identical to tiktoken's
//! `encode_ordinary`: special-token text such as `<|endoftext|>` is tokenized as plain text.
//!
//!   SELECT sum(tiktoken_count(body)) FROM docs;                          -- o200k_base by default
//!   SELECT tiktoken_count(body, 'gpt-4') FROM docs;                      -- an encoding or a model
//!   SELECT tiktoken_encode('hello world');                               -- [24912, 2375]
//!   SELECT tiktoken_decode([24912, 2375]);
//!   SELECT tiktoken_truncate(body, 8191, 'text-embedding-3-small') FROM docs;

mod encoding;

use encoding::Encoding;
use duckdb::{
    Connection, Result,
    core::{DataChunkHandle, FlatVector, Inserter, LogicalTypeHandle, LogicalTypeId},
    duckdb_entrypoint_c_api,
    ffi::{duckdb_string_t, duckdb_string_t_data, duckdb_string_t_length},
    vscalar::{ScalarFunctionSignature, VScalar},
    vtab::arrow::WritableVector,
};
use rayon::prelude::*;
use std::error::Error;

type BoxError = Box<dyn Error>;

/// Chunks with at least this much text are tokenized on the rayon pool. DuckDB only
/// parallelizes across row groups (~122k rows), so a table of a few thousand long
/// documents would otherwise be tokenized on one thread.
const PARALLEL_BYTES: usize = 64 * 1024;

/// A VARCHAR input column. DuckDB flattens C API scalar function inputs.
struct Varchars<'a> {
    vector: FlatVector<'a>,
    data: *const duckdb_string_t,
}

impl<'a> Varchars<'a> {
    fn new(input: &'a DataChunkHandle, col: usize) -> Self {
        let vector = input.flat_vector(col);
        let data = unsafe { vector.as_mut_ptr::<duckdb_string_t>() };
        Self { vector, data }
    }

    fn get(&self, row: usize) -> Result<Option<&'a str>, BoxError> {
        if self.vector.row_is_null(row as u64) {
            return Ok(None);
        }
        // Short strings are stored inline in the duckdb_string_t, so read it in place.
        let bytes = unsafe {
            let s = self.data.add(row).cast_mut();
            std::slice::from_raw_parts(duckdb_string_t_data(s).cast::<u8>(), duckdb_string_t_length(*s) as usize)
        };
        Ok(Some(std::str::from_utf8(bytes)?))
    }
}

/// The optional trailing encoding argument, if the signature has one.
struct EncodingArg<'a>(Option<Varchars<'a>>);

impl<'a> EncodingArg<'a> {
    fn new(input: &'a DataChunkHandle, col: usize) -> Self {
        Self((input.num_columns() > col).then(|| Varchars::new(input, col)))
    }

    /// None if the encoding is NULL.
    fn get(&self, row: usize) -> Result<Option<&'static Encoding>, BoxError> {
        let name = match &self.0 {
            Some(col) => col.get(row)?,
            None => Some(encoding::DEFAULT),
        };
        Ok(name.map(encoding::resolve).transpose()?)
    }
}

/// One input row with no NULL arguments.
#[derive(Clone, Copy)]
struct Row<'a> {
    text: &'a str,
    tok: &'static Encoding,
    limit: usize,
}

/// Reads the text (column 0), optional limit and optional encoding; None where any is NULL.
fn read_rows<'a>(
    input: &'a DataChunkHandle,
    limit_col: Option<usize>,
    enc_col: usize,
) -> Result<Vec<Option<Row<'a>>>, BoxError> {
    let text = Varchars::new(input, 0);
    let enc = EncodingArg::new(input, enc_col);
    let limit = limit_col.map(|c| input.flat_vector(c));
    let limits = limit.as_ref().map(|v| unsafe { v.as_slice_with_len::<i64>(input.len()) });
    let mut rows = Vec::with_capacity(input.len());
    for row in 0..input.len() {
        let limit = match (&limit, limits) {
            (Some(v), _) if v.row_is_null(row as u64) => None,
            (Some(_), Some(l)) if l[row] < 0 => return Err("max_tokens must not be negative".into()),
            (Some(_), Some(l)) => Some(l[row] as usize),
            _ => Some(0),
        };
        rows.push(match (text.get(row)?, limit, enc.get(row)?) {
            (Some(text), Some(limit), Some(tok)) => Some(Row { text, tok, limit }),
            _ => None,
        });
    }
    Ok(rows)
}

/// Applies `f` to each non-NULL row, in parallel when there's enough text to be worth it.
fn map_rows<'a, T: Send>(rows: &[Option<Row<'a>>], f: impl Fn(Row<'a>) -> T + Sync) -> Vec<Option<T>> {
    let bytes: usize = rows.iter().flatten().map(|r| r.text.len()).sum();
    if bytes >= PARALLEL_BYTES {
        rows.par_iter().map(|r| r.map(&f)).collect()
    } else {
        rows.iter().map(|r| r.map(&f)).collect()
    }
}

fn varchar() -> LogicalTypeHandle {
    LogicalTypeId::Varchar.into()
}

fn tokens_type() -> LogicalTypeHandle {
    LogicalTypeHandle::list(&LogicalTypeId::Integer.into())
}

/// `f(params)` and `f(params, encoding)`.
fn with_optional_encoding(
    params: fn() -> Vec<LogicalTypeHandle>,
    ret: fn() -> LogicalTypeHandle,
) -> Vec<ScalarFunctionSignature> {
    let mut with_encoding = params();
    with_encoding.push(varchar());
    vec![
        ScalarFunctionSignature::exact(params(), ret()),
        ScalarFunctionSignature::exact(with_encoding, ret()),
    ]
}

// ---------------------------------------------------------------------------
// tiktoken_count(text [, encoding]) -> BIGINT
// ---------------------------------------------------------------------------

struct Count;

impl VScalar for Count {
    type State = ();

    fn invoke(_: &(), input: &mut DataChunkHandle, output: &mut dyn WritableVector) -> Result<(), BoxError> {
        let rows = read_rows(input, None, 1)?;
        let counts = map_rows(&rows, |r| r.tok.count(r.text) as i64);
        let mut out = output.flat_vector();
        for (row, count) in counts.into_iter().enumerate() {
            match count {
                Some(n) => unsafe { out.as_mut_slice::<i64>()[row] = n },
                None => out.set_null(row),
            }
        }
        Ok(())
    }

    fn signatures() -> Vec<ScalarFunctionSignature> {
        with_optional_encoding(|| vec![varchar()], || LogicalTypeId::Bigint.into())
    }
}

// ---------------------------------------------------------------------------
// tiktoken_encode(text [, encoding]) -> INTEGER[]
// ---------------------------------------------------------------------------

struct Encode;

impl VScalar for Encode {
    type State = ();

    fn invoke(_: &(), input: &mut DataChunkHandle, output: &mut dyn WritableVector) -> Result<(), BoxError> {
        let rows = read_rows(input, None, 1)?;
        let encoded = map_rows(&rows, |r| r.tok.encode(r.text));
        let mut out = output.list_vector();
        let mut tokens: Vec<i32> = Vec::with_capacity(encoded.iter().flatten().map(Vec::len).sum());
        for (row, ids) in encoded.into_iter().enumerate() {
            match ids {
                Some(ids) => {
                    out.set_entry(row, tokens.len(), ids.len());
                    tokens.extend(ids.into_iter().map(|id| id as i32));
                }
                None => out.set_null(row),
            }
        }
        unsafe { out.set_child(&tokens) };
        Ok(())
    }

    fn signatures() -> Vec<ScalarFunctionSignature> {
        with_optional_encoding(|| vec![varchar()], tokens_type)
    }
}

// ---------------------------------------------------------------------------
// tiktoken_decode(tokens [, encoding]) -> VARCHAR
// ---------------------------------------------------------------------------

struct Decode;

impl VScalar for Decode {
    type State = ();

    fn invoke(_: &(), input: &mut DataChunkHandle, output: &mut dyn WritableVector) -> Result<(), BoxError> {
        let lists = input.list_vector(0);
        let child = lists.child(lists.len());
        let ids = unsafe { child.as_slice_with_len::<i32>(lists.len()) };
        let enc = EncodingArg::new(input, 1);
        let mut out = output.flat_vector();
        let mut bytes = Vec::new();
        for row in 0..input.len() {
            let Some(tok) = enc.get(row)? else {
                out.set_null(row);
                continue;
            };
            if lists.row_is_null(row as u64) {
                out.set_null(row);
                continue;
            }
            let (offset, len) = lists.get_entry(row);
            bytes.clear();
            for i in offset..offset + len {
                let id = ids[i];
                if child.row_is_null(i as u64) {
                    return Err("tiktoken_decode: token list contains NULL".into());
                }
                if id < 0 || id as usize >= tok.bpe.num_tokens() {
                    return Err(format!("tiktoken_decode: invalid token id {id}").into());
                }
                bytes.extend_from_slice(tok.bpe.token_bytes(id as u32));
            }
            // Token boundaries can split a character; tiktoken's decode replaces those too.
            out.insert(row, String::from_utf8_lossy(&bytes).as_ref());
        }
        Ok(())
    }

    fn signatures() -> Vec<ScalarFunctionSignature> {
        with_optional_encoding(|| vec![tokens_type()], varchar)
    }
}

// ---------------------------------------------------------------------------
// tiktoken_truncate(text, max_tokens [, encoding]) -> VARCHAR
// ---------------------------------------------------------------------------

struct Truncate;

impl VScalar for Truncate {
    type State = ();

    fn invoke(_: &(), input: &mut DataChunkHandle, output: &mut dyn WritableVector) -> Result<(), BoxError> {
        let rows = read_rows(input, Some(1), 2)?;
        let cut = map_rows(&rows, |r| r.tok.truncate(r.text, r.limit));
        let mut out = output.flat_vector();
        for (row, text) in cut.into_iter().enumerate() {
            match text {
                Some(t) => out.insert(row, t),
                None => out.set_null(row),
            }
        }
        Ok(())
    }

    fn signatures() -> Vec<ScalarFunctionSignature> {
        with_optional_encoding(|| vec![varchar(), LogicalTypeId::Bigint.into()], varchar)
    }
}

#[duckdb_entrypoint_c_api]
pub unsafe fn extension_entrypoint(con: Connection) -> Result<(), Box<dyn Error>> {
    con.register_scalar_function::<Count>("tiktoken_count")?;
    con.register_scalar_function::<Encode>("tiktoken_encode")?;
    con.register_scalar_function::<Decode>("tiktoken_decode")?;
    con.register_scalar_function::<Truncate>("tiktoken_truncate")?;
    Ok(())
}
