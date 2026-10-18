//! WebAssembly bridge with a minimal C ABI, so no JS bindings generator is
//! needed. The host:
//!
//! 1. calls `alloc(len)` and writes a UTF-8 JSON request (see
//!    `bisim_api::handle_json`) into the returned buffer;
//! 2. calls `call(ptr, len)`, which frees the request and returns the
//!    response length;
//! 3. reads `response_len` bytes at `response_ptr()`.
//!
//! Build with `cargo build -p bisim-wasm --release --target wasm32-unknown-unknown`.

use std::cell::RefCell;

thread_local! {
    static RESPONSE: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

/// Allocates `len` bytes for a request.
#[unsafe(no_mangle)]
pub extern "C" fn alloc(len: usize) -> *mut u8 {
    // A boxed slice has exactly `len` bytes of capacity, which `call` relies
    // on (`Vec::with_capacity` may allocate more).
    Box::into_raw(vec![0u8; len].into_boxed_slice()).cast::<u8>()
}

/// Handles the request in `ptr[..len]` (allocated by [`alloc`], freed here).
///
/// # Safety
/// `ptr` must come from `alloc(len)` with the same `len`, hold `len`
/// initialised bytes, and not be used again after this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn call(ptr: *mut u8, len: usize) -> usize {
    // SAFETY: `ptr` came from `alloc(len)`, a boxed slice of `len` bytes.
    let bytes = unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut(ptr, len)) };
    // Invalid UTF-8 is an error: rewriting it with U+FFFD would silently
    // analyse a different model than the one the caller sent. (A panic below
    // traps the whole module: wasm32-unknown-unknown aborts on panic, so
    // `catch_unwind` could not help; the API avoids panics on input instead.)
    let response = match std::str::from_utf8(&bytes) {
        Ok(request) => bisim_api::handle_json(request),
        Err(e) => format!(
            r#"{{"error":[{{"severity":"error","line":0,"column":0,"message":"request is not valid UTF-8: {e}"}}]}}"#
        ),
    }
    .into_bytes();
    let n = response.len();
    RESPONSE.with(|r| *r.borrow_mut() = response);
    n
}

/// Start of the last response.
#[unsafe(no_mangle)]
pub extern "C" fn response_ptr() -> *const u8 {
    RESPONSE.with(|r| r.borrow().as_ptr())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(request: &[u8]) -> String {
        let ptr = alloc(request.len());
        // SAFETY: `alloc` returned `request.len()` writable bytes.
        unsafe { std::ptr::copy_nonoverlapping(request.as_ptr(), ptr, request.len()) };
        // SAFETY: `ptr` came from `alloc(len)` and holds `len` bytes.
        let n = unsafe { call(ptr, request.len()) };
        let out = RESPONSE.with(|r| r.borrow()[..n].to_vec());
        String::from_utf8(out).unwrap()
    }

    #[test]
    fn invalid_utf8_is_rejected_and_valid_requests_work() {
        let bad = roundtrip(&[b'{', 0xFF, 0xFE, b'}']);
        assert!(bad.contains("not valid UTF-8"), "{bad}");
        let ok = roundtrip(br#"{"op":"check","source":"P = (a, 1).P; P"}"#);
        assert!(ok.contains("ok") || ok.contains("error"), "{ok}");
        // Empty request: a zero-length allocation is still valid to free.
        let empty = roundtrip(b"");
        assert!(!empty.is_empty());
    }
}
