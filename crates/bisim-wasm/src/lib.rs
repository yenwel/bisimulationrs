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
    let mut buf = Vec::<u8>::with_capacity(len);
    let ptr = buf.as_mut_ptr();
    std::mem::forget(buf);
    ptr
}

/// Handles the request in `ptr[..len]` (allocated by [`alloc`], freed here).
///
/// # Safety
/// `ptr` must come from `alloc(len)` with the same `len`, hold `len`
/// initialised bytes, and not be used again after this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn call(ptr: *mut u8, len: usize) -> usize {
    // SAFETY: `ptr` came from `alloc(len)` and the host wrote `len` bytes.
    let bytes = unsafe { Vec::from_raw_parts(ptr, len, len) };
    let request = String::from_utf8_lossy(&bytes);
    let response = bisim_api::handle_json(&request).into_bytes();
    let n = response.len();
    RESPONSE.with(|r| *r.borrow_mut() = response);
    n
}

/// Start of the last response.
#[unsafe(no_mangle)]
pub extern "C" fn response_ptr() -> *const u8 {
    RESPONSE.with(|r| r.borrow().as_ptr())
}
