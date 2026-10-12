//! Native half of the `bisimulationrs` Python package.
//!
//! It exposes one function, `handle_json`, which forwards to
//! `bisim_api::handle_json`. The pandas-friendly API lives in Python
//! (`python/bisimulationrs/__init__.py`), so the browser, Python and R all
//! run exactly the same Rust code.

use pyo3::prelude::*;

/// Handles one JSON request (see `bisim_api::handle_json`) and returns the
/// JSON response. The GIL is released while the model is analysed.
#[pyfunction]
fn handle_json(py: Python<'_>, request: &str) -> String {
    let request = request.to_owned();
    py.detach(move || bisim_api::handle_json(&request))
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(handle_json, m)?)?;
    Ok(())
}
