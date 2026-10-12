//! Native half of the `bisimulationrs` R package.
//!
//! One function, `handle_json`, forwards to `bisim_api::handle_json`; the R
//! functions in `R/pepa.R` turn the JSON into data frames. The web editor,
//! Python and R therefore run exactly the same Rust code.

use extendr_api::prelude::*;

/// Handle one JSON request (internal; use the `pepa_*` functions).
/// @keywords internal
#[extendr]
fn handle_json(request: &str) -> String {
    bisim_api::handle_json(request)
}

extendr_module! {
    mod bisimulationrs;
    fn handle_json;
}
