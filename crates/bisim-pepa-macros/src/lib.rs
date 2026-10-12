//! `pepa!`: write a PEPA model inline and have it checked at compile time.
//!
//! Use it through the facade, `bisimulationrs::pepa!`; the expansion refers to
//! `::bisimulationrs::algebra`. The macro body uses the same syntax as `.pepa`
//! files and is parsed and validated by the same parser while compiling, so a
//! syntax error, an undefined component or an unsynchronised passive action is
//! a compile error. Two restrictions come from Rust's tokenizer: comments are
//! Rust comments, and names cannot contain `'`.
//!
//! ```ignore
//! let model = bisimulationrs::pepa! {
//!     r = 0.02;
//!     Healthy = (disease, r).Sick;
//!     Sick = (healing, 0.1).Healthy + (death, 0.01).Dead;
//!     Dead = 0;
//!     Healthy[50]
//! };
//! ```

use proc_macro::TokenStream;
use quote::quote;

#[proc_macro]
pub fn pepa(input: TokenStream) -> TokenStream {
    let source = input.to_string();
    match bisim_algebra::pepa::parse_pepa(&source) {
        Ok(_) => quote! {
            ::bisimulationrs::algebra::pepa::parse_pepa(#source)
                .expect("pepa! model was validated at compile time")
        }
        .into(),
        Err(e) => {
            let message = format!("invalid PEPA model: {e}\n  in: {source}");
            quote! { ::core::compile_error!(#message) }.into()
        }
    }
}
