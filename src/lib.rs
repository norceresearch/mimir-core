// In your proc-macro crate (Cargo.toml needs `proc-macro = true`)

use proc_macro::TokenStream;
use quote::quote;
use syn::{Data, DeriveInput, Fields, parse_macro_input};

/// Convert a snake_case string to camelCase
fn snake_to_camel(s: &str) -> String {
    let mut result = String::new();
    let mut capitalize_next = false;

    for c in s.chars() {
        if c == '_' {
            capitalize_next = true;
        } else if capitalize_next {
            result.push(c.to_ascii_uppercase());
            capitalize_next = false;
        } else {
            result.push(c);
        }
    }
    result
}

#[proc_macro_attribute]
pub fn mimir_type(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut input = parse_macro_input!(item as DeriveInput);
    let name = &input.ident;

    // Add serde alias attributes to each field for camelCase deserialization
    if let Data::Struct(ref mut data) = input.data
        && let Fields::Named(ref mut fields) = data.fields
    {
        for field in fields.named.iter_mut() {
            if let Some(ident) = &field.ident {
                let camel_case_name = snake_to_camel(&ident.to_string());
                // Only add alias if camelCase differs from snake_case
                #[allow(clippy::cmp_owned)]
                if camel_case_name != ident.to_string() {
                    let alias_attr: syn::Attribute = syn::parse_quote! {
                        #[serde(alias = #camel_case_name)]
                    };
                    field.attrs.push(alias_attr);
                }
            }
        }
    }

    // Extract named fields (after mutation)
    let fields = match &input.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(fields) => &fields.named,
            _ => panic!("mimir_type only supports structs with named fields"),
        },
        _ => panic!("mimir_type only supports structs"),
    };

    // Build function parameters: `bar: i64, fizz: String`
    let params = fields
        .iter()
        .map(|f| {
            let name = &f.ident;
            let ty = &f.ty;
            quote! { #name: #ty }
        })
        .collect::<Vec<_>>();

    // Build struct init: `bar, fizz`
    let field_names = fields.iter().map(|f| &f.ident).collect::<Vec<_>>();

    let expanded = quote! {


        #[pyo3::pyclass(get_all)]
        #[derive(ts_rs::TS, Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
        #[ts(export, export_to = format!("{}/{}.ts", module_path!().replace("::", "/"), stringify!(#name)))]
        #input  // Re-emit the original struct (now with serde aliases)


        impl crate::TypeBehavior for #name {}

        // Python methods
        #[pyo3::pymethods]
        impl #name {
            #[new]
            #[allow(clippy::too_many_arguments)]
            pub fn __init__(#(#params),*) -> Self {
                Self {
                    #(#field_names),*
                }
            }

            /// Convert to a Python dict
            pub fn to_dict<'py>(&self, py: pyo3::prelude::Python<'py>) -> pyo3::prelude::PyResult<pyo3::prelude::Bound<'py, pyo3::prelude::PyAny>> {
                let dict = pythonize::pythonize(py, &self)?;
                Ok(dict)
            }

            /// Create from Python dict
            #[classmethod]
            pub fn from_dict(_cls: &pyo3::Bound<'_, pyo3::types::PyType>, obj: &pyo3::Bound<'_, pyo3::prelude::PyAny>) -> pyo3::prelude::PyResult<Self> {
                let slf = pythonize::depythonize(obj)?;
                Ok(slf)
            }

            /// Convert to JSON string
            pub fn to_json(&self) -> pyo3::prelude::PyResult<String> {
                let json = crate::TypeBehavior::to_json(self)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
                Ok(json)
            }

            /// Create from JSON string
            #[classmethod]
            pub fn from_json(_cls: &pyo3::Bound<'_, pyo3::types::PyType>, json: &str) -> pyo3::prelude::PyResult<Self> {
                <Self as crate::TypeBehavior>::from_json(json.to_owned())
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
            }

            /// Compare to another instance
            pub fn __eq__(&self, other: &Self) -> bool {
                self == other
            }

            /// Debug visual of object
            pub fn __str__(&self) -> String {
                format!("{:?}", self)
            }
        }

        // Another for Rust interface Self::new
        impl #name {
            #[allow(clippy::too_many_arguments)]
            pub fn new(#(#params),*) -> Self {
                Self {
                    #(#field_names),*
                }
            }
        }

    };

    expanded.into()
}
