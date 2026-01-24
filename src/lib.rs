// In your proc-macro crate (Cargo.toml needs `proc-macro = true`)

use proc_macro::TokenStream;
use quote::{format_ident, quote};
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

    // Generate a unique identifier for the static registration
    let registration_ident = format_ident!("__MIMIR_TYPE_REG_{}", name);

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


        #[cfg_attr(feature = "pyo3", pyo3::pyclass(get_all))]
        #[derive(ts_rs::TS, Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
        #[ts(export, export_to = format!("{}/{}.ts", module_path!().replace("::", "/"), stringify!(#name)))]
        #input  // Re-emit the original struct (now with serde aliases)


        impl crate::TypeBehavior for #name {}

        // Python methods
        #[cfg(feature = "pyo3")]
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

            /// Create from Python dict (or from an existing instance)
            #[classmethod]
            pub fn from_dict(_cls: &pyo3::Bound<'_, pyo3::types::PyType>, obj: &pyo3::Bound<'_, pyo3::prelude::PyAny>) -> pyo3::prelude::PyResult<Self> {
                use pyo3::prelude::*;

                // If the input is already an instance of Self, just clone it
                if let Ok(slf) = obj.extract::<Self>() {
                    return Ok(slf);
                }

                // Normalize datetime objects to ISO strings recursively
                fn normalize_datetimes<'py>(py: Python<'py>, obj: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
                    let datetime_mod = py.import("datetime")?;
                    let datetime_cls = datetime_mod.getattr("datetime")?;

                    // Check if it's a datetime object
                    if obj.is_instance(&datetime_cls)? {
                        let iso_str = obj.call_method0("isoformat")?;
                        return Ok(iso_str);
                    }

                    // Check if it's a dict - recursively normalize values
                    if let Ok(dict) = obj.cast::<pyo3::types::PyDict>() {
                        let new_dict = pyo3::types::PyDict::new(py);
                        for (key, value) in dict.iter() {
                            let normalized_value = normalize_datetimes(py, &value)?;
                            new_dict.set_item(key, normalized_value)?;
                        }
                        return Ok(new_dict.clone().into_any());
                    }

                    // Check if it's a list - recursively normalize elements
                    if let Ok(list) = obj.cast::<pyo3::types::PyList>() {
                        let new_list = pyo3::types::PyList::empty(py);
                        for item in list.iter() {
                            let normalized_item = normalize_datetimes(py, &item)?;
                            new_list.append(normalized_item)?;
                        }
                        return Ok(new_list.into_any());
                    }

                    // Return as-is for other types
                    Ok(obj.clone())
                }

                let py = obj.py();
                let normalized = normalize_datetimes(py, obj)?;

                // Otherwise, try to deserialize from a dict
                use pythonize::{Depythonizer, PythonizeError};
                use serde::Deserialize;

                fn depythonize_with_path<'py, T>(obj: &Bound<'py, PyAny>) -> ::std::result::Result<T, serde_path_to_error::Error<PythonizeError>>
                    where
                        T: Deserialize<'py>,
                    {
                        let mut deserializer = Depythonizer::from_object(obj);
                        serde_path_to_error::deserialize(&mut deserializer)
                    }

                let slf: Self = depythonize_with_path(&normalized).map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                    "Invalid structure at `{}`: {}", e.path(), e.inner()
                ))
                })?;

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

        // Auto-registration for PyO3 module building
        #[cfg(feature = "pyo3")]
        #[allow(non_upper_case_globals)]
        #[linkme::distributed_slice(crate::MIMIR_TYPES)]
        #[linkme(crate = linkme)]
        static #registration_ident: crate::MimirTypeEntry = crate::MimirTypeEntry {
            rust_module_path: module_path!(),
            type_name: stringify!(#name),
            register: |m: &pyo3::Bound<'_, pyo3::types::PyModule>| -> pyo3::PyResult<()> {
                use pyo3::types::PyModuleMethods;
                m.add_class::<#name>()
            },
        };

    };

    expanded.into()
}
