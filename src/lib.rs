//! Procedural macros for mimirtypes.
//!
//! ## `#[mimir_type]`
//!
//! Attribute macro for structs that generates:
//! - PyO3 bindings (pyclass, pymethods)
//! - Serde serialization with camelCase aliases
//! - TypeScript exports via ts-rs
//! - Automatic module registration via linkme
//! - Python stub info for .pyi generation
//!
//! ## `#[mimir_function]`
//!
//! Attribute macro for functions that generates:
//! - PyO3 pyfunction binding
//! - Automatic module registration via linkme
//! - Python stub info for .pyi generation
//!
//! Supports both sync and async functions. Async functions are automatically
//! wrapped with a tokio runtime for Python callers.

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::{
    Data, DeriveInput, Field, Fields, FnArg, Ident, ItemFn, Pat, ReturnType, Type, parse_macro_input,
    punctuated::Punctuated, token::Comma,
};

// ============================================================================
// Main macro
// ============================================================================

/// Attribute macro that transforms a struct into a full mimir type with:
/// - PyO3 Python bindings
/// - Serde serialization with camelCase aliases
/// - TypeScript type exports
/// - Automatic module registration
/// - Stub info for .pyi generation
#[proc_macro_attribute]
pub fn mimir_type(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut input = parse_macro_input!(item as DeriveInput);
    let name = input.ident.clone();

    // add serde aliases for camelCase field names
    add_serde_aliases(&mut input);

    // extract fields and build params
    let fields = extract_named_fields(&input);
    let (params, field_names) = build_params_and_field_names(fields);
    let stub_fields = generate_stub_fields_info(fields);
    let stub_methods = generate_stub_methods();

    // generate all code blocks
    let struct_def = generate_struct_definition(&input, &name);
    let type_behavior = generate_type_behavior_impl(&name);
    let pymethods = generate_pymethods_impl(&name, &params, &field_names);
    let rust_impl = generate_rust_impl(&name, &params, &field_names);
    let registration = generate_registration(&name, &stub_fields, &stub_methods);

    let expanded = quote! {
        #struct_def
        #type_behavior
        #pymethods
        #rust_impl
        #registration
    };

    expanded.into()
}

// ============================================================================
// mimir_function macro
// ============================================================================

/// Attribute macro that transforms a function into a Python-exposed function with:
/// - PyO3 pyfunction binding
/// - Automatic module registration
/// - Stub info for .pyi generation
///
/// # Example
///
/// ```ignore
/// #[mimir_function]
/// pub fn foo(fizz: String, buzz: Option<usize>) -> Vec<Bar> {
///     // implementation
/// }
/// ```
///
/// This generates a `#[pyfunction]` that can be called from Python and
/// automatically registers it in the appropriate module based on its Rust path.
#[proc_macro_attribute]
pub fn mimir_function(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as ItemFn);
    let fn_name = &input.sig.ident;
    let fn_vis = &input.vis;
    let fn_block = &input.block;
    let fn_attrs = &input.attrs;
    let fn_generics = &input.sig.generics;
    let fn_output = &input.sig.output;
    let fn_asyncness = &input.sig.asyncness;

    // extract parameters
    let params: Vec<_> = input
        .sig
        .inputs
        .iter()
        .filter_map(|arg| {
            if let FnArg::Typed(pat_type) = arg {
                Some(pat_type)
            } else {
                None // skip self parameters
            }
        })
        .collect();

    // build parameter tokens for the function sig
    let param_tokens: Vec<TokenStream2> = params
        .iter()
        .map(|p| {
            let pat = &p.pat;
            let ty = &p.ty;
            quote! { #pat: #ty }
        })
        .collect();

    let stub_signature = generate_function_stub_signature(fn_name, &params, fn_output);
    let registration = generate_function_registration(fn_name, &stub_signature);

    // Impl depends on if we need to wrap it with tokio async
    let fn_impl = if fn_asyncness.is_some() {
        quote! {
            #(#fn_attrs)*
            #[cfg(feature = "pyo3")]
            #[pyo3::pyfunction]
            #fn_vis fn #fn_name #fn_generics (#(#param_tokens),*) #fn_output {
                tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("Failed to create tokio runtime")
                    .block_on(async #fn_block)
            }

            #(#fn_attrs)*
            #[cfg(not(feature = "pyo3"))]
            #fn_vis async fn #fn_name #fn_generics (#(#param_tokens),*) #fn_output
                #fn_block
        }
    } else {
        // sync function
        quote! {
            #(#fn_attrs)*
            #[cfg(feature = "pyo3")]
            #[pyo3::pyfunction]
            #fn_vis fn #fn_name #fn_generics (#(#param_tokens),*) #fn_output
                #fn_block

            #(#fn_attrs)*
            #[cfg(not(feature = "pyo3"))]
            #fn_vis fn #fn_name #fn_generics (#(#param_tokens),*) #fn_output
                #fn_block
        }
    };

    let expanded = quote! {
        #fn_impl
        #registration
    };

    expanded.into()
}

/// Generate the Python stub signature for a function.
/// Returns a string like "def search_ted(query: str, limit: int | None) -> list[Tender]: ..."
fn generate_function_stub_signature(fn_name: &Ident, params: &[&syn::PatType], return_type: &ReturnType) -> String {
    let param_stubs: Vec<String> = params
        .iter()
        .filter_map(|p| {
            // Extract parameter name
            let name = if let Pat::Ident(pat_ident) = p.pat.as_ref() {
                let n = pat_ident.ident.to_string();

                // Ignore 'pÿ́' specific, if the function takes a reference to Python
                if &n == "py" {
                    return None;
                }
                n
            } else {
                return None;
            };

            // Convert type to Python hint
            let py_type = rust_type_to_python_hint(&p.ty);
            Some(format!("{}: {}", name, py_type))
        })
        .collect();

    let return_hint = match return_type {
        ReturnType::Default => "None".to_string(),
        ReturnType::Type(_, ty) => rust_type_to_python_hint(ty),
    };

    format!("def {}({}) -> {}: ...", fn_name, param_stubs.join(", "), return_hint)
}

/// Generate the linkme registration for a function.
fn generate_function_registration(fn_name: &Ident, stub_signature: &str) -> TokenStream2 {
    let registration_ident = format_ident!("__MIMIR_FUNC_REG_{}", fn_name);
    let fn_name_str = fn_name.to_string();

    quote! {
        #[cfg(feature = "pyo3")]
        #[allow(non_upper_case_globals)]
        #[linkme::distributed_slice(crate::MIMIR_FUNCTIONS)]
        #[linkme(crate = linkme)]
        static #registration_ident: crate::MimirFunctionEntry = crate::MimirFunctionEntry {
            rust_module_path: module_path!(),
            function_name: #fn_name_str,
            stub_signature: #stub_signature,
            register: |m: &pyo3::Bound<'_, pyo3::types::PyModule>| -> pyo3::PyResult<()> {
                use pyo3::types::PyModuleMethods;
                use pyo3::wrap_pyfunction;
                m.add_function(wrap_pyfunction!(#fn_name, m)?)
            },
        };
    }
}

// ============================================================================
// String utilities
// ============================================================================

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

// ============================================================================
// Type conversion: Rust -> Python type hints
// ============================================================================

/// Convert a Rust type to a Python type hint string for stub generation.
fn rust_type_to_python_hint(ty: &Type) -> String {
    let type_str = quote!(#ty).to_string();
    convert_type_string(&type_str)
}

/// Convert a stringified Rust type to Python type hint.
fn convert_type_string(s: &str) -> String {
    let s = s.trim();

    // Option<T> -> T | None
    if s.starts_with("Option <") || s.starts_with("Option<") {
        let inner = extract_generic_arg(s, "Option");
        return format!("{} | None", convert_type_string(&inner));
    }

    // Vec<T> -> list[T]
    if s.starts_with("Vec <") || s.starts_with("Vec<") {
        let inner = extract_generic_arg(s, "Vec");
        return format!("list[{}]", convert_type_string(&inner));
    }

    // HashMap<K, V> -> dict[K, V]
    if s.starts_with("HashMap <") || s.starts_with("HashMap<") {
        let inner = extract_generic_arg(s, "HashMap");
        let parts: Vec<&str> = split_generic_args(&inner);
        if parts.len() == 2 {
            return format!(
                "dict[{}, {}]",
                convert_type_string(parts[0]),
                convert_type_string(parts[1])
            );
        }
    }

    // handle (most/all?) common primitive types
    match s {
        "String" | "& str" | "&str" => "str".to_string(),
        "i8" | "i16" | "i32" | "i64" | "i128" | "isize" => "int".to_string(),
        "u8" | "u16" | "u32" | "u64" | "u128" | "usize" => "int".to_string(),
        "f32" | "f64" => "float".to_string(),
        "bool" => "bool".to_string(),
        "()" => "None".to_string(),
        _ => {
            // Handle chrono types
            if s.contains("DateTime") {
                return "datetime".to_string();
            }
            if s.contains("NaiveDate") {
                return "date".to_string();
            }
            // Handle jiff types
            if s.contains("Timestamp") || s.contains("Zoned") {
                return "datetime".to_string();
            }
            // default: use the type name as-is (for custom types)
            // extract just the type name without module path
            s.split("::").last().unwrap_or(s).trim().to_string()
        }
    }
}

/// Extract the generic argument from a type like "Option < T >" or "Vec<T>"
fn extract_generic_arg(s: &str, wrapper: &str) -> String {
    let start = s.find('<').unwrap_or(wrapper.len()) + 1;
    let end = s.rfind('>').unwrap_or(s.len());
    s[start..end].trim().to_string()
}

/// Split generic arguments respecting nested generics.
/// "K, V" -> ["K", "V"]
/// "String, Vec<i32>" -> ["String", "Vec<i32>"]
fn split_generic_args(s: &str) -> Vec<&str> {
    let mut result = Vec::new();
    let mut depth = 0;
    let mut start = 0;

    for (i, c) in s.char_indices() {
        match c {
            '<' => depth += 1,
            '>' => depth -= 1,
            ',' if depth == 0 => {
                result.push(s[start..i].trim());
                start = i + 1;
            }
            _ => {}
        }
    }
    result.push(s[start..].trim());
    result
}

// ============================================================================
// Field processing
// ============================================================================

/// Add serde alias attributes for camelCase deserialization to struct fields.
fn add_serde_aliases(input: &mut DeriveInput) {
    if let Data::Struct(ref mut data) = input.data
        && let Fields::Named(ref mut fields) = data.fields
    {
        for field in fields.named.iter_mut() {
            if let Some(ident) = &field.ident {
                let camel_case_name = snake_to_camel(&ident.to_string());
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
}

/// Extract named fields from a struct, panicking if not a struct with named fields.
fn extract_named_fields(input: &DeriveInput) -> &Punctuated<Field, Comma> {
    match &input.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(fields) => &fields.named,
            _ => panic!("mimir_type only supports structs with named fields"),
        },
        _ => panic!("mimir_type only supports structs"),
    }
}

/// Build function parameters and field names from fields.
fn build_params_and_field_names(fields: &Punctuated<Field, Comma>) -> (Vec<TokenStream2>, Vec<&Option<Ident>>) {
    let params = fields
        .iter()
        .map(|f| {
            let name = &f.ident;
            let ty = &f.ty;
            quote! { #name: #ty }
        })
        .collect();

    let field_names = fields.iter().map(|f| &f.ident).collect();

    (params, field_names)
}

/// Generate stub info string for a struct's fields.
/// Format: "field_name:python_type;field_name2:python_type2"
fn generate_stub_fields_info(fields: &Punctuated<Field, Comma>) -> String {
    fields
        .iter()
        .filter_map(|f| {
            f.ident.as_ref().map(|name| {
                let py_type = rust_type_to_python_hint(&f.ty);
                format!("{}:{}", name, py_type)
            })
        })
        .collect::<Vec<_>>()
        .join(";")
}

// ============================================================================
// Code generation
// ============================================================================

/// Generate the struct definition with all necessary derive macros and attributes.
fn generate_struct_definition(input: &DeriveInput, name: &Ident) -> TokenStream2 {
    quote! {
        #[cfg_attr(feature = "pyo3", pyo3::pyclass(get_all))]
        #[derive(ts_rs::TS, Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
        #[ts(export, export_to = format!("{}/{}.ts", module_path!().replace("::", "/"), stringify!(#name)))]
        #input
    }
}

/// Generate the TypeBehavior trait implementation.
fn generate_type_behavior_impl(name: &Ident) -> TokenStream2 {
    quote! {
        impl crate::TypeBehavior for #name {}
    }
}

// ============================================================================
// Common PyO3 methods - Single source of truth for impl AND stubs
// ============================================================================

/// Definition of a common method available on all mimir types.
/// Each method defines both its Rust implementation and Python stub signature.
struct CommonMethod {
    /// Python stub signature (e.g., "def to_dict(self) -> dict[str, Any]: ...")
    stub: &'static str,
    /// Function that generates the Rust implementation tokens
    impl_tokens: fn() -> TokenStream2,
}

/// All common methods available on mimir types.
/// This is the SINGLE SOURCE OF TRUTH - add new methods here and both
/// the Rust implementation and Python stubs will be generated automatically.
fn common_methods() -> Vec<CommonMethod> {
    vec![
        CommonMethod {
            stub: "def to_dict(self) -> dict[str, Any]: ...",
            impl_tokens: || {
                quote! {
                    /// Convert to a Python dict
                    pub fn to_dict<'py>(
                        &self,
                        py: pyo3::prelude::Python<'py>
                    ) -> pyo3::prelude::PyResult<pyo3::prelude::Bound<'py, pyo3::prelude::PyAny>> {
                        let dict = pythonize::pythonize(py, &self)?;
                        Ok(dict)
                    }
                }
            },
        },
        CommonMethod {
            stub: "@classmethod\n    def from_dict(cls, obj: dict[str, Any] | Self) -> Self: ...",
            impl_tokens: || {
                quote! {
                    /// Create from Python dict (or from an existing instance)
                    #[classmethod]
                    pub fn from_dict(
                        _cls: &pyo3::Bound<'_, pyo3::types::PyType>,
                        obj: &pyo3::Bound<'_, pyo3::prelude::PyAny>
                    ) -> pyo3::prelude::PyResult<Self> {
                        use pyo3::prelude::*;

                        // If the input is already an instance of Self, just clone it
                        if let Ok(slf) = obj.extract::<Self>() {
                            return Ok(slf);
                        }

                        let py = obj.py();
                        let normalized = crate::pyutil::normalize_datetimes(py, obj)?;

                        let result: Self = crate::pyutil::depythonize_with_path(&normalized)
                            .map_err(|e| {
                                pyo3::exceptions::PyValueError::new_err(format!(
                                    "Invalid structure at `{}`: {}",
                                    e.path(),
                                    e.inner()
                                ))
                            })?;

                        Ok(result)
                    }
                }
            },
        },
        CommonMethod {
            stub: "def to_json(self) -> str: ...",
            impl_tokens: || {
                quote! {
                    /// Convert to JSON string
                    pub fn to_json(&self) -> pyo3::prelude::PyResult<String> {
                        let json = crate::TypeBehavior::to_json(self)
                            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
                        Ok(json)
                    }
                }
            },
        },
        CommonMethod {
            stub: "@classmethod\n    def from_json(cls, json: str) -> Self: ...",
            impl_tokens: || {
                quote! {
                    /// Create from JSON string
                    #[classmethod]
                    pub fn from_json(
                        _cls: &pyo3::Bound<'_, pyo3::types::PyType>,
                        json: &str
                    ) -> pyo3::prelude::PyResult<Self> {
                        <Self as crate::TypeBehavior>::from_json(json.to_owned())
                            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
                    }
                }
            },
        },
        CommonMethod {
            stub: "def __eq__(self, other: Self) -> bool: ...",
            impl_tokens: || {
                quote! {
                    /// Compare to another instance
                    pub fn __eq__(&self, other: &Self) -> bool {
                        self == other
                    }
                }
            },
        },
        CommonMethod {
            stub: "def __str__(self) -> str: ...",
            impl_tokens: || {
                quote! {
                    /// Debug representation
                    pub fn __str__(&self) -> String {
                        format!("{:?}", self)
                    }
                }
            },
        },
    ]
}

/// Generate the combined stub methods string from all common methods.
fn generate_stub_methods() -> String {
    common_methods()
        .iter()
        .map(|m| format!("    {}", m.stub))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Generate the PyO3 pymethods implementation block.
fn generate_pymethods_impl(name: &Ident, params: &[TokenStream2], field_names: &[&Option<Ident>]) -> TokenStream2 {
    // generate tokens for all common methods
    let method_impls: Vec<TokenStream2> = common_methods().iter().map(|m| (m.impl_tokens)()).collect();

    // also add the default init method
    quote! {
        #[cfg(feature = "pyo3")]
        #[pyo3::pymethods]
        impl #name {
            #[new]
            #[allow(clippy::too_many_arguments)]
            pub fn __init__(#(#params),*) -> Self {
                Self { #(#field_names),* }
            }

            #(#method_impls)*
        }
    }
}

/// Generate the Rust-facing impl block with new() constructor.
fn generate_rust_impl(name: &Ident, params: &[TokenStream2], field_names: &[&Option<Ident>]) -> TokenStream2 {
    quote! {
        impl #name {
            #[allow(clippy::too_many_arguments)]
            pub fn new(#(#params),*) -> Self {
                Self { #(#field_names),* }
            }
        }
    }
}

/// Generate the linkme registration for automatic PyO3 module building.
fn generate_registration(name: &Ident, stub_fields: &str, stub_methods: &str) -> TokenStream2 {
    let registration_ident = format_ident!("__MIMIR_TYPE_REG_{}", name);

    quote! {
        #[cfg(feature = "pyo3")]
        #[allow(non_upper_case_globals)]
        #[linkme::distributed_slice(crate::MIMIR_TYPES)]
        #[linkme(crate = linkme)]
        static #registration_ident: crate::MimirTypeEntry = crate::MimirTypeEntry {
            rust_module_path: module_path!(),
            type_name: stringify!(#name),
            stub_fields: #stub_fields,
            stub_methods: #stub_methods,
            register: |m: &pyo3::Bound<'_, pyo3::types::PyModule>| -> pyo3::PyResult<()> {
                use pyo3::types::PyModuleMethods;
                m.add_class::<#name>()
            },
        };
    }
}
