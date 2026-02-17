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
    Data, DeriveInput, Field, Fields, FnArg, Ident, ItemFn, Lit, Meta, Pat, ReturnType, Type, parse_macro_input,
    punctuated::Punctuated, token::Comma,
};

// ============================================================================
// Attribute parsing
// ============================================================================

/// Parsed arguments from `#[mimir_type(dbt_model = "name", dbt_source = "table")]`
struct MimirTypeArgs {
    dbt_model: Option<String>,
    dbt_primary_key: Option<String>,
    dbt_source: Option<String>,
}

impl MimirTypeArgs {
    fn parse(attr: TokenStream) -> Self {
        let mut args = MimirTypeArgs {
            dbt_model: None,
            dbt_primary_key: None,
            dbt_source: None,
        };

        if attr.is_empty() {
            return args;
        }

        let parsed = syn::parse::Parser::parse(Punctuated::<Meta, Comma>::parse_terminated, attr)
            .expect("Failed to parse mimir_type attributes");

        for meta in parsed {
            if let Meta::NameValue(nv) = meta {
                let key = nv.path.get_ident().map(|i| i.to_string());
                if let syn::Expr::Lit(syn::ExprLit { lit: Lit::Str(val), .. }) = &nv.value {
                    match key.as_deref() {
                        Some("dbt_model") => args.dbt_model = Some(val.value()),
                        Some("dbt_primary_key") => args.dbt_primary_key = Some(val.value()),
                        Some("dbt_source") => args.dbt_source = Some(val.value()),
                        Some(other) => panic!("Unknown mimir_type attribute: {other}"),
                        None => {}
                    }
                }
            }
        }

        args
    }
}

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
pub fn mimir_type(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = MimirTypeArgs::parse(attr);
    let mut input = parse_macro_input!(item as DeriveInput);
    let name = input.ident.clone();

    // add serde aliases for camelCase field names
    add_serde_aliases(&mut input);

    // extract struct-level doc comment for schema metadata
    let struct_doc = extract_doc_comment(&input.attrs);

    // extract fields and build params
    let fields = extract_named_fields(&input);
    let (params, field_names) = build_params_and_field_names(fields);
    let stub_fields = generate_stub_fields_info(fields);
    let stub_methods = generate_stub_methods();
    let arrow_fields = generate_arrow_fields(fields);

    // generate all code blocks
    let struct_def = generate_struct_definition(&input, &name);
    let type_behavior = generate_type_behavior_impl(&name);
    let pymethods = generate_pymethods_impl(&name, &field_names, &arrow_fields, &args);
    let rust_impl = generate_rust_impl(&name, &params, &field_names, &arrow_fields, &struct_doc);
    let registration = generate_registration(&name, &stub_fields, &stub_methods, &args);

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

/// Strip the `r#` prefix from raw identifiers.
/// In Rust, `r#type` is used to use reserved keywords as identifiers,
/// but the prefix should not appear in generated output (Arrow schemas, Python stubs, serde aliases).
fn strip_raw_prefix(s: &str) -> String {
    s.strip_prefix("r#").unwrap_or(s).to_string()
}

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
// Type conversion: Rust -> Arrow DataType
// ============================================================================

/// Convert a Rust type to Arrow DataType tokens for schema generation.
/// Returns (datatype_tokens, nullable) where nullable indicates if the type is Option<T>.
fn rust_type_to_arrow_datatype(ty: &Type) -> (TokenStream2, bool) {
    let type_str = quote!(#ty).to_string();
    convert_to_arrow_datatype(&type_str)
}

/// Convert a stringified Rust type to Arrow DataType tokens.
/// Returns (datatype_tokens, nullable).
fn convert_to_arrow_datatype(s: &str) -> (TokenStream2, bool) {
    convert_to_arrow_datatype_inner(s, false)
}

/// Inner conversion function that tracks whether we're inside an Option context.
/// When inside_option is true, nested struct fields should be made nullable.
fn convert_to_arrow_datatype_inner(s: &str, inside_option: bool) -> (TokenStream2, bool) {
    let s = s.trim();

    // Option<T> -> inner type with nullable=true, and mark that we're inside an Option
    if s.starts_with("Option <") || s.starts_with("Option<") {
        let inner = extract_generic_arg(s, "Option");
        let (inner_tokens, _) = convert_to_arrow_datatype_inner(&inner, true);
        return (inner_tokens, true);
    }

    // Vec<T> -> List(T)
    if s.starts_with("Vec <") || s.starts_with("Vec<") {
        let inner = extract_generic_arg(s, "Vec");
        let (inner_tokens, inner_nullable) = convert_to_arrow_datatype_inner(&inner, inside_option);
        return (
            quote! {
                arrow_schema::DataType::List(
                    std::sync::Arc::new(arrow_schema::Field::new("item", #inner_tokens, #inner_nullable))
                )
            },
            false,
        );
    }

    // HashMap<K, V> -> Map(K, V)
    if s.starts_with("HashMap <") || s.starts_with("HashMap<") {
        let inner = extract_generic_arg(s, "HashMap");
        let parts: Vec<&str> = split_generic_args(&inner);
        if parts.len() == 2 {
            let (key_tokens, _) = convert_to_arrow_datatype_inner(parts[0], inside_option);
            let (value_tokens, value_nullable) = convert_to_arrow_datatype_inner(parts[1], inside_option);
            return (
                quote! {
                    arrow_schema::DataType::Map(
                        std::sync::Arc::new(arrow_schema::Field::new(
                            "entries",
                            arrow_schema::DataType::Struct(
                                arrow_schema::Fields::from(vec![
                                    arrow_schema::Field::new("key", #key_tokens, false),
                                    arrow_schema::Field::new("value", #value_tokens, #value_nullable),
                                ])
                            ),
                            false
                        )),
                        false // keys_sorted
                    )
                },
                false,
            );
        }
    }

    // Primitive types
    // Note: We use LargeUtf8 to match Polars' default string representation
    let tokens = match s {
        "String" | "& str" | "&str" => quote! { arrow_schema::DataType::LargeUtf8 },
        "i8" => quote! { arrow_schema::DataType::Int8 },
        "i16" => quote! { arrow_schema::DataType::Int16 },
        "i32" => quote! { arrow_schema::DataType::Int32 },
        "i64" | "isize" => quote! { arrow_schema::DataType::Int64 },
        "i128" => quote! { arrow_schema::DataType::Decimal128(38, 0) },
        "u8" => quote! { arrow_schema::DataType::UInt8 },
        "u16" => quote! { arrow_schema::DataType::UInt16 },
        "u32" => quote! { arrow_schema::DataType::UInt32 },
        "u64" | "usize" => quote! { arrow_schema::DataType::UInt64 },
        "f32" => quote! { arrow_schema::DataType::Float32 },
        "f64" => quote! { arrow_schema::DataType::Float64 },
        "bool" => quote! { arrow_schema::DataType::Boolean },
        "()" => quote! { arrow_schema::DataType::Null },
        _ => {
            // Handle chrono/jiff datetime types
            if s.contains("NaiveDateTime") {
                quote! { arrow_schema::DataType::Timestamp(arrow_schema::TimeUnit::Microsecond, None) }
            } else if s.contains("DateTime") || s.contains("Timestamp") || s.contains("Zoned") {
                quote! { arrow_schema::DataType::Timestamp(arrow_schema::TimeUnit::Microsecond, Some("UTC".into())) }
            } else if s.contains("NaiveDate") {
                quote! { arrow_schema::DataType::Date32 }
            } else {
                // For nested mimir types, call their get_arrow_schema() to build a Struct type
                // Extract just the type name (without module path)
                let type_name = s.split("::").last().unwrap_or(s).trim();
                let type_ident = format_ident!("{}", type_name);

                // When inside an Option, make all nested struct fields nullable
                // because when the Option is None, all inner fields will be null
                if inside_option {
                    quote! {
                        arrow_schema::DataType::Struct(
                            arrow_schema::Fields::from(
                                #type_ident::get_arrow_schema()
                                    .fields()
                                    .iter()
                                    .map(|f| arrow_schema::Field::new(f.name(), f.data_type().clone(), true))
                                    .collect::<Vec<_>>()
                            )
                        )
                    }
                } else {
                    quote! {
                        arrow_schema::DataType::Struct(
                            #type_ident::get_arrow_schema().fields().clone()
                        )
                    }
                }
            }
        }
    };

    (tokens, false)
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
                let field_name = strip_raw_prefix(&ident.to_string());
                let camel_case_name = snake_to_camel(&field_name);
                #[allow(clippy::cmp_owned)]
                if camel_case_name != field_name {
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
                format!("{}:{}", strip_raw_prefix(&name.to_string()), py_type)
            })
        })
        .collect::<Vec<_>>()
        .join(";")
}

/// Extract the doc comment from `/// ...` attributes.
fn extract_doc_comment(attrs: &[syn::Attribute]) -> Option<String> {
    let doc_lines: Vec<String> = attrs
        .iter()
        .filter_map(|attr| {
            // name value, value is a string literal and attr is doc (ie #[doc = "foo"])
            if let Meta::NameValue(nv) = &attr.meta
                && let syn::Expr::Lit(syn::ExprLit { lit: Lit::Str(val), .. }) = &nv.value
                && attr.path().is_ident("doc")
            {
                return Some(val.value().trim().to_string());
            }
            None
        })
        .filter(|s| !s.is_empty())
        .collect();

    if doc_lines.is_empty() {
        None
    } else {
        // multiple doc comment lines are joined with spaces after trimming
        Some(doc_lines.join(" "))
    }
}

/// Generate Arrow field construction tokens for a struct's fields.
/// Doc comments on fields are attached as Arrow field metadata under the "description" key.
fn generate_arrow_fields(fields: &Punctuated<Field, Comma>) -> Vec<TokenStream2> {
    fields
        .iter()
        .filter_map(|f| {
            f.ident.as_ref().map(|name| {
                let name_str = strip_raw_prefix(&name.to_string());
                let (datatype_tokens, nullable) = rust_type_to_arrow_datatype(&f.ty);

                match extract_doc_comment(&f.attrs) {
                    Some(desc) => quote! {
                        arrow_schema::Field::new(#name_str, #datatype_tokens, #nullable)
                            .with_metadata(
                                std::collections::HashMap::from([
                                    // place doc/description into description key of metadata
                                    ("description".to_string(), #desc.to_string())
                                ])
                            )
                    },
                    None => quote! {
                        arrow_schema::Field::new(#name_str, #datatype_tokens, #nullable)
                    },
                }
            })
        })
        .collect()
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
    let common_stubs: Vec<String> = common_methods().iter().map(|m| format!("    {}", m.stub)).collect();

    // Add arrow schema and record batch stubs
    let arrow_stubs = vec![
        "    @classmethod\n    def arrow_schema(cls) -> \"pyarrow.Schema\": ...".to_string(),
        "    @classmethod\n    def to_record_batch(cls, items: list[Self]) -> \"pyarrow.RecordBatch\": ...".to_string(),
        "    @classmethod\n    def from_record_batch(cls, batch: \"pyarrow.RecordBatch\") -> list[Self]: ..."
            .to_string(),
        "    @classmethod\n    def iceberg_table(cls) -> str | None: ...".to_string(),
    ];

    [common_stubs, arrow_stubs].concat().join("\n")
}

/// Generate the PyO3 pymethods implementation block.
fn generate_pymethods_impl(
    name: &Ident,
    field_names: &[&Option<Ident>],
    _arrow_fields: &[TokenStream2],
    args: &MimirTypeArgs,
) -> TokenStream2 {
    // generate tokens for all common methods
    let method_impls: Vec<TokenStream2> = common_methods().iter().map(|m| (m.impl_tokens)()).collect();

    // Build __init__ params (all Bound<PyAny>) and dict insertion statements.
    // __init__ builds a dict and delegates to normalize_datetimes + depythonize,
    // reusing the same path as from_dict. This handles datetime objects with any
    // tzinfo (zoneinfo, timezone.utc, etc.) and nested mimir type instances.
    let init_field_idents: Vec<&Ident> = field_names.iter().map(|n| n.as_ref().unwrap()).collect();
    let init_field_name_strs: Vec<String> = init_field_idents
        .iter()
        .map(|id| strip_raw_prefix(&id.to_string()))
        .collect();

    // Generate iceberg_table() classmethod - per-type since it depends on dbt_model value
    let iceberg_table_impl = match &args.dbt_model {
        Some(model) => quote! {
            /// Returns the iceberg table identifier for this type,
            /// ie "datalake.processed.norce.profiles"
            #[classmethod]
            pub fn iceberg_table(
                _cls: &pyo3::Bound<'_, pyo3::types::PyType>,
            ) -> Option<String> {
                let module = module_path!();
                let stripped = module.strip_prefix("mimirtypes::").unwrap_or(module);
                let namespace = stripped
                    .rsplit_once("::")
                    .map(|(ns, _)| ns.replace("::", "."))
                    .unwrap_or_default();
                Some(format!("{}.{}", namespace, #model))
            }
        },
        None => quote! {
            /// Return None, this type does not map to a dbt created iceberg table
            #[classmethod]
            pub fn iceberg_table(
                _cls: &pyo3::Bound<'_, pyo3::types::PyType>,
            ) -> Option<String> {
                None
            }
        },
    };

    quote! {
        #[cfg(feature = "pyo3")]
        #[pyo3::pymethods]
        impl #name {
            #[new]
            #[allow(clippy::too_many_arguments)]
            pub fn __init__(
                py: pyo3::prelude::Python<'_>,
                #(#init_field_idents: pyo3::Bound<'_, pyo3::prelude::PyAny>),*
            ) -> pyo3::prelude::PyResult<Self> {
                use pyo3::types::PyDictMethods;
                let dict = pyo3::types::PyDict::new(py);
                #(dict.set_item(#init_field_name_strs, &#init_field_idents)?;)*
                let normalized = crate::pyutil::normalize_datetimes(py, &dict.clone().into_any())?;
                crate::pyutil::depythonize_with_path(&normalized)
                    .map_err(|e| {
                        pyo3::exceptions::PyValueError::new_err(format!(
                            "Invalid structure at `{}`: {}",
                            e.path(),
                            e.inner()
                        ))
                    })
            }

            #(#method_impls)*

            #[classmethod]
            pub fn arrow_schema<'py>(
                _cls: &pyo3::Bound<'py, pyo3::types::PyType>,
                py: pyo3::prelude::Python<'py>,
            ) -> pyo3::prelude::PyResult<pyo3::prelude::Bound<'py, pyo3::prelude::PyAny>> {
                let schema = Self::get_arrow_schema();
                ::pyo3_arrow::PySchema::new(std::sync::Arc::new(schema)).into_pyarrow(py)
            }

            #[classmethod]
            pub fn to_record_batch<'py>(
                _cls: &pyo3::Bound<'py, pyo3::types::PyType>,
                py: pyo3::prelude::Python<'py>,
                items: Vec<Self>,
            ) -> pyo3::prelude::PyResult<pyo3::prelude::Bound<'py, pyo3::prelude::PyAny>> {
                let fields: Vec<std::sync::Arc<arrow_schema::Field>> = Self::get_arrow_schema()
                    .fields()
                    .iter()
                    .cloned()
                    .collect();

                let batch = serde_arrow::to_record_batch(&fields, &items)
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

                ::pyo3_arrow::PyRecordBatch::new(batch).into_pyarrow(py)
            }

            #[classmethod]
            pub fn from_record_batch(
                _cls: &pyo3::Bound<'_, pyo3::types::PyType>,
                batch: ::pyo3_arrow::PyRecordBatch,
            ) -> pyo3::prelude::PyResult<Vec<Self>> {
                serde_arrow::from_record_batch(&batch.into_inner())
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
            }

            #iceberg_table_impl
        }
    }
}

/// Generate the Rust-facing impl block with new() constructor.
fn generate_rust_impl(
    name: &Ident,
    params: &[TokenStream2],
    field_names: &[&Option<Ident>],
    arrow_fields: &[TokenStream2],
    struct_doc: &Option<String>,
) -> TokenStream2 {
    let schema_constructor = match struct_doc {
        Some(desc) => quote! {
            arrow_schema::Schema::new_with_metadata(
                vec![#(#arrow_fields),*],
                std::collections::HashMap::from([
                    ("description".to_string(), #desc.to_string())
                ]),
            )
        },
        None => quote! {
            arrow_schema::Schema::new(vec![
                #(#arrow_fields),*
            ])
        },
    };

    quote! {
        impl #name {
            #[allow(clippy::too_many_arguments)]
            pub fn new(#(#params),*) -> Self {
                Self { #(#field_names),* }
            }
        }

        // Arrow schema method - available in both pyo3 and non-pyo3 builds
        // For pyo3 builds, use the arrow_schema() classmethod instead for Python interop
        impl #name {
            /// Get the Arrow schema for this type.
            pub fn get_arrow_schema() -> arrow_schema::Schema {
                #schema_constructor
            }
        }
    }
}

/// Generate the linkme registration for automatic PyO3 module building.
fn generate_registration(name: &Ident, stub_fields: &str, stub_methods: &str, args: &MimirTypeArgs) -> TokenStream2 {
    let registration_ident = format_ident!("__MIMIR_TYPE_REG_{}", name);

    let dbt_model_tokens = match &args.dbt_model {
        Some(model) => quote! { Some(#model) },
        None => quote! { None },
    };
    let dbt_primary_key_tokens = match &args.dbt_primary_key {
        Some(key) => quote! { Some(#key) },
        None => quote! { None },
    };
    let dbt_source_tokens = match &args.dbt_source {
        Some(source) => quote! { Some(#source) },
        None => quote! { None },
    };

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
            dbt_model: #dbt_model_tokens,
            dbt_primary_key: #dbt_primary_key_tokens,
            dbt_source: #dbt_source_tokens,
            get_schema: <#name>::get_arrow_schema,
            register: |m: &pyo3::Bound<'_, pyo3::types::PyModule>| -> pyo3::PyResult<()> {
                use pyo3::types::PyModuleMethods;
                m.add_class::<#name>()
            },
        };
    }
}
