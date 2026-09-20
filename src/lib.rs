//! [![github]](https://github.com/tamaskis/numdiff)&ensp;[![crates-io]](https://crates.io/crates/numdiff)&ensp;[![docs-rs]](https://docs.rs/numdiff)
//!
//! [github]: https://img.shields.io/badge/github-8da0cb?style=for-the-badge&labelColor=555555&logo=github
//! [crates-io]: https://img.shields.io/badge/crates.io-fc8d62?style=for-the-badge&labelColor=555555&logo=rust
//! [docs-rs]: https://img.shields.io/badge/docs.rs-66c2a5?style=for-the-badge&labelColor=555555&logo=docs.rs
//!
//! Automatic and numerical differentiation.
//!
//! # Overview
//!
//! This crate implements two different methods for evaluating derivatives in Rust:
//!
//! 1. Automatic differentiation (forward-mode using first-order dual numbers).
//! 2. Numerical differentiation (using forward difference and central difference approximations).
//!
//! This crate provides generic functions (for numerical differentiation) and macros (for automatic
//! differentiation) to evaluate various types of derivatives of the following types of functions:
//!
//! * Univariate, scalar-valued functions ($f:\mathbb{R}\to\mathbb{R}$)
//! * Univariate, vector-valued functions ($\mathbf{f}:\mathbb{R}\to\mathbb{R}^{m}$)
//! * Multivariate, scalar-valued functions ($f:\mathbb{R}^{n}\to\mathbb{R}$)
//! * Multivariate, vector-valued functions ($\mathbf{f}:\mathbb{R}^{n}\to\mathbb{R}^{m}$)
//!
//! These functions and macros are made generic over the choice of vector representation, as long as
//! the vector type implements the `linalg_traits::Vector` trait. See the
//! [`linalg_traits` documentation](https://docs.rs/linalg-traits/latest/linalg_traits/) for more
//! information.
//!
//! # Automatic Differentiation (Forward-Mode)
//!
//! This crate provides forward-mode automatic differentiation of functions that are generic over
//! their vector and scalar types.
//!
//! Typing `R: linalg_traits::RealField` allows the automatic differentiation macros to substitute
//! this crate's [`Dual`] and [`HyperDual`] types, which allow propagating 1st and 2nd-order
//! derivative information through the computation. The resulting way functions have to be written
//! differs mainly from a more "standard" numerical computing by requiring concrete types such as
//! `f64` and `SVector<f64>` to be written in terms of generic types such as
//! `R: linalg_traits::RealField` and `V: linalg_traits::Vector<R>`. It is, however, a close
//! approximation to the standard Rust numeric style: the
//! [`linalg_traits::RealField`](https://docs.rs/linalg-traits/latest/linalg_traits/trait.RealField.html)
//! requires a superset of the functionality provided by common traits such as `num_traits::Float`
//! and `nalgebra::RealField`, while additionally requiring a high degree of interoperability with
//! `f64`. In that sense, this crate aims to stay near the ergonomics of normal Rust code without
//! forcing a custom AD type at every call site.
//!
//! A major benefit of this approach is that the same function can be evaluated with an `f64`,
//! allowing it to be used both in ordinary numerical computations and in automatic differentiation
//! contexts. As an example, consider a function operating with conrete `f64`'s:
//!
//! ```rust
//! fn f_f64(x: f64) -> f64 {
//!     x * x.sin() + 1.0
//! }
//! ```
//!
//! To write this function in a generic way suitable for automatic differentiation:
//!
//! ```
//! use linalg_traits::RealField;
//!
//! fn f_generic<R: RealField>(x: R) -> R {
//!     x * x.sin() + 1.0
//! }
//! ```
//!
//! `f_generic` can be called with an `f64` just like `f_f64`, but it can also be called with a
//! [`Dual`] or [`HyperDual`] type, which is the special sauce that allows this function to be
//! automatically differentiated with one of the derivative macros provided by this crate.
//!
//! ## 1st-Order Derivatives
//!
//! | Derivative Type | Function Type | Macro to Generate Derivative Function |
//! | --------------- | ------------- | ------------------------------------- |
//! | derivative | $f:\mathbb{R}\to\mathbb{R}$ | [`get_sderivative!`] |
//! | derivative | $\mathbf{f}:\mathbb{R}\to\mathbb{R}^{m}$ | [`get_vderivative!`] |
//! | partial derivative | $f:\mathbb{R}^{n}\to\mathbb{R}$ | [`get_spartial_derivative!`] |
//! | partial derivative | $\mathbf{f}:\mathbb{R}^{n}\to\mathbb{R}^{m}$ | [`get_vpartial_derivative!`] |
//! | gradient | $f:\mathbb{R}^{n}\to\mathbb{R}$ | [`get_gradient!`] |
//! | directional derivative | $f:\mathbb{R}^{n}\to\mathbb{R}$ | [`get_directional_derivative!`] |
//! | Jacobian| $\mathbf{f}:\mathbb{R}^{n}\to\mathbb{R}^{m}$ | [`get_jacobian!`] |
//!
//! ## 2nd-Order Derivatives
//!
//! | Derivative Type | Function Type | Macro to Generate Derivative Function |
//! | --------------- | ------------- | ------------------------------------- |
//! | 2nd derivative | $f:\mathbb{R}\to\mathbb{R}$ | [`get_sderivative2!`] |
//! | 2nd derivative | $\mathbf{f}:\mathbb{R}\to\mathbb{R}^{m}$ | [`get_vderivative2!`] |
//! | 2nd partial derivative | $f:\mathbb{R}^{n}\to\mathbb{R}$ | [`get_spartial_derivative2!`] |
//! | 2nd partial derivative | $\mathbf{f}:\mathbb{R}^{n}\to\mathbb{R}^{m}$ | [`get_vpartial_derivative2!`] |
//! | mixed 2nd partial derivative | $f:\mathbb{R}^{n}\to\mathbb{R}$ | [`get_mixed_spartial_derivative2!`] |
//! | mixed 2nd partial derivative | $\mathbf{f}:\mathbb{R}^{n}\to\mathbb{R}^{m}$ | [`get_mixed_vpartial_derivative2!`] |
//! | Hessian | $f:\mathbb{R}^{n}\to\mathbb{R}$ | [`get_shessian!`] |
//! | Hessian | $\mathbf{f}:\mathbb{R}^{n}\to\mathbb{R}^{m}$ | [`get_vhessian!`] |
//!
//! ## Passing Runtime Parameters
//!
//! Many times, we want to automatically differentiate functions that can also depend on parameters
//! defined at runtime. However, automatic differentiation is performed at compile time, so we
//! cannot simply "capture" these parameters using closures. To solve this problem, all automatic
//! differentiation macros expect the function being differentiated to not only accept the point at
//! which it is differentiated, but also a runtime parameter of an arbitrary type (could be a
//! `&[f64]`, could be a `&T` where `T` is some custom struct, etc.).
//!
//! Examples are included for each macro (for example, here is the example for the
//! [`get_jacobian!`] macro):
//! [`get_jacobian!` - Example Passing Custom Parameter Types](macro@get_jacobian#example-passing-custom-parameter-types)
//!
//! ## Limitations
//!
//! * These macros only work on functions that are generic both over the type of scalar and the type
//!   of vector.
//!     - Consequently, these macros do _not_ work on closures.
//! * Occasionally, constants (e.g. `5.0_f64`) need to be defined using `R::from` (e.g. if a
//!   function has the generic parameter `R: RealField`, then instead of defining a constant number
//!   such as `5.0_f64`, we need to do `5.0`).
//!     - However, this is not always required, for example if you are doing something like
//!       `5.0 * x` where `x: R` and `R: RealField`, then you don't have to do `5.0 * x`
//!       because `RealField` requires that a type is interoperable with `f64`.
//! * We cannot do assignment operations on `f64` where right hand side is an `x: R` where
//!   `R: RealField` (e.g. `let y = 1.0; y += x;`).
//!     - In cases like this you should do `let y = 1.0; y += x;`.
//!
//! ## Alternatives
//!
//! The Rust ecosystem has many different crates and approaches for automatic differentiation. Here
//! we summarize some of the alternatives.
//!
//! ### Standard Library
//!
//! The most prominent alternative nowadays is the standard library's
//! [`std::autodiff`](https://doc.rust-lang.org/std/autodiff/index.html) module which is in
//! `nightly` Rust.
//!
//! This module provides low-level AD capabilities, but is not yet available in stable Rust. If it
//! were, this crate perhaps would have been written as a convenience wrapper around that
//! lower-level API.
//!
//! ### Forward-mode implementations
//!
//! These crates are broadly similar in computational model to this one, but they tend to require a
//! custom numeric type or library-specific expression style rather than a more generic,
//! backend-agnostic (i.e. not tied to a specific linear algebra library such as `nalgebra`).
//!
//! * [`num-dual`](https://docs.rs/num-dual/latest/num_dual/) — similar to this approach, but uses a
//!   `DualNum` trait instead of a more generic-sounding `RealField`, and is only compatible with
//!   `nalgebra` types.
//! * [`autodiff`](https://github.com/elrnv/autodiff) — a solid forward-mode AD crate, but centered
//!   on custom types such as [`FT<T>`](https://docs.rs/autodiff/latest/autodiff/forward_autodiff/type.FT.html)
//!   instead of a generic scalar interface.
//! * [`autodj`](https://docs.rs/autodj/latest/autodj/) — works through custom AD types, but writing
//!   autodifferentiable mathematical functions is more cumbersome, especially for multivariate
//!   cases.
//! * [`fwd_ad`](https://crates.io/crates/fwd_ad) — another crate using forward-mode AD, but
//!   requires significant changes to make functions autodifferentiable.
//! * [`kophy/autodiff`](https://github.com/kophy/autodiff) — only provides a very minimal `Dual`
//!   number implementation, and is unpublished.
//!
//! ### Reverse-mode implementations
//!
//! All three of these libraries implement reverse-mode AD, but require users to use a significantly
//! different syntax when writing autodifferentiable functions:
//!
//! * [`gad`](https://github.com/facebookresearch/gad)
//! * [`reverse`](https://github.com/al-jshen/reverse)
//! * [`revad`](https://github.com/Rufflewind/revad)
//! * [`rustydiff`](https://github.com/Janko-dev/rustydiff)
//! * [`rust-autograd`](https://github.com/raskr/rust-autograd)
//!
//! Note that the first four crates listed here also have not been maintained for 3+ years (some
//! much longer), with at least one of them ([`gad`](https://github.com/facebookresearch/gad))
//! having already been archived.
//!
//! # Finite Difference Methods
//!
//! ## Central Difference Approximations
//!
//! ### First-Order Derivatives
//!
//! | Derivative Type | Function Type | Function to Approximate Derivative |
//! | --------------- | ------------- | ---------------------------------- |
//! | derivative | $f:\mathbb{R}\to\mathbb{R}$ | [`central_difference::sderivative()`] |
//! | derivative | $\mathbf{f}:\mathbb{R}\to\mathbb{R}^{m}$ | [`central_difference::vderivative()`] |
//! | partial derivative | $f:\mathbb{R}^{n}\to\mathbb{R}$ | [`central_difference::spartial_derivative()`] |
//! | partial derivative | $\mathbf{f}:\mathbb{R}^{n}\to\mathbb{R}^{m}$ | [`central_difference::vpartial_derivative()`] |
//! | gradient | $f:\mathbb{R}^{n}\to\mathbb{R}$ | [`central_difference::gradient()`] |
//! | directional derivative | $f:\mathbb{R}^{n}\to\mathbb{R}$ | [`central_difference::directional_derivative()`] |
//! | Jacobian | $\mathbf{f}:\mathbb{R}^{n}\to\mathbb{R}^{m}$ | [`central_difference::jacobian()`] |
//!
//! ### Second-Order Derivatives
//!
//! | Derivative Type | Function Type | Function to Approximate Derivative |
//! | --------------- | ------------- | ---------------------------------- |
//! | Hessian | $f:\mathbb{R}^{n}\to\mathbb{R}$ | [`central_difference::shessian()`] |
//! | Hessian | $\mathbf{f}:\mathbb{R}^{n}\to\mathbb{R}^{m}$ | [`central_difference::vhessian()`] |
//! | 2nd derivative | $f:\mathbb{R}\to\mathbb{R}$ | [`central_difference::sderivative2()`] |
//! | 2nd derivative | $\mathbf{f}:\mathbb{R}\to\mathbb{R}^{m}$ | [`central_difference::vderivative2()`] |
//! | 2nd partial derivative | $f:\mathbb{R}^{n}\to\mathbb{R}$ | [`central_difference::spartial_derivative2()`] |
//! | 2nd partial derivative | $\mathbf{f}:\mathbb{R}^{n}\to\mathbb{R}^{m}$ | [`central_difference::vpartial_derivative2()`] |
//! | mixed 2nd partial derivative | $f:\mathbb{R}^{n}\to\mathbb{R}$ | [`central_difference::mixed_spartial_derivative2()`] |
//! | mixed 2nd partial derivative | $\mathbf{f}:\mathbb{R}^{n}\to\mathbb{R}^{m}$ | [`central_difference::mixed_vpartial_derivative2()`] |
//!
//! ## Forward Difference Approximations
//!
//! ### First-Order Derivatives
//!
//! | Derivative Type | Function Type | Function to Approximate Derivative |
//! | --------------- | ------------- | ---------------------------------- |
//! | derivative | $f:\mathbb{R}\to\mathbb{R}$ | [`forward_difference::sderivative()`] |
//! | derivative | $\mathbf{f}:\mathbb{R}\to\mathbb{R}^{m}$ | [`forward_difference::vderivative()`] |
//! | partial derivative | $f:\mathbb{R}^{n}\to\mathbb{R}$ | [`forward_difference::spartial_derivative()`] |
//! | partial derivative | $\mathbf{f}:\mathbb{R}^{n}\to\mathbb{R}^{m}$ | [`forward_difference::vpartial_derivative()`] |
//! | gradient | $f:\mathbb{R}^{n}\to\mathbb{R}$ | [`forward_difference::gradient()`] |
//! | directional derivative | $f:\mathbb{R}^{n}\to\mathbb{R}$ | [`forward_difference::directional_derivative()`] |
//! | Jacobian | $\mathbf{f}:\mathbb{R}^{n}\to\mathbb{R}^{m}$ | [`forward_difference::jacobian()`] |
//!
//! ### Second-Order Derivatives
//!
//! | Derivative Type | Function Type | Function to Approximate Derivative |
//! | --------------- | ------------- | ---------------------------------- |
//! | Hessian | $f:\mathbb{R}^{n}\to\mathbb{R}$ | [`forward_difference::shessian()`] |
//! | Hessian | $\mathbf{f}:\mathbb{R}^{n}\to\mathbb{R}^{m}$ | [`forward_difference::vhessian()`] |
//! | 2nd derivative | $f:\mathbb{R}\to\mathbb{R}$ | [`forward_difference::sderivative2()`] |
//! | 2nd derivative | $\mathbf{f}:\mathbb{R}\to\mathbb{R}^{m}$ | [`forward_difference::vderivative2()`] |
//! | 2nd partial derivative | $f:\mathbb{R}^{n}\to\mathbb{R}$ | [`forward_difference::spartial_derivative2()`] |
//! | 2nd partial derivative | $\mathbf{f}:\mathbb{R}^{n}\to\mathbb{R}^{m}$ | [`forward_difference::vpartial_derivative2()`] |
//! | mixed 2nd partial derivative | $f:\mathbb{R}^{n}\to\mathbb{R}$ | [`forward_difference::mixed_spartial_derivative2()`] |
//! | mixed 2nd partial derivative | $\mathbf{f}:\mathbb{R}^{n}\to\mathbb{R}^{m}$ | [`forward_difference::mixed_vpartial_derivative2()`] |
//!
//! ## Passing Runtime Parameters
//!
//! For the finite difference methods, we can simply capture any runtime parameters using closures.
//!
//! Examples are included for each function (for example, here is the example for the
//! [`central_difference::jacobian()`] function):
//! [`central_difference::jacobian()` - Example Passing Runtime Parameters](fn@central_difference::jacobian#example-passing-runtime-parameters)

// Linter setup.
#![warn(missing_docs, warnings, clippy::all, clippy::pedantic, clippy::cargo)]
#![allow(
    clippy::float_cmp,
    clippy::multiple_crate_versions,
    clippy::unreadable_literal
)]

// Module declarations.
pub(crate) mod automatic_differentiation;
pub mod central_difference;
pub mod constants;
pub mod forward_difference;

// Module declarations for utils used for testing only.
#[cfg(test)]
pub(crate) mod test_utils;

// Re-exports.
pub use automatic_differentiation::dual::dual::Dual;
pub use automatic_differentiation::dual::dual_vector::DualVector;
pub use automatic_differentiation::hyper_dual::hyper_dual::HyperDual;
pub use automatic_differentiation::hyper_dual::hyper_dual_vector::HyperDualVector;
