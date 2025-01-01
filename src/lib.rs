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
//! | Derivative Type | Function Type | Macro to Generate Derivative Function |
//! | --------------- | ------------- | ---------------------------------- |
//! | derivative | $f:\mathbb{R}\to\mathbb{R}$ | [`get_sderivative!`] |
//! | derivative | $\mathbf{f}:\mathbb{R}\to\mathbb{R}^{m}$ | [`get_vderivative!`] |
//! | partial derivative | $f:\mathbb{R}^{n}\to\mathbb{R}$ | [`get_spartial_derivative!`] |
//! | partial derivative | $\mathbf{f}:\mathbb{R}^{n}\to\mathbb{R}^{m}$ | [`get_vpartial_derivative!`] |
//! | gradient | $f:\mathbb{R}^{n}\to\mathbb{R}$ | [`get_gradient!`] |
//! | directional derivative | $f:\mathbb{R}^{n}\to\mathbb{R}$ | [`get_directional_derivative!`] |
//! | Jacobian| $\mathbf{f}:\mathbb{R}^{n}\to\mathbb{R}^{m}$ | [`get_jacobian!`] |
//!
//! ## Limitations
//!
//! * These macros only work on functions that are generic both over the type of scalar and the type
//!   of vector.
//! * Consequently, these macros do _not_ work on closures.
//!     - Currently, this also means we can't "pass extra parameters" to a function.
//! * Constants (e.g. `5.0_f64`) need to be defined using `linalg_traits::Scalar::new` (e.g. if a
//!   function has the generic parameter `S: Scalar`, then instead of defining a constant number
//!   such as `5.0_f64`, we need to do `S::new(5.0)`).
//!     - This is also the case for some functions that can take constants are arguments, such as
//!       [`num_traits::Float::powf`].
//! * When defining functions that operate on generic scalars (to make them compatible with
//!   automatic differentiation), we cannot do an assignment operation such as `1.0 += x` if
//!   `x: S` where `S: Scalar`.
//!
//! ## Alternatives
//!
//! There are already some alternative crates in the Rust ecosystem that already implement dual
//! numbers. Originally, I intended to implement the autodifferentiation functions in this crate
//! using one of those other dual number implementations in the backend. However, each crate had
//! certain shortcomings that ultimately led to me providing a custom implementation of dual numbers
//! in this crate. The alternative crates implementing dual numbers are described below.
//!
//! ##### [`num-dual`](https://docs.rs/num-dual/latest/num_dual/)
//!
//! * This crate _can_ be used to differentiate functions of generic types that implement the
//!   [`DualNum`](https://docs.rs/num-dual/latest/num_dual/trait.DualNum.html) trait. Since this
//!   trait is implemented for [`f32`] and [`f64`], it would allow us to write generic functions
//!   that can be simply evaluated using [`f64`]s, but can also be automatically differentiated if
//!   needed.
//! * However, there are some notable shortcomings that are described below.
//! * The [`Dual`](https://github.com/itt-ustutt/num-dual/blob/master/src/dual.rs) struct panics in
//!   its implementations of the following standard functions which are quite common in engineering:
//!     - [`num_traits::Float::floor`]
//!     - [`num_traits::Float::ceil`]
//!     - [`num_traits::Float::round`]
//!     - [`num_traits::Float::trunc`]
//!     - [`num_traits::Float::fract`]
//! * `num-dual` has a required dependency on
//!   [`nalgebra`](https://docs.rs/nalgebra/latest/nalgebra/), which is quite a heavy dependency for
//!   those who do not need it.
//!
//! ##### [`autodj`](https://docs.rs/autodj/latest/autodj/)
//!
//! * Can only differentiate functions written using custom types, such as
//!   [`DualF64`](https://docs.rs/autodj/latest/autodj/solid/single/type.DualF64.html).
//! * Multivariate functions, especially those with a dynamic number of variables, can be extremely
//!   clunky (see
//!   [this example](https://docs.rs/autodj/latest/autodj/index.html#dynamic-number-of-variables)).
//!
//! ##### [`autodiff`](https://docs.rs/autodiff/latest/autodiff/)
//!
//! * Can only differentiate functions written using the custom type
//!   [`FT<T>`](https://docs.rs/autodiff/latest/autodiff/forward_autodiff/type.FT.html).
//! * Incorrect implementation of certain functions, such as [`num_traits::Float::floor`] (see the
//!   [source code](https://github.com/elrnv/autodiff/blob/master/src/forward_autodiff.rs)).
//!
//! # Central Difference Approximations
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
//! | Hessian | $f:\mathbb{R}^{n}\to\mathbb{R}$ | [`central_difference::shessian()`] |
//! | Hessian | $\mathbf{f}:\mathbb{R}^{n}\to\mathbb{R}^{m}$ | [`central_difference::vhessian()`] |
//!
//! # Forward Difference Approximations
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
//! | Hessian | $f:\mathbb{R}^{n}\to\mathbb{R}$ | [`forward_difference::shessian()`] |
//! | Hessian | $\mathbf{f}:\mathbb{R}^{n}\to\mathbb{R}^{m}$ | [`forward_difference::vhessian()`] |

// Linter setup.
#![warn(missing_docs)]

// Module declarations.
pub(crate) mod automatic_differentiation;
pub mod central_difference;
pub mod constants;
pub mod forward_difference;

// Module declarations for utils used for testing only.
#[cfg(test)]
pub(crate) mod test_utils;

// Re-exports.
pub use automatic_differentiation::dual::Dual;
pub use automatic_differentiation::dual_vector::DualVector;
