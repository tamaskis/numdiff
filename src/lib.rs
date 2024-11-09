//! [![github]](https://github.com/tamaskis/numdiff)&ensp;[![crates-io]](https://crates.io/crates/numdiff)&ensp;[![docs-rs]](https://docs.rs/numdiff)
//!
//! [github]: https://img.shields.io/badge/github-8da0cb?style=for-the-badge&labelColor=555555&logo=github
//! [crates-io]: https://img.shields.io/badge/crates.io-fc8d62?style=for-the-badge&labelColor=555555&logo=rust
//! [docs-rs]: https://img.shields.io/badge/docs.rs-66c2a5?style=for-the-badge&labelColor=555555&logo=docs.rs
//!
//! Numerical differentiation.
//!
//! # Overview
//!
//! This crate provides generic functions to evaluate various types of derivatives of the following
//! types of functions:
//!
//! * Univariate, scalar-valued functions ($f:\mathbb{R}\to\mathbb{R}$)
//! * Univariate, vector-valued functions ($\mathbf{f}:\mathbb{R}\to\mathbb{R}^{m}$)
//! * Multivariate, scalar-valued functions ($f:\mathbb{R}^{n}\to\mathbb{R}$)
//! * Multivariate, vector-valued functions ($\mathbf{f}:\mathbb{R}^{n}\to\mathbb{R}^{m}$)
//!
//! These functions are made generic over the choice of vector representation, as long as the vector
//! type implements the `linalg_traits::Vector` trait. See the
//! [`linalg_traits` documentation](https://docs.rs/linalg-traits/latest/linalg_traits/) for more
//! information.
//!
//! # Example
//!
//! Consider the function
//!
//! $$f(\mathbf{x})=x_{0}^{5}+\sin^{3}{x_{1}}$$
//!
//! The `numdiff` crate provides various functions that can be used to approximate its gradient.
//! Here, we approximate its gradient at $\mathbf{x}=(5,8)^{T}$ using
//! [`forward_difference::gradient()`] (i.e. using the forward difference approximation). We perform
//! this gradient approximation three times, each time using a different vector type to define the
//! function $f(\mathbf{x})$.
//!
//! ```
//! use nalgebra::SVector;
//! use ndarray::{array, Array1};
//! use numtest::*;
//!
//! use numdiff::forward_difference::gradient;
//!
//! // f(x) written in terms of a dynamically-sized standard vector (f1), a statically-sized
//! // nalgebra vector (f2), and a dynamically-sized ndarray vector (f3).
//! let f1 = |x: &Vec<f64>| x[0].powi(5) + x[1].sin().powi(3);
//! let f2 = |x: &SVector<f64,2>| x[0].powi(5) + x[1].sin().powi(3);
//! let f3 = |x: &Array1<f64>| x[0].powi(5) + x[1].sin().powi(3);
//!
//! // Evaluation points using the three types of vectors.
//! let x1: Vec<f64> = vec![5.0, 8.0];
//! let x2: SVector<f64, 2> = SVector::from_row_slice(&[5.0, 8.0]);
//! let x3: Array1<f64> = array![5.0, 8.0];
//!
//! // Approximate the gradients.
//! let grad_f1: Vec<f64> = gradient(&f1, &x1, None);
//! let grad_f2: SVector<f64, 2> = gradient(&f2, &x2, None);
//! let grad_f3: Array1<f64> = gradient(&f3, &x3, None);
//!
//! // Verify that the gradient approximations are all identical.
//! assert_arrays_equal!(grad_f1, grad_f2);
//! assert_arrays_equal!(grad_f1, grad_f3);
//! ```
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
pub mod constants;
pub mod forward_difference;

// Module declarations for utils used for testing only.
#[cfg(test)]
pub(crate) mod test_utils;
