# numdiff

[<img alt="github" src="https://img.shields.io/badge/github-tamaskis/numdiff-8da0cb?style=for-the-badge&labelColor=555555&logo=github" height="20">](https://github.com/tamaskis/numdiff)
[<img alt="crates.io" src="https://img.shields.io/crates/v/numdiff.svg?style=for-the-badge&color=fc8d62&logo=rust" height="20">](https://crates.io/crates/numdiff)
[<img alt="docs.rs" src="https://img.shields.io/badge/docs.rs-numdiff-66c2a5?style=for-the-badge&labelColor=555555&logo=docs.rs" height="20">](https://docs.rs/numdiff)

Automatic and numerical differentiation.

## Documentation

Please see https://docs.rs/numdiff.

## Overview

This crate implements two different methods for evaluating derivatives in Rust:

1. Automatic differentiation (forward-mode using first-order dual numbers).
2. Numerical differentiation (using forward difference and central difference approximations).

This crate provides generic functions (for numerical differentiation) and macros (for automatic differentiation) to evaluate various types of derivatives of the following types of functions:

* Univariate, scalar-valued functions (`f: ℝ → ℝ`)
* Univariate, vector-valued functions (`f: ℝ → ℝᵐ`)
* Multivariate, scalar-valued functions (`f: ℝⁿ → ℝ`)
* Multivariate, vector-valued functions (`f: ℝⁿ → ℝᵐ`)

These functions and macros are made generic over the choice of vector representation, as long as the vector type implements the `linalg_traits::Vector` trait. See the [`linalg_traits` documentation](https://docs.rs/linalg-traits/latest/linalg_traits/) for more information.

#### License

<sup>
Licensed under either of <a href="LICENSE-APACHE">Apache License, Version 2.0</a> or 
<a href="LICENSE-MIT">MIT license</a> at your option.
</sup>

<br>

<sub>
Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in
this crate by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without
any additional terms or conditions.
</sub>