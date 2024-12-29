# numdiff

[<img alt="github" src="https://img.shields.io/badge/github-tamaskis/numdiff-8da0cb?style=for-the-badge&labelColor=555555&logo=github" height="20">](https://github.com/tamaskis/numdiff)
[<img alt="crates.io" src="https://img.shields.io/crates/v/numdiff.svg?style=for-the-badge&color=fc8d62&logo=rust" height="20">](https://crates.io/crates/numdiff)
[<img alt="docs.rs" src="https://img.shields.io/badge/docs.rs-numdiff-66c2a5?style=for-the-badge&labelColor=555555&logo=docs.rs" height="20">](https://docs.rs/numdiff)

Numerical differentiation via forward-mode automatic differentiation and finite difference approximations.

## Documentation

Please see https://docs.rs/numdiff.

## Example

Consider the function

```
f(x) = (x₀)⁵ + sin³(x₁)
```

The `numdiff` crate provides various functions that can be used to approximate its gradient. Here, we approximate its gradient at `x = (5, 8)ᵀ` using `numdiff::forward_difference::gradient()` (i.e. using the forward difference approximation). We perform this gradient approximation three times, each time using a different vector type to define the function `f(x)`.

```rust
use nalgebra::SVector;
use ndarray::{array, Array1};
use numtest::*;

use numdiff::forward_difference::gradient;

// f(x) written in terms of a dynamically-sized standard vector (f1), a statically-sized
// nalgebra vector (f2), and a dynamically-sized ndarray vector (f3).
let f1 = |x: &Vec<f64>| x[0].powi(5) + x[1].sin().powi(3);
let f2 = |x: &SVector<f64,2>| x[0].powi(5) + x[1].sin().powi(3);
let f3 = |x: &Array1<f64>| x[0].powi(5) + x[1].sin().powi(3);

// Evaluation points using the three types of vectors.
let x1: Vec<f64> = vec![5.0, 8.0];
let x2: SVector<f64, 2> = SVector::from_row_slice(&[5.0, 8.0]);
let x3: Array1<f64> = array![5.0, 8.0];

// Approximate the gradients.
let grad_f1: Vec<f64> = gradient(&f1, &x1, None);
let grad_f2: SVector<f64, 2> = gradient(&f2, &x2, None);
let grad_f3: Array1<f64> = gradient(&f3, &x3, None);

// Verify that the gradient approximations are all identical.
assert_arrays_equal!(grad_f1, grad_f2);
assert_arrays_equal!(grad_f1, grad_f3);
```

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