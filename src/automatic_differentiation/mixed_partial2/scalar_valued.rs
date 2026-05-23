/// Get a function that returns the mixed second-order partial derivative of the provided
/// multivariate, scalar-valued function.
///
/// The mixed second-order partial derivative is computed using forward-mode automatic
/// differentiation.
///
/// # Arguments
///
/// * `f` - Multivariate, scalar-valued function, $f:\mathbb{R}^{n}\to\mathbb{R}$.
/// * `func_name` - Name of the function that will return the mixed second-order partial derivative
///   of $f(\mathbf{x})$ with respect to $x_{i}$ and $x_{j}$ at any point
///   $\mathbf{x}\in\mathbb{R}^{n}$.
/// * `param_type` (optional) - Type of the extra runtime parameter `p` that is passed to `f`.
///   Defaults to `[f64]` (implying that `f` accepts `p: &[f64]`).
///
/// # Warning
///
/// `f` cannot be defined as closure. It must be defined as a function.
///
/// # Note
///
/// The function produced by this macro will perform 1 evaluation of $f(\mathbf{x})$ to evaluate its
/// mixed second-order partial derivative with respect to $x_{i}$ and $x_{j}$.
///
/// # Examples
///
/// ## Basic Example
///
/// Compute the mixed second-order partial derivative of
///
/// $$f(x,y)=x^{4}+2x^{2}y+y^{3}$$
///
/// with respect to $x$ and $y$ at $(x,y)=(2,1)$, and compare the result to the true result of
///
/// $$\frac{\partial^{2}f}{\partial x\partial y}=4x$$
///
/// with $\left.\frac{\partial^{2}f}{\partial x\partial y}\right|_{(2,1)}=8$.
///
/// ```
/// use linalg_traits::{Scalar, Vector};
/// use numtest::*;
///
/// use numdiff::{get_mixed_spartial_derivative2, HyperDual, HyperDualVector};
///
/// // Define the function, f(x).
/// fn f<S: Scalar, V: Vector<S>>(x: &V, _p: &[f64]) -> S {
///     x.vget(0).powi(4) + S::new(2.0) * x.vget(0).powi(2) * x.vget(1) + x.vget(1).powi(3)
/// }
///
/// // Define the evaluation point.
/// let x0 = vec![2.0, 1.0];
///
/// // Autogenerate the mixed second-order partial derivative function.
/// get_mixed_spartial_derivative2!(f, d2fij);
///
/// // Compute the mixed partial derivative with respect to x₀ and x₁.
/// let d2f_dx0dx1 = d2fij(&x0, 0, 1, &[]);
///
/// // True result: ∂²f/∂x₀∂x₁ = 4x₀ = 8.
/// assert_eq!(d2f_dx0dx1, 8.0);
/// ```
///
/// ## Example Passing Runtime Parameters
///
/// Compute the mixed second-order partial derivative of a parameterized function
///
/// $$f(\mathbf{x}) = a x_{0}^{2}x_{1} + b x_{0}x_{1}^{2} + c\sin(d x_{0}x_{1})$$
///
/// where $a$, $b$, $c$, and $d$ are runtime parameters.
///
/// ```
/// use linalg_traits::{Scalar, Vector};
/// use numtest::*;
///
/// use numdiff::{get_mixed_spartial_derivative2, HyperDual, HyperDualVector};
///
/// // Define the function, f(x).
/// fn f<S: Scalar, V: Vector<S>>(x: &V, p: &[f64]) -> S {
///     let a = S::new(p[0]);
///     let b = S::new(p[1]);
///     let c = S::new(p[2]);
///     let d = S::new(p[3]);
///     a * x.vget(0).powi(2) * x.vget(1) + b * x.vget(0) * x.vget(1).powi(2)
///         + c * (d * x.vget(0) * x.vget(1)).sin()
/// }
///
/// // Runtime parameters.
/// let a = 1.5;
/// let b = 2.0;
/// let c = 0.3;
/// let d = 0.5;
/// let p = [a, b, c, d];
///
/// // Evaluation point.
/// let x0 = vec![1.2, -0.7];
///
/// // Autogenerate the mixed second-order partial derivative function.
/// get_mixed_spartial_derivative2!(f, d2fij);
///
/// // True mixed second-order partial derivative.
/// let d2f_dx0dx1_true = |x: &[f64]| {
///     let theta = d * x[0] * x[1];
///     2.0 * a * x[0] + 2.0 * b * x[1] + c * d * theta.cos() - c * d * d * x[0] * x[1] * theta.sin()
/// };
///
/// // Compute the mixed partial derivative and compare with the true result.
/// let d2f_dx0dx1: f64 = d2fij(&x0, 0, 1, &p);
/// let expected: f64 = d2f_dx0dx1_true(&x0);
/// assert_equal_to_decimal!(d2f_dx0dx1, expected, 14);
/// ```
///
/// ## Example Passing Custom Parameter Types
///
/// Use a custom parameter struct instead of `f64` values.
///
/// ```
/// use linalg_traits::{Scalar, Vector};
/// use numtest::*;
///
/// use numdiff::{get_mixed_spartial_derivative2, HyperDual, HyperDualVector};
///
/// struct Data {
///     a: f64,
///     b: f64,
///     c: f64,
///     d: f64,
/// }
///
/// // Define the function, f(x).
/// fn f<S: Scalar, V: Vector<S>>(x: &V, p: &Data) -> S {
///     let a = S::new(p.a);
///     let b = S::new(p.b);
///     let c = S::new(p.c);
///     let d = S::new(p.d);
///     a * x.vget(0).powi(2) * x.vget(1) + b * x.vget(0) * x.vget(1).powi(2)
///         + c * (d * x.vget(0) * x.vget(1)).sin()
/// }
///
/// // Runtime parameter struct.
/// let p = Data {
///     a: 1.5,
///     b: 2.0,
///     c: 0.3,
///     d: 0.5,
/// };
///
/// // Evaluation point.
/// let x0 = vec![1.2, -0.7];
///
/// // Autogenerate the mixed second-order partial derivative function, telling the macro to expect
/// // a runtime parameter of type &Data.
/// get_mixed_spartial_derivative2!(f, d2fij, Data);
///
/// // True mixed second-order partial derivative.
/// let d2f_dx0dx1_true = |x: &[f64]| {
///     let theta = p.d * x[0] * x[1];
///     2.0 * p.a * x[0] + 2.0 * p.b * x[1] + p.c * p.d * theta.cos()
///         - p.c * p.d * p.d * x[0] * x[1] * theta.sin()
/// };
///
/// // Compute the mixed partial derivative and compare with the true result.
/// let d2f_dx0dx1: f64 = d2fij(&x0, 0, 1, &p);
/// let expected: f64 = d2f_dx0dx1_true(&x0);
/// assert_equal_to_decimal!(d2f_dx0dx1, expected, 14);
/// ```
#[macro_export]
macro_rules! get_mixed_spartial_derivative2 {
    ($f:ident, $func_name:ident) => {
        get_mixed_spartial_derivative2!($f, $func_name, [f64]);
    };
    ($f:ident, $func_name:ident, $param_type:ty) => {
        /// Mixed second-order partial derivative of a multivariate, scalar-valued function
        /// `f: ℝⁿ → ℝ`.
        ///
        /// This function is generated for a specific function `f` using the
        /// `numdiff::get_mixed_spartial_derivative2!` macro.
        ///
        /// # Arguments
        ///
        /// * `x0` - Evaluation point, `x₀ ∈ ℝⁿ`.
        /// * `i` - First element of `x` to differentiate with respect to. Note that this uses
        ///   0-based indexing (e.g. `x = (x₀,...,xᵢ,...,xₙ₋₁)ᵀ`).
        /// * `j` - Second element of `x` to differentiate with respect to. Note that this uses
        ///   0-based indexing (e.g. `x = (x₀,...,xⱼ,...,xₙ₋₁)ᵀ`).
        /// * `p` - Extra runtime parameter. This is a parameter (can be of any arbitrary type)
        ///   defined at runtime that the function may depend on but is not differentiated with
        ///   respect to.
        ///
        /// # Returns
        ///
        /// Mixed second-order partial derivative of `f` with respect to `xᵢ` and `xⱼ`, evaluated at
        /// `x = x₀`.
        ///
        /// `(∂²f/∂xᵢ∂xⱼ)|ₓ₌ₓ₀ ∈ ℝ`
        fn $func_name<S, V>(x0: &V, i: usize, j: usize, p: &$param_type) -> f64
        where
            S: Scalar,
            V: Vector<S>,
        {
            // Promote the evaluation point to a vector of hyper-dual numbers.
            let mut x0_hyperdual = x0.clone().to_hyper_dual_vector();

            if i == j {
                // For i == j, seed both hyper-dual directions on the same variable.
                x0_hyperdual.vset(
                    i,
                    HyperDual::new(x0_hyperdual.vget(i).get_a(), 1.0, 1.0, 0.0),
                );
            } else {
                // Seed separate hyper-dual directions for each variable.
                x0_hyperdual.vset(
                    i,
                    HyperDual::new(x0_hyperdual.vget(i).get_a(), 1.0, 0.0, 0.0),
                );
                x0_hyperdual.vset(
                    j,
                    HyperDual::new(x0_hyperdual.vget(j).get_a(), 0.0, 1.0, 0.0),
                );
            }

            // Evaluate the function at the hyper-dual point.
            let f_x0 = $f(&x0_hyperdual, p);

            // Mixed second-order partial derivative.
            f_x0.get_d()
        }
    };
}

#[cfg(test)]
mod tests {
    use crate::{HyperDual, HyperDualVector};
    use linalg_traits::{Scalar, Vector};
    use nalgebra::SVector;
    use numtest::*;
    use std::f64::consts::PI;

    #[test]
    fn test_mixed_spartial_derivative2_basic() {
        // Function to take the mixed second-order partial derivative of:
        // f(x₀, x₁) = x₀⁴ + 2x₀²x₁ + x₁³
        fn f<S: Scalar, V: Vector<S>>(x: &V, _p: &[f64]) -> S {
            x.vget(0).powi(4) + S::new(2.0) * x.vget(0).powi(2) * x.vget(1) + x.vget(1).powi(3)
        }

        // Define the evaluation point (x₀, x₁) = (2.0, 1.0).
        let x0: Vec<f64> = vec![2.0, 1.0];

        // Generate mixed second-order partial derivative function.
        get_mixed_spartial_derivative2!(f, d2fij);

        // Check ∂²f/∂x₀∂x₁.
        // --> ∂²/∂x₀∂x₁(x₀⁴ + 2x₀²x₁ + x₁³) = 4x₀ = 4(2) = 8
        let d2f_dx0dx1 = d2fij(&x0, 0, 1, &[]);
        assert_equal_to_decimal!(d2f_dx0dx1, 8.0, 15);

        // Check ∂²f/∂x₁∂x₀.
        // --> By symmetry for smooth functions, ∂²f/∂x₁∂x₀ = ∂²f/∂x₀∂x₁ = 8
        let d2f_dx1dx0 = d2fij(&x0, 1, 0, &[]);
        assert_equal_to_decimal!(d2f_dx1dx0, 8.0, 15);

        // Check i == j case (reduces to pure second partial derivative).
        // --> ∂²/∂x₀²(x₀⁴ + 2x₀²x₁ + x₁³) = 12x₀² + 4x₁ = 12(2)² + 4(1) = 52
        let d2f_dx0dx0 = d2fij(&x0, 0, 0, &[]);
        assert_equal_to_decimal!(d2f_dx0dx0, 52.0, 15);
    }

    #[test]
    fn test_mixed_spartial_derivative2_polynomial() {
        // Function to test polynomial terms with mixed coupling:
        // f(x₀, x₁) = x₀³x₁² + x₀²x₁³
        fn f<S: Scalar, V: Vector<S>>(x: &V, _p: &[f64]) -> S {
            x.vget(0).powi(3) * x.vget(1).powi(2) + x.vget(0).powi(2) * x.vget(1).powi(3)
        }

        // Define the evaluation point (x₀, x₁) = (2.0, 1.0).
        let x0: Vec<f64> = vec![2.0, 1.0];

        // Generate mixed second-order partial derivative function.
        get_mixed_spartial_derivative2!(f, d2fij);

        // Check ∂²f/∂x₀∂x₁.
        // --> ∂²/∂x₀∂x₁(x₀³x₁² + x₀²x₁³) = 6x₀²x₁ + 6x₀x₁² = 36
        let d2f_dx0dx1 = d2fij(&x0, 0, 1, &[]);
        assert_eq!(d2f_dx0dx1, 36.0);
    }

    #[test]
    fn test_mixed_spartial_derivative2_multivariate() {
        // Function to take mixed second-order partial derivatives of:
        // f(x₀, x₁, x₂) = x₀x₁x₂ + x₀²x₁ + x₁²x₂
        fn f<S: Scalar, V: Vector<S>>(x: &V, _p: &[f64]) -> S {
            x.vget(0) * x.vget(1) * x.vget(2)
                + x.vget(0).powi(2) * x.vget(1)
                + x.vget(1).powi(2) * x.vget(2)
        }

        // Define the evaluation point (x₀, x₁, x₂) = (1.0, 2.0, 3.0).
        let x0: Vec<f64> = vec![1.0, 2.0, 3.0];

        // Generate mixed second-order partial derivative function.
        get_mixed_spartial_derivative2!(f, d2fij);

        // Check ∂²f/∂x₀∂x₁.
        // --> x₂ + 2x₀ = 3 + 2(1) = 5
        let d2f_dx0dx1 = d2fij(&x0, 0, 1, &[]);
        assert_eq!(d2f_dx0dx1, 5.0);

        // Check ∂²f/∂x₁∂x₂.
        // --> x₀ + 2x₁ = 1 + 2(2) = 5
        let d2f_dx1dx2 = d2fij(&x0, 1, 2, &[]);
        assert_eq!(d2f_dx1dx2, 5.0);

        // Check ∂²f/∂x₀∂x₂.
        // --> x₁ = 2
        let d2f_dx0dx2 = d2fij(&x0, 0, 2, &[]);
        assert_eq!(d2f_dx0dx2, 2.0);
    }

    #[test]
    fn test_mixed_spartial_derivative2_trig() {
        // Function to take mixed second-order partial derivatives of:
        // f(x₀, x₁) = sin(x₀x₁) + cos(x₁)
        fn f<S: Scalar, V: Vector<S>>(x: &V, _p: &[f64]) -> S {
            (x.vget(0) * x.vget(1)).sin() + x.vget(1).cos()
        }

        // Define the evaluation point (x₀, x₁) = (π/2, π/4).
        let x0: Vec<f64> = vec![PI / 2.0, PI / 4.0];

        // Generate mixed second-order partial derivative function.
        get_mixed_spartial_derivative2!(f, d2fij);

        // Check ∂²f/∂x₀∂x₁.
        // --> cos(x₀x₁) - x₀x₁ sin(x₀x₁)
        let expected = (x0[0] * x0[1]).cos() - x0[0] * x0[1] * (x0[0] * x0[1]).sin();
        let d2f_dx0dx1 = d2fij(&x0, 0, 1, &[]);
        assert_equal_to_decimal!(d2f_dx0dx1, expected, 14);
    }

    #[test]
    fn test_mixed_spartial_derivative2_exponential() {
        // Function to take mixed second-order partial derivatives of:
        // f(x₀, x₁) = exp(x₀x₁) + x₀²
        fn f<S: Scalar, V: Vector<S>>(x: &V, _p: &[f64]) -> S {
            (x.vget(0) * x.vget(1)).exp() + x.vget(0).powi(2)
        }

        // Define the evaluation point (x₀, x₁) = (1.0, 2.0).
        let x0: Vec<f64> = vec![1.0, 2.0];

        // Generate mixed second-order partial derivative function.
        get_mixed_spartial_derivative2!(f, d2fij);

        // Check ∂²f/∂x₀∂x₁.
        // --> exp(x₀x₁)(1 + x₀x₁)
        let expected = (x0[0] * x0[1]).exp() * (1.0 + x0[0] * x0[1]);
        let d2f_dx0dx1 = d2fij(&x0, 0, 1, &[]);
        assert_equal_to_decimal!(d2f_dx0dx1, expected, 14);
    }

    #[test]
    #[allow(clippy::many_single_char_names)]
    fn test_mixed_spartial_derivative2_with_runtime_parameters() {
        // Function: f(x₀, x₁) = ax₀²x₁ + bx₀x₁² + cx₀³ + d sin(ex₀x₁).
        #[allow(clippy::many_single_char_names)]
        fn f<S: Scalar, V: Vector<S>>(x: &V, p: &[f64]) -> S {
            let a = S::new(p[0]);
            let b = S::new(p[1]);
            let c = S::new(p[2]);
            let d = S::new(p[3]);
            let e = S::new(p[4]);
            a * x.vget(0).powi(2) * x.vget(1)
                + b * x.vget(0) * x.vget(1).powi(2)
                + c * x.vget(0).powi(3)
                + d * (e * x.vget(0) * x.vget(1)).sin()
        }

        // Runtime parameters.
        let a = 1.5;
        let b = 2.0;
        let c = 0.3;
        let d = 3.0;
        let e = 0.5;
        let p = [a, b, c, d, e];

        // Evaluation point.
        let x0: Vec<f64> = vec![1.2, -0.7];

        // Generate mixed second-order partial derivative function.
        get_mixed_spartial_derivative2!(f, d2fij);

        // True mixed second-order partial derivative.
        let d2f_dx0dx1_expected =
            2.0 * a * x0[0] + 2.0 * b * x0[1] + d * e * (e * x0[0] * x0[1]).cos()
                - d * e * e * x0[0] * x0[1] * (e * x0[0] * x0[1]).sin();

        // Test result.
        let d2f_dx0dx1 = d2fij(&x0, 0, 1, &p);
        assert_equal_to_decimal!(d2f_dx0dx1, d2f_dx0dx1_expected, 14);
    }

    #[test]
    fn test_mixed_spartial_derivative2_custom_params() {
        struct Data {
            a: f64,
            b: f64,
            c: f64,
            d: f64,
            e: f64,
        }

        // Function to take the mixed second-order partial derivative of.
        #[allow(clippy::many_single_char_names)]
        fn f<S: Scalar, V: Vector<S>>(x: &V, p: &Data) -> S {
            let a = S::new(p.a);
            let b = S::new(p.b);
            let c = S::new(p.c);
            let d = S::new(p.d);
            let e = S::new(p.e);
            a * x.vget(0).powi(2) * x.vget(1)
                + b * x.vget(0) * x.vget(1).powi(2)
                + c * x.vget(0).powi(3)
                + d * (e * x.vget(0) * x.vget(1)).sin()
        }

        // Runtime parameter struct.
        let p = Data {
            a: 1.5,
            b: 2.0,
            c: 0.3,
            d: 3.0,
            e: 0.5,
        };

        // Evaluation point.
        let x0: Vec<f64> = vec![1.2, -0.7];

        // Mixed second-order partial derivative function obtained via forward-mode automatic
        // differentiation.
        get_mixed_spartial_derivative2!(f, d2fij, Data);

        // True mixed second-order partial derivative.
        let d2f_dx0dx1_expected =
            2.0 * p.a * x0[0] + 2.0 * p.b * x0[1] + p.d * p.e * (p.e * x0[0] * x0[1]).cos()
                - p.d * p.e * p.e * x0[0] * x0[1] * (p.e * x0[0] * x0[1]).sin();

        // Evaluate the mixed second-order partial derivative.
        let d2f_dx0dx1 = d2fij(&x0, 0, 1, &p);

        // Test result.
        assert_equal_to_decimal!(d2f_dx0dx1, d2f_dx0dx1_expected, 14);
    }

    #[test]
    fn test_mixed_spartial_derivative2_vector_types() {
        // Function to take the mixed second-order partial derivative of:
        // f(x₀, x₁) = x₀²x₁ + x₁³
        fn f<S: Scalar, V: Vector<S>>(x: &V, _p: &[f64]) -> S {
            x.vget(0).powi(2) * x.vget(1) + x.vget(1).powi(3)
        }

        // Mixed second-order partial derivative function obtained via forward-mode automatic
        // differentiation.
        get_mixed_spartial_derivative2!(f, d2fij);

        // Using nalgebra SVector
        let x_nalgebra: SVector<f64, 2> = SVector::from([2.0, 3.0]);

        // ∂²f/∂x₀∂x₁ = 2x₀ = 4
        assert_eq!(d2fij(&x_nalgebra, 0, 1, &[]), 4.0);

        // Symmetry: ∂²f/∂x₁∂x₀ = 2x₀ = 4
        assert_eq!(d2fij(&x_nalgebra, 1, 0, &[]), 4.0);
    }
}
