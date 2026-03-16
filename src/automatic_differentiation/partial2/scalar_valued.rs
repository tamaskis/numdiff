/// Get a function that returns the second-order partial derivative of the provided multivariate,
/// scalar-valued function.
///
/// The second-order partial derivative is computed using forward-mode automatic differentiation.
///
/// # Arguments
///
/// * `f` - Multivariate, scalar-valued function, $f:\mathbb{R}^{n}\to\mathbb{R}$.
/// * `func_name` - Name of the function that will return the second-order partial derivative of
///   $f(\mathbf{x})$ with respect to $x_{k}$ at any point $\mathbf{x}\in\mathbb{R}^{n}$.
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
/// second-order partial derivative with respect to $x_{k}$.
///
/// # Examples
///
/// ## Basic Example
///
/// Compute the second-order partial derivative of
///
/// $$f(x,y)=x^{4}+2x^{2}y+y^{3}$$
///
/// with respect to $x$ at $(x,y)=(2,1)$, and compare the result to the true result of
///
/// $$\frac{\partial^{2}f}{\partial x^{2}}\bigg\rvert_{(x,y)=(2,1)}=12x^{2}+4y\bigg\rvert_{(x,y)=(2,1)}=52$$
///
/// First, note that we can rewrite this function as
///
/// $$f(\mathbf{x})=x_{0}^{4}+2x_{0}^{2}x_{1}+x_{1}^{3}$$
///
/// where $\mathbf{x}=(x_{0},x_{1})^{T}$ (note that we use 0-based indexing to aid with the
/// computational implementation). We are then trying to find
///
/// $$\frac{\partial^{2} f}{\partial x_{0}^{2}}\bigg\rvert_{\mathbf{x}=\mathbf{x}_{0}}$$
///
/// where $\mathbf{x}_{0}=(2,1)^{T}$.
///
/// #### Using standard vectors
///
/// ```
/// use linalg_traits::{Scalar, Vector};
///
/// use numdiff::{get_spartial_derivative2, HyperDual, HyperDualVector};
///
/// // Define the function, f(x).
/// fn f<S: Scalar, V: Vector<S>>(x: &V, _p: &[f64]) -> S {
///     x.vget(0).powi(4) + S::new(2.0) * x.vget(0).powi(2) * x.vget(1) + x.vget(1).powi(3)
/// }
///
/// // Define the evaluation point.
/// let x0 = vec![2.0, 1.0];
///
/// // Define the element of the vector (using 0-based indexing) we are differentiating with respect
/// // to.
/// let k = 0;
///
/// // Autogenerate the function "d2fkk" that can be used to compute the second-order partial
/// // derivative of f(x) with respect to xₖ at any point x.
/// get_spartial_derivative2!(f, d2fkk);
///
/// // Evaluate the second-order partial derivative.
/// let d2f_dx0dx0 = d2fkk(&x0, k, &[]);
///
/// // True result: 12x₀² + 4x₁ = 12(2)² + 4(1) = 52
/// assert_eq!(d2f_dx0dx0, 52.0);
/// ```
///
/// ## Example Passing Runtime Parameters
///
/// Compute the second-order partial derivative of a parameterized function
///
/// $$f(\mathbf{x})=a x_{0}^{4}+b x_{0}^{2}x_{1}+c x_{1}^{3}+d\sin(e x_{0})$$
///
/// where $a$, $b$, $c$, $d$, and $e$ are runtime parameters.
///
/// ```
/// use linalg_traits::{Scalar, Vector};
/// use numtest::*;
///
/// use numdiff::{get_spartial_derivative2, HyperDual, HyperDualVector};
///
/// // Define the function, f(x).
/// fn f<S: Scalar, V: Vector<S>>(x: &V, p: &[f64]) -> S {
///     let a = S::new(p[0]);
///     let b = S::new(p[1]);
///     let c = S::new(p[2]);
///     let d = S::new(p[3]);
///     let e = S::new(p[4]);
///     a * x.vget(0).powi(4)
///         + b * x.vget(0).powi(2) * x.vget(1)
///         + c * x.vget(1).powi(3)
///         + d * (e * x.vget(0)).sin()
/// }
///
/// // Runtime parameters.
/// let a = 1.5;
/// let b = 2.0;
/// let c = 0.8;
/// let d = 3.0;
/// let e = 0.5;
///
/// // Parameter vector.
/// let p = [a, b, c, d, e];
///
/// // Evaluation point.
/// let x0 = vec![1.0, -0.5];
///
/// // Autogenerate the second-order partial derivative function.
/// get_spartial_derivative2!(f, d2fkk);
///
/// // True second-order partial derivative functions.
/// let d2f_dx0dx0_true = |x: &[f64]| {
///     12.0 * a * x[0].powi(2) + 2.0 * b * x[1] - d * e * e * (e * x[0]).sin()
/// };
/// let d2f_dx1dx1_true = |x: &[f64]| 6.0 * c * x[1];
///
/// // Compute ∂²f/∂x₀² at x₀ and compare with true function.
/// let d2f_dx0dx0: f64 = d2fkk(&x0, 0, &p);
/// let expected_d2f_dx0dx0 = d2f_dx0dx0_true(&x0);
/// assert_equal_to_decimal!(d2f_dx0dx0, expected_d2f_dx0dx0, 14);
///
/// // Compute ∂²f/∂x₁² at x0 and compare with true function.
/// let d2f_dx1dx1: f64 = d2fkk(&x0, 1, &p);
/// let expected_d2f_dx1dx1 = d2f_dx1dx1_true(&x0);
/// assert_equal_to_decimal!(d2f_dx1dx1, expected_d2f_dx1dx1, 15);
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
/// use numdiff::{get_spartial_derivative2, HyperDual, HyperDualVector};
///
/// struct Data {
///     a: f64,
///     b: f64,
///     c: f64,
///     d: f64,
///     e: f64,
/// }
///
/// // Define the function, f(x).
/// fn f<S: Scalar, V: Vector<S>>(x: &V, p: &Data) -> S {
///     let a = S::new(p.a);
///     let b = S::new(p.b);
///     let c = S::new(p.c);
///     let d = S::new(p.d);
///     let e = S::new(p.e);
///     a * x.vget(0).powi(4)
///         + b * x.vget(0).powi(2) * x.vget(1)
///         + c * x.vget(1).powi(3)
///         + d * (e * x.vget(0)).sin()
/// }
///
/// // Runtime parameter struct.
/// let p = Data {
///     a: 1.5,
///     b: 2.0,
///     c: 0.8,
///     d: 3.0,
///     e: 0.5,
/// };
///
/// // Evaluation point.
/// let x0 = vec![1.0, -0.5];
///
/// // Autogenerate the second-order partial derivative function, telling the macro to expect a
/// // runtime parameter of type &Data.
/// get_spartial_derivative2!(f, d2fkk, Data);
///
/// // True second-order partial derivative functions.
/// let d2f_dx0dx0_true = |x: &[f64]| {
///     12.0 * p.a * x[0].powi(2) + 2.0 * p.b * x[1] - p.d * p.e * p.e * (p.e * x[0]).sin()
/// };
/// let d2f_dx1dx1_true = |x: &[f64]| 6.0 * p.c * x[1];
///
/// // Compute the second-order partial derivatives using both the automatically generated
/// // second-order partial derivative function and the true second-order partial derivative
/// // functions, and compare the results.
/// let d2f_dx0dx0: f64 = d2fkk(&x0, 0, &p);
/// let d2f_dx1dx1: f64 = d2fkk(&x0, 1, &p);
/// assert_equal_to_decimal!(d2f_dx0dx0, d2f_dx0dx0_true(&x0), 14);
/// assert_equal_to_decimal!(d2f_dx1dx1, d2f_dx1dx1_true(&x0), 15);
/// ```
#[macro_export]
macro_rules! get_spartial_derivative2 {
    ($f:ident, $func_name:ident) => {
        get_spartial_derivative2!($f, $func_name, [f64]);
    };
    ($f:ident, $func_name:ident, $param_type:ty) => {
        /// Second-order partial derivative of a multivariate, scalar-valued function `f: ℝⁿ → ℝ`.
        ///
        /// This function is generated for a specific function `f` using the
        /// `numdiff::get_spartial_derivative2!` macro.
        ///
        /// # Arguments
        ///
        /// * `x0` - Evaluation point, `x₀ ∈ ℝⁿ`.
        /// * `k` - Element of `x` to differentiate with respect to twice. Note that this uses
        ///   0-based indexing (e.g. `x = (x₀,...,xₖ,...,xₙ₋₁)ᵀ`).
        /// * `p` - Extra runtime parameter. This is a parameter (can be of any arbitrary type)
        ///   defined at runtime that the function may depend on but is not differentiated with
        ///   respect to.
        ///
        /// # Returns
        ///
        /// Second-order partial derivative of `f` with respect to `xₖ`, evaluated at `x = x₀`.
        ///
        /// `(∂²f/∂xₖ²)|ₓ₌ₓ₀ ∈ ℝ`
        fn $func_name<S, V>(x0: &V, k: usize, p: &$param_type) -> f64
        where
            S: Scalar,
            V: Vector<S>,
        {
            // Promote the evaluation point to a vector of hyper-dual numbers.
            let mut x0_hyperdual = x0.clone().to_hyper_dual_vector();

            // Take a unit step forward in both hyper-dual directions for the kth component.
            x0_hyperdual.vset(
                k,
                HyperDual::new(x0_hyperdual.vget(k).get_a(), 1.0, 1.0, 0.0),
            );

            // Evaluate the function at the hyper-dual number.
            let f_x0 = $f(&x0_hyperdual, p);

            // Second-order partial derivative of f with respect to xₖ.
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

    #[cfg(feature = "trig")]
    use trig::Trig;

    #[test]
    fn test_spartial_derivative2_basic() {
        // Function to take the second-order partial derivative of:
        // f(x₀, x₁) = x₀⁴ + 2x₀²x₁ + x₁³
        fn f<S: Scalar, V: Vector<S>>(x: &V, _p: &[f64]) -> S {
            x.vget(0).powi(4) + S::new(2.0) * x.vget(0).powi(2) * x.vget(1) + x.vget(1).powi(3)
        }

        // Define the evaluation point (x₀, x₁) = (2.0, 1.0).
        let x0 = vec![2.0, 1.0];

        // Generate second-order partial derivative function.
        get_spartial_derivative2!(f, d2fkk);

        // Check ∂²f/∂x₀².
        // --> ∂²/∂x₀²(x₀⁴ + 2x₀²x₁ + x₁³) = 12x₀² + 4x₁ = 12(2)² + 4(1) = 52
        let d2f_dx0dx0 = d2fkk(&x0, 0, &[]);
        assert_equal_to_decimal!(d2f_dx0dx0, 52.0, 15);

        // Check ∂²f/∂x₁².
        // --> ∂²/∂x₁²(x₀⁴ + 2x₀²x₁ + x₁³) = 6x₁ = 6(1) = 6
        let d2f_dx1dx1 = d2fkk(&x0, 1, &[]);
        assert_equal_to_decimal!(d2f_dx1dx1, 6.0, 15);
    }

    #[test]
    fn test_spartial_derivative2_polynomial() {
        // Function to test various polynomial orders:
        // f(x₀) = x₀² + x₀³ + x₀⁴
        fn f<S: Scalar, V: Vector<S>>(x: &V, _p: &[f64]) -> S {
            x.vget(0).powi(2) + x.vget(0).powi(3) + x.vget(0).powi(4)
        }

        // Define the evaluation point x₀ = 2.0.
        let x0 = vec![2.0];

        // Generate second-order partial derivative function.
        get_spartial_derivative2!(f, d2fkk);

        // Check ∂²f/∂x₀².
        // --> ∂²/∂x₀²(x₀² + x₀³ + x₀⁴) = 2 + 6x₀ + 12x₀² = 2 + 6(2) + 12(4) = 62
        let d2f_dx0dx0 = d2fkk(&x0, 0, &[]);
        assert_eq!(d2f_dx0dx0, 62.0);
    }

    #[test]
    fn test_spartial_derivative2_multivariate() {
        // Function to take the second-order partial derivative of:
        // f(x₀, x₁, x₂) = x₀³ + x₁⁴ + x₂²
        fn f<S: Scalar, V: Vector<S>>(x: &V, _p: &[f64]) -> S {
            x.vget(0).powi(3) + x.vget(1).powi(4) + x.vget(2).powi(2)
        }

        // Define the evaluation point (x₀, x₁, x₂) = (1.0, 2.0, 3.0).
        let x0 = vec![1.0, 2.0, 3.0];

        // Generate second-order partial derivative function.
        get_spartial_derivative2!(f, d2fkk);

        // Check ∂²f/∂x₀².
        // --> ∂²/∂x₀²(x₀³ + x₁⁴ + x₂²) = 6x₀ = 6(1) = 6
        let d2f_dx0dx0 = d2fkk(&x0, 0, &[]);
        assert_eq!(d2f_dx0dx0, 6.0);

        // Check ∂²f/∂x₁².
        // --> ∂²/∂x₁²(x₀³ + x₁⁴ + x₂²) = 12x₁² = 12(2)² = 48
        let d2f_dx1dx1 = d2fkk(&x0, 1, &[]);
        assert_eq!(d2f_dx1dx1, 48.0);

        // Check ∂²f/∂x₂².
        // --> ∂²/∂x₂²(x₀³ + x₁⁴ + x₂²) = 2
        let d2f_dx2dx2 = d2fkk(&x0, 2, &[]);
        assert_eq!(d2f_dx2dx2, 2.0);
    }

    #[test]
    fn test_spartial_derivative2_trig() {
        // Function to take the second-order partial derivative of:
        // f(x₀, x₁) = sin(x₀) + cos(x₁)
        fn f<S: Scalar, V: Vector<S>>(x: &V, _p: &[f64]) -> S {
            x.vget(0).sin() + x.vget(1).cos()
        }

        // Define the evaluation point (x₀, x₁) = (π/2, π/4).
        let x0 = vec![PI / 2.0, PI / 4.0];

        // Generate second-order partial derivative function.
        get_spartial_derivative2!(f, d2fkk);

        // Check ∂²f/∂x₀².
        // --> ∂²/∂x₀²(sin(x₀) + cos(x₁)) = -sin(x₀) = -sin(π/2) = -1
        let d2f_dx0dx0 = d2fkk(&x0, 0, &[]);
        assert_equal_to_decimal!(d2f_dx0dx0, -1.0, 15);

        // Check ∂²f/∂x₁².
        // --> ∂²/∂x₁²(sin(x₀) + cos(x₁)) = -cos(x₁) = -cos(π/4) = -√(2)/2
        let expected = -(2.0_f64.sqrt() / 2.0);
        let d2f_dx1dx1 = d2fkk(&x0, 1, &[]);
        assert_equal_to_decimal!(d2f_dx1dx1, expected, 15);
    }

    #[test]
    fn test_spartial_derivative2_exponential() {
        // Function to take the second-order partial derivative of:
        // f(x₀, x₁) = exp(x₀) + x₁²
        fn f<S: Scalar, V: Vector<S>>(x: &V, _p: &[f64]) -> S {
            x.vget(0).exp() + x.vget(1).powi(2)
        }

        // Define the evaluation point (x₀, x₁) = (1.0, 2.0).
        let x0 = vec![1.0, 2.0];

        // Generate second-order partial derivative function.
        get_spartial_derivative2!(f, d2fkk);

        // Check ∂²f/∂x₀².
        // --> ∂²/∂x₀²(exp(x₀) + x₁²) = exp(x₀) = exp(1) = e
        let d2f_dx0dx0 = d2fkk(&x0, 0, &[]);
        assert_equal_to_decimal!(d2f_dx0dx0, std::f64::consts::E, 15);

        // Check ∂²f/∂x₁².
        // --> ∂²/∂x₁²(exp(x₀) + x₁²) = 2
        let d2f_dx1dx1 = d2fkk(&x0, 1, &[]);
        assert_eq!(d2f_dx1dx1, 2.0);
    }

    #[test]
    #[allow(clippy::many_single_char_names)]
    fn test_spartial_derivative2_with_runtime_parameters() {
        // Function: f(x₀, x₁) = ax₀⁴ + bx₀²x₁ + cx₁³ + dsin(ex₀).
        fn f<S: Scalar, V: Vector<S>>(x: &V, p: &[f64]) -> S {
            let a = S::new(p[0]);
            let b = S::new(p[1]);
            let c = S::new(p[2]);
            let d = S::new(p[3]);
            let e = S::new(p[4]);
            a * x.vget(0).powi(4)
                + b * x.vget(0).powi(2) * x.vget(1)
                + c * x.vget(1).powi(3)
                + d * (e * x.vget(0)).sin()
        }

        // Runtime parameters.
        let a = 1.5;
        let b = 2.0;
        let c = 0.8;
        let d = 3.0;
        let e = 0.5;
        let p = [a, b, c, d, e];

        // Evaluation point.
        let x0 = vec![1.0, -0.5];

        // Generate second-order partial derivative function.
        get_spartial_derivative2!(f, d2fkk);

        // True second-order partial derivatives.
        let d2f_dx0dx0_expected = 12.0_f64 * a * (x0[0] as f64).powi(2_i32) + 2.0_f64 * b * x0[1]
            - d * e * e * (e * x0[0]).sin();
        let d2f_dx1dx1_expected = 6.0_f64 * c * x0[1];

        // Test results.
        let d2f_dx0dx0 = d2fkk(&x0, 0, &p);
        let d2f_dx1dx1 = d2fkk(&x0, 1, &p);

        assert_equal_to_decimal!(d2f_dx0dx0, d2f_dx0dx0_expected, 14);
        assert_equal_to_decimal!(d2f_dx1dx1, d2f_dx1dx1_expected, 15);
    }

    #[test]
    fn test_spartial_derivative2_custom_params() {
        struct Data {
            a: f64,
            b: f64,
            c: f64,
            d: f64,
            e: f64,
        }

        // Function to take the second-order partial derivative of.
        #[allow(clippy::many_single_char_names)]
        fn f<S: Scalar, V: Vector<S>>(x: &V, p: &Data) -> S {
            let a = S::new(p.a);
            let b = S::new(p.b);
            let c = S::new(p.c);
            let d = S::new(p.d);
            let e = S::new(p.e);
            a * x.vget(0).powi(4)
                + b * x.vget(0).powi(2) * x.vget(1)
                + c * x.vget(1).powi(3)
                + d * (e * x.vget(0)).sin()
        }

        // Runtime parameter struct.
        let p = Data {
            a: 1.5,
            b: 2.0,
            c: 0.8,
            d: 3.0,
            e: 0.5,
        };

        // Evaluation point.
        let x0 = vec![1.0, -0.5];

        // Second-order partial derivative function obtained via forward-mode automatic
        // differentiation.
        get_spartial_derivative2!(f, d2fkk, Data);

        // True second-order partial derivative functions.
        let d2f_dx0dx0_expected = 12.0_f64 * p.a * (x0[0] as f64).powi(2_i32)
            + 2.0_f64 * p.b * x0[1]
            - p.d * p.e * p.e * (p.e * x0[0]).sin();
        let d2f_dx1dx1_expected = 6.0_f64 * p.c * x0[1];

        // Evaluate the second-order partial derivatives.
        let d2f_dx0dx0 = d2fkk(&x0, 0, &p);
        let d2f_dx1dx1 = d2fkk(&x0, 1, &p);

        // Test results.
        assert_equal_to_decimal!(d2f_dx0dx0, d2f_dx0dx0_expected, 14);
        assert_equal_to_decimal!(d2f_dx1dx1, d2f_dx1dx1_expected, 15);
    }

    #[test]
    fn test_spartial_derivative2_vector_types() {
        // Test with nalgebra SVector.
        fn f<S: Scalar, V: Vector<S>>(x: &V, _p: &[f64]) -> S {
            x.vget(0).powi(3) + x.vget(1).powi(2)
        }

        // Second-order partial derivative function obtained via forward-mode automatic
        // differentiation.
        get_spartial_derivative2!(f, d2fkk);

        // Using nalgebra SVector
        let x_nalgebra: SVector<f64, 2> = SVector::from([2.0, 3.0]);

        // ∂²f/∂x₀² = 6x₀ = 12
        assert_eq!(d2fkk(&x_nalgebra, 0, &[]), 12.0);

        // ∂²f/∂x₁² = 2
        assert_eq!(d2fkk(&x_nalgebra, 1, &[]), 2.0);
    }
}
