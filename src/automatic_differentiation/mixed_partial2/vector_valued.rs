/// Get a function that returns the mixed second-order partial derivative of the provided
/// multivariate, vector-valued function.
///
/// The mixed second-order partial derivative is computed using forward-mode automatic
/// differentiation.
///
/// # Arguments
///
/// * `f` - Multivariate, vector-valued function, $\mathbf{f}:\mathbb{R}^{n}\to\mathbb{R}^{m}$.
/// * `func_name` - Name of the function that will return the mixed second-order partial derivative
///   of $\mathbf{f}(\mathbf{x})$ with respect to $x_{i}$ and $x_{j}$ at any point
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
/// The function produced by this macro will perform 1 evaluation of $\mathbf{f}(\mathbf{x})$ to
/// evaluate its mixed second-order partial derivative with respect to $x_{i}$ and $x_{j}$.
///
/// # Examples
///
/// ## Basic Example
///
/// Compute the mixed second-order partial derivative of the vector-valued function
///
/// $$\mathbf{f}(x,y) = \begin{bmatrix}x^4 + 2x^2y \\\ y^3 + xy^2 \end{bmatrix}$$
///
/// with respect to $x$ and $y$ at $(x,y)=(2,1)$, and compare the result to the true derivative
///
/// $$\frac{\partial^2 \mathbf{f}}{\partial x \partial y} = \begin{bmatrix}4x \\\ 2y \end{bmatrix}$$
///
/// which evaluates to $[8, 2]$ at $(2,1)$.
///
/// ```
/// use linalg_traits::{Scalar, Vector};
/// use numtest::*;
///
/// use numdiff::{get_mixed_vpartial_derivative2, HyperDual, HyperDualVector};
///
/// // Define the vector-valued function, f(x).
/// fn f<S: Scalar, V: Vector<S>>(x: &V, _p: &[f64]) -> Vec<S> {
///     let f0 = x[0].powi(4) + S::new(2.0) * x[0].powi(2) * x[1];
///     let f1 = x[1].powi(3) + x[0] * x[1].powi(2);
///     vec![f0, f1]
/// }
///
/// // Define the evaluation point.
/// let x0 = vec![2.0, 1.0];
///
/// // Autogenerate the mixed second-order partial derivative function.
/// get_mixed_vpartial_derivative2!(f, d2fij);
///
/// // Compute the mixed partial derivative with respect to x₀ and x₁.
/// let d2f_dx0dx1 = d2fij(&x0, 0, 1, &[]);
///
/// // True result: [8, 2].
/// assert_eq!(d2f_dx0dx1[0], 8.0);
/// assert_eq!(d2f_dx0dx1[1], 2.0);
/// ```
///
/// ## Example Passing Runtime Parameters
///
/// Compute the mixed second-order partial derivative of a parameterized vector-valued function
///
/// $$\mathbf{f}(\mathbf{x})=\begin{bmatrix}ax_{0}^{2}x_{1}+bx_{0}x_{1}^{2}+c\sin(dx_{0}x_{1}) \\\ x_{0}x_{1}\end{bmatrix}$$
///
/// where $a$, $b$, $c$, and $d$ are runtime parameters.
///
/// ```
/// use linalg_traits::{Scalar, Vector};
/// use numtest::*;
///
/// use numdiff::{get_mixed_vpartial_derivative2, HyperDual, HyperDualVector};
///
/// // Define the vector-valued function, f(x).
/// fn f<S: Scalar, V: Vector<S>>(x: &V, p: &[f64]) -> Vec<S> {
///     let a = S::new(p[0]);
///     let b = S::new(p[1]);
///     let c = S::new(p[2]);
///     let d = S::new(p[3]);
///     vec![
///         a * x[0].powi(2) * x[1] + b * x[0] * x[1].powi(2)
///             + c * (d * x[0] * x[1]).sin(),
///         x[0] * x[1],
///     ]
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
/// get_mixed_vpartial_derivative2!(f, d2fij);
///
/// // True mixed second-order partial derivative.
/// let theta: f64 = d * x0[0] * x0[1];
/// let expected: Vec<f64> = vec![
///     2.0 * a * x0[0] + 2.0 * b * x0[1] + c * d * theta.cos()
///         - c * d * d * x0[0] * x0[1] * theta.sin(),
///     1.0,
/// ];
///
/// // Compute the mixed partial derivative and compare with the true result.
/// let d2f_dx0dx1 = d2fij(&x0, 0, 1, &p);
/// assert_equal_to_decimal!(d2f_dx0dx1[0], expected[0], 14);
/// assert_eq!(d2f_dx0dx1[1], expected[1]);
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
/// use numdiff::{get_mixed_vpartial_derivative2, HyperDual, HyperDualVector};
///
/// struct Data {
///     a: f64,
///     b: f64,
///     c: f64,
///     d: f64,
/// }
///
/// // Define the vector-valued function, f(x).
/// fn f<S: Scalar, V: Vector<S>>(x: &V, p: &Data) -> Vec<S> {
///     let a = S::new(p.a);
///     let b = S::new(p.b);
///     let c = S::new(p.c);
///     let d = S::new(p.d);
///     vec![
///         a * x[0].powi(2) * x[1] + b * x[0] * x[1].powi(2)
///             + c * (d * x[0] * x[1]).sin(),
///         x[0] * x[1],
///     ]
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
/// get_mixed_vpartial_derivative2!(f, d2fij, Data);
///
/// // True mixed second-order partial derivative.
/// let theta: f64 = p.d * x0[0] * x0[1];
/// let expected: Vec<f64> = vec![
///     2.0 * p.a * x0[0] + 2.0 * p.b * x0[1] + p.c * p.d * theta.cos()
///         - p.c * p.d * p.d * x0[0] * x0[1] * theta.sin(),
///     1.0,
/// ];
///
/// // Compute the mixed partial derivative and compare with the true result.
/// let d2f_dx0dx1 = d2fij(&x0, 0, 1, &p);
/// assert_equal_to_decimal!(d2f_dx0dx1[0], expected[0], 14);
/// assert_eq!(d2f_dx0dx1[1], expected[1]);
/// ```
#[macro_export]
macro_rules! get_mixed_vpartial_derivative2 {
    ($f:ident, $func_name:ident) => {
        get_mixed_vpartial_derivative2!($f, $func_name, [f64]);
    };
    ($f:ident, $func_name:ident, $param_type:ty) => {
        /// Mixed second-order partial derivative of a multivariate, vector-valued function
        /// `f: ℝⁿ → ℝᵐ`.
        ///
        /// This function is generated for a specific function `f` using the
        /// `numdiff::get_mixed_vpartial_derivative2!` macro.
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
        /// `(∂²f/∂xᵢ∂xⱼ)|ₓ₌ₓ₀ ∈ ℝᵐ`
        fn $func_name<S, V>(x0: &V, i: usize, j: usize, p: &$param_type) -> V::DVectorf64
        where
            S: Scalar,
            V: Vector<S>,
        {
            // Promote the evaluation point to a vector of hyper-dual numbers.
            let mut x0_hyperdual = x0.clone().to_hyper_dual_vector();

            if i == j {
                // For i == j, seed both hyper-dual directions on the same variable.
                let original = x0_hyperdual[i];
                x0_hyperdual[i] = HyperDual::new(original.get_a(), 1.0, 1.0, 0.0);
            } else {
                // Seed separate hyper-dual directions for each variable.
                let original_i = x0_hyperdual[i];
                let original_j = x0_hyperdual[j];
                x0_hyperdual[i] = HyperDual::new(original_i.get_a(), 1.0, 0.0, 0.0);
                x0_hyperdual[j] = HyperDual::new(original_j.get_a(), 0.0, 1.0, 0.0);
            }

            // Evaluate the function at the hyper-dual point.
            let f_x0 = $f(&x0_hyperdual, p);

            // Extract mixed second-order partial derivatives.
            V::DVectorf64::from_slice(&f_x0.iter().map(|xi| xi.get_d()).collect::<Vec<_>>())
        }
    };
}

#[cfg(test)]
mod tests {
    use crate::{HyperDual, HyperDualVector};
    use linalg_traits::{Scalar, Vector};
    #[cfg(feature = "nalgebra")]
    use nalgebra::SVector;
    use numtest::*;
    use std::f64::consts::PI;

    #[test]
    fn test_mixed_vpartial_derivative2_basic() {
        // Function to take the mixed second-order partial derivative of:
        // f(x₀, x₁) = [x₀⁴ + 2x₀²x₁, x₁³ + x₀x₁²]
        fn f<S: Scalar, V: Vector<S>>(x: &V, _p: &[f64]) -> Vec<S> {
            let f0 = x[0].powi(4) + S::new(2.0) * x[0].powi(2) * x[1];
            let f1 = x[1].powi(3) + x[0] * x[1].powi(2);
            vec![f0, f1]
        }

        // Define the evaluation point (x₀, x₁) = (2.0, 1.0).
        let x0: Vec<f64> = vec![2.0, 1.0];

        // Generate mixed second-order partial derivative function.
        get_mixed_vpartial_derivative2!(f, d2fij);

        // Check ∂²f/∂x₀∂x₁.
        //  --> ∂²f₀/∂x₀∂x₁ = 4x₀ = 8
        //  --> ∂²f₁/∂x₀∂x₁ = 2x₁ = 2
        let d2f_dx0dx1 = d2fij(&x0, 0, 1, &[]);
        assert_equal_to_decimal!(d2f_dx0dx1[0], 8.0, 15);
        assert_equal_to_decimal!(d2f_dx0dx1[1], 2.0, 15);

        // Check ∂²f/∂x₁∂x₀.
        //  --> By symmetry for smooth functions, ∂²f/∂x₁∂x₀ = ∂²f/∂x₀∂x₁
        let d2f_dx1dx0 = d2fij(&x0, 1, 0, &[]);
        assert_equal_to_decimal!(d2f_dx1dx0[0], 8.0, 15);
        assert_equal_to_decimal!(d2f_dx1dx0[1], 2.0, 15);

        // Check i == j case (reduces to pure second partial derivative).
        let d2f_dx0dx0 = d2fij(&x0, 0, 0, &[]);
        assert_equal_to_decimal!(d2f_dx0dx0[0], 52.0, 15);
        assert_equal_to_decimal!(d2f_dx0dx0[1], 0.0, 15);
    }

    #[test]
    #[cfg(feature = "nalgebra")]
    fn test_mixed_vpartial_derivative2_polynomial() {
        // Function to test various polynomial couplings:
        // f(x₀, x₁) = [x₀³x₁², x₀²x₁³, x₀x₁]
        fn f<S: Scalar, V: Vector<S>>(x: &V, _p: &[f64]) -> Vec<S> {
            vec![
                x[0].powi(3) * x[1].powi(2),
                x[0].powi(2) * x[1].powi(3),
                x[0] * x[1],
            ]
        }

        // Define the evaluation point (x₀, x₁) = (2.0, 1.0).
        let x0: Vec<f64> = vec![2.0, 1.0];

        // Generate mixed second-order partial derivative function.
        get_mixed_vpartial_derivative2!(f, d2fij);

        // Check ∂²f/∂x₀∂x₁.
        //  --> ∂²f₀/∂x₀∂x₁ = 6x₀²x₁ = 24
        //  --> ∂²f₁/∂x₀∂x₁ = 6x₀x₁² = 12
        //  --> ∂²f₂/∂x₀∂x₁ = 1
        let result = d2fij(&x0, 0, 1, &[]);
        assert_eq!(result[0], 24.0);
        assert_eq!(result[1], 12.0);
        assert_eq!(result[2], 1.0);
    }

    #[test]
    fn test_mixed_vpartial_derivative2_multivariate() {
        // Function to take mixed second-order partial derivatives of:
        // f(x₀, x₁, x₂) = [x₀x₁x₂, x₀²x₁ + x₁²x₂, x₀x₂²]
        fn f<S: Scalar, V: Vector<S>>(x: &V, _p: &[f64]) -> Vec<S> {
            vec![
                x[0] * x[1] * x[2],
                x[0].powi(2) * x[1] + x[1].powi(2) * x[2],
                x[0] * x[2].powi(2),
            ]
        }

        // Define the evaluation point (x₀, x₁, x₂) = (1.0, 2.0, 3.0).
        let x0: Vec<f64> = vec![1.0, 2.0, 3.0];

        // Generate mixed second-order partial derivative function.
        get_mixed_vpartial_derivative2!(f, d2fij);

        // Check ∂²f/∂x₀∂x₁.
        //  --> [x₂, 2x₀, 0] = [3, 2, 0]
        let result_x0x1 = d2fij(&x0, 0, 1, &[]);
        assert_eq!(result_x0x1[0], 3.0);
        assert_eq!(result_x0x1[1], 2.0);
        assert_eq!(result_x0x1[2], 0.0);

        // Check ∂²f/∂x₁∂x₂.
        //  --> [x₀, 2x₁, 0] = [1, 4, 0]
        let result_x1x2 = d2fij(&x0, 1, 2, &[]);
        assert_eq!(result_x1x2[0], 1.0);
        assert_eq!(result_x1x2[1], 4.0);
        assert_eq!(result_x1x2[2], 0.0);

        // Check ∂²f/∂x₀∂x₂.
        //  --> [x₁, 0, 2x₂] = [2, 0, 6]
        let result_x0x2 = d2fij(&x0, 0, 2, &[]);
        assert_eq!(result_x0x2[0], 2.0);
        assert_eq!(result_x0x2[1], 0.0);
        assert_eq!(result_x0x2[2], 6.0);
    }

    #[test]
    fn test_mixed_vpartial_derivative2_trig() {
        // Function to take mixed second-order partial derivatives of:
        // f(x₀, x₁) = [sin(x₀x₁), cos(x₁)]
        fn f<S: Scalar, V: Vector<S>>(x: &V, _p: &[f64]) -> Vec<S> {
            vec![(x[0] * x[1]).sin(), x[1].cos()]
        }

        // Define the evaluation point (x₀, x₁) = (π/2, π/4).
        let x0: Vec<f64> = vec![PI / 2.0, PI / 4.0];

        // Generate mixed second-order partial derivative function.
        get_mixed_vpartial_derivative2!(f, d2fij);

        // Check ∂²f/∂x₀∂x₁.
        //  --> ∂²f₀/∂x₀∂x₁ = cos(x₀x₁) - x₀x₁ sin(x₀x₁)
        //  --> ∂²f₁/∂x₀∂x₁ = 0
        let expected0 = (x0[0] * x0[1]).cos() - x0[0] * x0[1] * (x0[0] * x0[1]).sin();
        let result_x0x1 = d2fij(&x0, 0, 1, &[]);
        assert_equal_to_decimal!(result_x0x1[0], expected0, 14);
        assert_eq!(result_x0x1[1], 0.0);
    }

    #[test]
    fn test_mixed_vpartial_derivative2_exponential() {
        // Function to take mixed second-order partial derivatives of:
        // f(x₀, x₁) = [exp(x₀x₁), x₀² + x₁²]
        fn f<S: Scalar, V: Vector<S>>(x: &V, _p: &[f64]) -> Vec<S> {
            vec![(x[0] * x[1]).exp(), x[0].powi(2) + x[1].powi(2)]
        }

        // Define the evaluation point (x₀, x₁) = (1.0, 2.0).
        let x0: Vec<f64> = vec![1.0, 2.0];

        // Generate mixed second-order partial derivative function.
        get_mixed_vpartial_derivative2!(f, d2fij);

        // Check ∂²f/∂x₀∂x₁.
        //  --> ∂²f₀/∂x₀∂x₁ = exp(x₀x₁)(1 + x₀x₁)
        //  --> ∂²f₁/∂x₀∂x₁ = 0
        let expected0 = (x0[0] * x0[1]).exp() * (1.0 + x0[0] * x0[1]);
        let result_x0x1 = d2fij(&x0, 0, 1, &[]);
        assert_equal_to_decimal!(result_x0x1[0], expected0, 14);
        assert_eq!(result_x0x1[1], 0.0);
    }

    #[test]
    #[allow(clippy::many_single_char_names)]
    fn test_mixed_vpartial_derivative2_with_runtime_parameters() {
        // Function to take the mixed second-order partial derivative of:
        // f(x₀, x₁) = [ax₀²x₁ + bx₀x₁² + d sin(ex₀x₁), cx₀x₁ + dx₁²]
        fn f<S: Scalar, V: Vector<S>>(x: &V, p: &[f64]) -> Vec<S> {
            let a = S::new(p[0]);
            let b = S::new(p[1]);
            let c = S::new(p[2]);
            let d = S::new(p[3]);
            let e = S::new(p[4]);
            vec![
                a * x[0].powi(2) * x[1] + b * x[0] * x[1].powi(2) + d * (e * x[0] * x[1]).sin(),
                c * x[0] * x[1] + d * x[1].powi(2),
            ]
        }

        // Runtime parameters.
        let a = 1.5;
        let b = 2.0;
        let c = -0.3;
        let d = 0.4;
        let e = 0.5;
        let p = [a, b, c, d, e];

        // Define the evaluation point (x₀, x₁) = (1.2, -0.7).
        let x0: Vec<f64> = vec![1.2, -0.7];

        // Generate mixed second-order partial derivative function.
        get_mixed_vpartial_derivative2!(f, d2fij);

        // Check ∂²f/∂x₀∂x₁.
        //  --> ∂²f₀/∂x₀∂x₁ = 2ax₀ + 2bx₁ + de cos(ex₀x₁) - de²x₀x₁ sin(ex₀x₁)
        //  --> ∂²f₁/∂x₀∂x₁ = c
        let d2f_dx0dx1 = d2fij(&x0, 0, 1, &p);
        let d2f0_dx0dx1_expected =
            2.0 * a * x0[0] + 2.0 * b * x0[1] + d * e * (e * x0[0] * x0[1]).cos()
                - d * e * e * x0[0] * x0[1] * (e * x0[0] * x0[1]).sin();
        let d2f1_dx0dx1_expected = c;
        assert_equal_to_decimal!(d2f_dx0dx1[0], d2f0_dx0dx1_expected, 14);
        assert_equal_to_decimal!(d2f_dx0dx1[1], d2f1_dx0dx1_expected, 15);

        // Check i == j case (reduces to pure second partial derivative).
        //  --> ∂²f₀/∂x₀² = 2ax₁ - de²x₁² sin(ex₀x₁)
        //  --> ∂²f₁/∂x₀² = 0
        let d2f_dx0dx0 = d2fij(&x0, 0, 0, &p);
        let d2f0_dx0dx0_expected =
            2.0 * a * x0[1] - d * e * e * x0[1] * x0[1] * (e * x0[0] * x0[1]).sin();
        let d2f1_dx0dx0_expected = 0.0;
        assert_equal_to_decimal!(d2f_dx0dx0[0], d2f0_dx0dx0_expected, 14);
        assert_equal_to_decimal!(d2f_dx0dx0[1], d2f1_dx0dx0_expected, 15);
    }

    #[test]
    fn test_mixed_vpartial_derivative2_custom_params() {
        struct Data {
            a: f64,
            b: f64,
            c: f64,
            d: f64,
            e: f64,
        }

        // Function to take the mixed second-order partial derivative of.
        #[allow(clippy::many_single_char_names)]
        fn f<S: Scalar, V: Vector<S>>(x: &V, p: &Data) -> Vec<S> {
            let a = S::new(p.a);
            let b = S::new(p.b);
            let c = S::new(p.c);
            let d = S::new(p.d);
            let e = S::new(p.e);
            vec![
                a * x[0].powi(2) * x[1] + b * x[0] * x[1].powi(2) + d * (e * x[0] * x[1]).sin(),
                c * x[0] * x[1] + d * x[1].powi(2),
            ]
        }

        // Runtime parameter struct.
        let p = Data {
            a: 1.5,
            b: 2.0,
            c: -0.3,
            d: 0.4,
            e: 0.5,
        };

        // Define the evaluation point (x₀, x₁) = (1.2, -0.7).
        let x0: Vec<f64> = vec![1.2, -0.7];

        // Generate mixed second-order partial derivative function.
        get_mixed_vpartial_derivative2!(f, d2fij, Data);

        // Check ∂²f/∂x₀∂x₁.
        //  --> ∂²f₀/∂x₀∂x₁ = 2ax₀ + 2bx₁ + de cos(ex₀x₁) - de²x₀x₁ sin(ex₀x₁)
        //  --> ∂²f₁/∂x₀∂x₁ = c
        let d2f_dx0dx1 = d2fij(&x0, 0, 1, &p);
        let d2f0_dx0dx1_expected =
            2.0 * p.a * x0[0] + 2.0 * p.b * x0[1] + p.d * p.e * (p.e * x0[0] * x0[1]).cos()
                - p.d * p.e * p.e * x0[0] * x0[1] * (p.e * x0[0] * x0[1]).sin();
        let d2f1_dx0dx1_expected = p.c;
        assert_equal_to_decimal!(d2f_dx0dx1[0], d2f0_dx0dx1_expected, 14);
        assert_equal_to_decimal!(d2f_dx0dx1[1], d2f1_dx0dx1_expected, 15);
    }

    #[test]
    #[cfg(feature = "nalgebra")]
    fn test_mixed_vpartial_derivative2_vector_types() {
        // Function to take the mixed second-order partial derivative of:
        // f(x₀, x₁) = [x₀²x₁, x₀x₁²]
        fn f<S: Scalar, V: Vector<S>>(x: &V, _p: &[f64]) -> Vec<S> {
            vec![x[0].powi(2) * x[1], x[0] * x[1].powi(2)]
        }

        // Define the evaluation point (x₀, x₁) = (2.0, 3.0).
        let x_nalgebra: SVector<f64, 2> = SVector::from([2.0, 3.0]);

        // Generate mixed second-order partial derivative function.
        get_mixed_vpartial_derivative2!(f, d2fij);

        // Check ∂²f/∂x₀∂x₁.
        //  --> ∂²f₀/∂x₀∂x₁ = 2x₀ = 4
        //  --> ∂²f₁/∂x₀∂x₁ = 2x₁ = 6
        let result_x0x1 = d2fij(&x_nalgebra, 0, 1, &[]);
        assert_eq!(result_x0x1[0], 4.0);
        assert_eq!(result_x0x1[1], 6.0);

        // Check ∂²f/∂x₁∂x₀.
        //  --> same result by symmetry
        let result_x1x0 = d2fij(&x_nalgebra, 1, 0, &[]);
        assert_eq!(result_x1x0[0], 4.0);
        assert_eq!(result_x1x0[1], 6.0);
    }

    #[test]
    fn test_mixed_vpartial_derivative2_single_component() {
        // Function to take the mixed second-order partial derivative of:
        // f(x₀, x₁) = [x₀²x₁ + x₁²]
        fn f<S: Scalar, V: Vector<S>>(x: &V, _p: &[f64]) -> Vec<S> {
            vec![x[0].powi(2) * x[1] + x[1].powi(2)]
        }

        // Define the evaluation point (x₀, x₁) = (2.0, 1.0).
        let x0: Vec<f64> = vec![2.0, 1.0];

        // Generate mixed second-order partial derivative function.
        get_mixed_vpartial_derivative2!(f, d2fij);

        // Check ∂²f/∂x₀∂x₁.
        //  --> ∂²f₀/∂x₀∂x₁ = 2x₀ = 4
        let result_x0x1 = d2fij(&x0, 0, 1, &[]);
        assert_eq!(result_x0x1[0], 4.0);

        // Check ∂²f/∂x₁∂x₀.
        //  --> ∂²f₀/∂x₁∂x₀ = 2x₀ = 4
        let result_x1x0 = d2fij(&x0, 1, 0, &[]);
        assert_eq!(result_x1x0[0], 4.0);
    }
}
