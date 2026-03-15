/// Get a function that returns the second derivative of the provided univariate, vector-valued
/// function.
///
/// The second derivative is computed using forward-mode automatic differentiation.
///
/// # Arguments
///
/// * `f` - Univariate, vector-valued function, $\mathbb{f}:\mathbb{R}\to\mathbb{R}^{m}$.
/// * `func_name` - Name of the function that will return the second derivative of $\mathbf{f}(x)$
///   at any point $x\in\mathbb{R}$.
/// * `param_type` (optional) - Type of the extra runtime parameter `p` that is passed to `f`.
///   Defaults to `[f64]` (implying that `f` accepts `p: &[f64]`).
///
/// # Warning
///
/// `f` cannot be defined as closure. It must be defined as a function.
///
/// # Note
///
/// The function produced by this macro will perform 1 evaluation of $\mathbf{f}(x)$ to evaluate its
/// second derivative.
///
/// # Examples
///
/// ## Basic Example
///
/// Compute the second derivative of
///
/// $$f(t)=\begin{bmatrix}\sin{t}\\\\\cos{t}\end{bmatrix}$$
///
/// at $t=1$, and compare the result to the true result of
///
/// $$\frac{d^{2}\mathbf{f}}{dt^{2}}\bigg\rvert_{t=1}=\begin{bmatrix}-\sin{(1)}\\\\-\cos{(1)}\end{bmatrix}$$
///
/// #### Using standard vectors
///
/// ```
/// use linalg_traits::{Scalar, Vector};
/// use numtest::*;
///
/// use numdiff::{get_vderivative2, HyperDual};
///
/// // Define the function, f(t).
/// fn f<S: Scalar, V: Vector<S>>(t: S, _p: &[f64]) -> V {
///     V::from_slice(&[t.sin(), t.cos()])
/// }
///
/// // Autogenerate the function "d2f" that can be used to compute the second derivative of f(t) at
/// // any point t.
/// get_vderivative2!(f, d2f);
///
/// // Compute the second derivative of f(t) at the evaluation point, t = 1.
/// let d2f_at_1 = d2f::<f64, Vec<f64>>(1.0, &[]);
///
/// // True second derivative of f(t) at the evaluation point.
/// let d2f_at_1_true: Vec<f64> = vec![-1.0_f64.sin(), -1.0_f64.cos()];
///
/// // Check the accuracy of the second derivative.
/// assert_arrays_equal_to_decimal!(d2f_at_1, d2f_at_1_true, 16);
/// ```
///
/// #### Using other vector types
///
/// The function produced by `get_vderivative2!` can accept _any_ type for `x0`, as long as it
/// implements the `linalg_traits::Vector` trait.
///
/// ```
/// use faer::Mat;
/// use linalg_traits::{Scalar, Vector};
/// use nalgebra::{dvector, DVector, SVector};
/// use ndarray::{array, Array1};
/// use numtest::*;
///
///  use numdiff::{get_vderivative2, HyperDual};
///
/// // Define the function, f(t).
/// fn f<S: Scalar, V: Vector<S>>(t: S, _p: &[f64]) -> V {
///     V::from_slice(&[t.sin(), t.cos()])
/// }
///
/// // Autogenerate the function "d2f" that can be used to compute the second derivative of f(t) at
/// // any point t.
/// get_vderivative2!(f, d2f);
///
/// // True second derivative of f(t) at the evaluation point.
/// let d2f_at_1_true: Vec<f64> = vec![-1.0_f64.sin(), -1.0_f64.cos()];
///
/// // nalgebra::DVector
/// let d2f_at_1_dvector: DVector<f64> = d2f::<f64, DVector<f64>>(1.0, &[]);
/// assert_arrays_equal_to_decimal!(d2f_at_1_dvector, d2f_at_1_true, 16);
///
/// // nalgebra::SVector
/// let d2f_at_1_svector: SVector<f64, 2> = d2f::<f64, SVector<f64, 2>>(1.0, &[]);
/// assert_arrays_equal_to_decimal!(d2f_at_1_svector, d2f_at_1_true, 16);
///
/// // ndarray::Array1
/// let d2f_at_1_array1: Array1<f64> = d2f::<f64, Array1<f64>>(1.0, &[]);
/// assert_arrays_equal_to_decimal!(d2f_at_1_array1, d2f_at_1_true, 16);
///
/// // faer::Mat
/// let d2f_at_1_mat: Mat<f64> = d2f::<f64, Mat<f64>>(1.0, &[]);
/// assert_arrays_equal_to_decimal!(d2f_at_1_mat.as_slice(), d2f_at_1_true, 16);
/// ```
///
/// ## Example Passing Runtime Parameters
///
/// Compute the second derivative of a parameterized vector function
///
/// $$f(t)=\begin{bmatrix}at^{2}+b\\\\ce^{t}+d\end{bmatrix}$$
///
/// where $a$, $b$, $c$, and $d$ are runtime parameters. Compare the result against the true second
/// derivative of
///
/// $$f''(t)=\begin{bmatrix}2a\\\\ce^{t}\end{bmatrix}$$
///
/// ```
/// use linalg_traits::{Scalar, Vector};
/// use numtest::*;
///
/// use numdiff::{get_vderivative2, HyperDual};
///
/// // Define the function, f(x).
/// fn f<S: Scalar, V: Vector<S>>(t: S, p: &[f64]) -> V {
///     let a = S::new(p[0]);
///     let b = S::new(p[1]);
///     let c = S::new(p[2]);
///     let d = S::new(p[3]);
///     V::from_slice(&[a * t.powi(2) + b, c * t.exp() + d])
/// }
///
/// // Parameter vector.
/// let a = 1.5;
/// let b = -2.0;
/// let c = 0.8;
/// let d = 3.0;
/// let p = [a, b, c, d];
///
/// // Autogenerate the second derivative function.
/// get_vderivative2!(f, d2f);
///
/// // True second derivative function.
/// let d2f_true = |t: f64| vec![2.0 * a, c * t.exp()];
///
/// // Compute the second derivative at t = 1.0 using both the automatically generated second
/// // derivative function and the true second derivative function, and compare the results.
/// let d2f_at_1: Vec<f64> = d2f::<f64, Vec<f64>>(1.0, &p);
/// let d2f_at_1_true: Vec<f64> = d2f_true(1.0);
/// assert_arrays_equal_to_decimal!(d2f_at_1, d2f_at_1_true, 15);
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
/// use numdiff::{get_vderivative2, HyperDual};
///
/// struct Data {
///     a: f64,
///     b: f64,
///     c: f64,
///     d: f64,
/// }
///
/// // Define the function, f(t).
/// fn f<S: Scalar, V: Vector<S>>(t: S, p: &Data) -> V {
///     let a = S::new(p.a);
///     let b = S::new(p.b);
///     let c = S::new(p.c);
///     let d = S::new(p.d);
///     V::from_slice(&[a * t.powi(2) + b, c * t.exp() + d])
/// }
///
/// // Runtime parameter struct.
/// let p = Data {
///     a: 1.5,
///     b: -2.0,
///     c: 0.8,
///     d: 3.0,
/// };
///
/// // Autogenerate the second derivative function, telling the macro to expect a runtime parameter
/// // of type &Data.
/// get_vderivative2!(f, d2f, Data);
///
/// // True second derivative function.
/// let d2f_true = |t: f64| vec![2.0 * p.a, p.c * t.exp()];
///
/// // Compute the second derivative at t = 1.0 using both the automatically generated second
/// // derivative function and the true second derivative function, and compare the results.
/// let d2f_at_1: Vec<f64> = d2f::<f64, Vec<f64>>(1.0, &p);
/// let d2f_at_1_true: Vec<f64> = d2f_true(1.0);
/// assert_arrays_equal_to_decimal!(d2f_at_1, d2f_at_1_true, 15);
/// ```
#[macro_export]
macro_rules! get_vderivative2 {
    ($f:ident, $func_name:ident) => {
        get_vderivative2!($f, $func_name, [f64]);
    };
    ($f:ident, $func_name:ident, $param_type:ty) => {
        /// Second derivative of a univariate, vector-valued function `f: ℝ → ℝᵐ`.
        ///
        /// This function is generated for a specific function `f` using the
        /// `numdiff::get_vderivative2!` macro.
        ///
        /// # Arguments
        ///
        /// * `x0` - Evaluation point, `x₀ ∈ ℝ`.
        /// * `p` - Extra runtime parameter. This is a parameter (can be of any arbitrary type)
        ///   defined at runtime that the function may depend on but is not differentiated with
        ///   respect to.
        ///
        /// # Returns
        ///
        /// Second derivative of `f` with respect to `x`, evaluated at `x = x₀`.
        ///
        /// `(d²f/dx²)|ₓ₌ₓ₀ ∈ ℝᵐ`
        fn $func_name<S: Scalar, V: Vector<S>>(value: S, p: &$param_type) -> V::Vectorf64 {
            // Step forward in both hyper-dual directions.
            let temp_value = HyperDual::new(value.to_f64().unwrap(), 1.0, 1.0, 0.0);

            // Evaluate the function at the hyper-dual number.
            let f_x0: V::VectorT<HyperDual> = $f(temp_value, p);

            // Extract second derivatives from each component of the result.
            let mut d2f = V::Vectorf64::new_with_length(f_x0.len());
            for i in 0..d2f.len() {
                d2f.vset(i, f_x0.vget(i).get_d());
            }

            // Second derivative of f with respect to x evaluated at x = x₀.
            d2f
        }
    };
}

#[cfg(test)]
mod tests {
    use crate::HyperDual;
    use linalg_traits::{Scalar, Vector};
    use numtest::*;

    #[test]
    fn test_vderivative2_1() {
        fn f<S: Scalar, V: Vector<S>>(x: S, _p: &[f64]) -> V {
            V::from_slice(&[x.sin(), x.cos()])
        }
        let x0 = 2.0;
        get_vderivative2!(f, d2f);
        let d2f_actual = |x: f64| vec![-x.sin(), -x.cos()];
        assert_arrays_equal!(d2f::<f64, Vec<f64>>(x0, &[]), d2f_actual(x0));
    }

    #[test]
    fn test_vderivative2_2() {
        // Function to take the second derivative of.
        #[allow(clippy::many_single_char_names)]
        fn f<S: Scalar, V: Vector<S>>(x: S, p: &[f64]) -> V {
            let a = S::new(p[0]);
            let b = S::new(p[1]);
            let c = S::new(p[2]);
            let d = S::new(p[3]);
            V::from_slice(&[a * (b * x).sin(), c * (d * x).cos()])
        }

        // True second derivative function.
        #[allow(clippy::many_single_char_names)]
        fn d2f(x: f64, p: &[f64]) -> Vec<f64> {
            let a = p[0];
            let b = p[1];
            let c = p[2];
            let d = p[3];
            vec![-a * b * b * (b * x).sin(), -c * d * d * (d * x).cos()]
        }

        // Evaluation point.
        let x0 = 0.2;

        // Parameter vector.
        let p = [2.0, 3.0, 1.5, 0.5];

        // Second derivative function obtained via forward-mode automatic differentation.
        get_vderivative2!(f, d2f_autodiff);

        // Evaluate the second derivative using both functions.
        let d2f_eval_autodiff: Vec<f64> = d2f_autodiff::<f64, Vec<f64>>(x0, &p);
        let d2f_eval: Vec<f64> = d2f(x0, &p);

        // Test autodiff second derivative against true second derivative.
        assert_arrays_equal!(d2f_eval_autodiff, d2f_eval);
    }

    #[test]
    fn test_vderivative2_custom_params() {
        struct Data {
            a: f64,
            b: f64,
            c: f64,
            d: f64,
        }

        // Function to take the second derivative of.
        #[allow(clippy::many_single_char_names)]
        fn f<S: Scalar, V: Vector<S>>(t: S, p: &Data) -> V {
            let a = S::new(p.a);
            let b = S::new(p.b);
            let c = S::new(p.c);
            let d = S::new(p.d);
            V::from_slice(&[a * t.powi(2) + b, c * t.exp() + d])
        }

        // Runtime parameter struct.
        let p = Data {
            a: 1.5,
            b: -2.0,
            c: 0.8,
            d: 3.0,
        };

        // Second derivative function obtained via forward-mode automatic differentiation.
        get_vderivative2!(f, d2f, Data);

        // True second derivative function.
        let d2f_true = |t: f64| vec![2.0 * p.a, p.c * t.exp()];

        // Evaluation point.
        let t0 = 1.0;

        // Evaluate the second derivative using both functions.
        let d2f_eval: Vec<f64> = d2f::<f64, Vec<f64>>(t0, &p);
        let d2f_eval_true: Vec<f64> = d2f_true(t0);

        // Test autodiff second derivative against true second derivative.
        assert_arrays_equal_to_decimal!(d2f_eval, d2f_eval_true, 15);
    }
}
