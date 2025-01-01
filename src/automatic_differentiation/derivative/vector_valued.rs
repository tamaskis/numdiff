/// Get a function that returns the derivative of the provided univariate, vector-valued function.
///
/// The derivative is computed using forward-mode automatic differentiation.
///
/// # Arguments
///
/// * `f` - Univariate, vector-valued function, $\mathbb{f}:\mathbb{R}\to\mathbb{R}^{m}$.
/// * `func_name` - Name of the function that will return the derivative of $\mathbf{f}(x)$ at any
///                 point $x\in\mathbb{R}$.
///
/// # Warning
///
/// `f` cannot be defined as closure. It must be defined as a function.
///
/// # Note
///
/// The function produced by this macro will perform 1 evaluation of $\mathbf{f}(x)$ to evaluate its
/// derivative.
///
/// # Example
///
/// Compute the derivative of
///
/// $$f(t)=\begin{bmatrix}\sin{t}\\\\\cos{t}\end{bmatrix}$$
///
/// at $t=1$, and compare the result to the true result of
///
/// $$\frac{d\mathbf{f}}{dt}\bigg\rvert_{t=1}=\begin{bmatrix}\cos{(1)}\\\\-\sin{(1)}\end{bmatrix}$$
///
/// #### Using standard vectors
///
/// ```
/// use linalg_traits::{Scalar, Vector};
/// use numtest::*;
///
/// use numdiff::{get_vderivative, Dual};
///
/// // Define the function, f(t).
/// fn f<S: Scalar, V: Vector<S>>(t: S) -> V {
///     V::from_slice(&[t.sin(), t.cos()])
/// }
///
/// // Autogenerate the function "df" that can be used to compute the derivative of f(t) at any
/// // point t.
/// get_vderivative!(f, df);
///
/// // Compute the derivative of f(t) at the evaluation point, t = 1.
/// let df_at_1 = df::<f64, Vec<f64>>(1.0);
///
/// // True derivative of f(t) at the evaluation point.
/// let df_at_1_true: Vec<f64> = vec![1.0_f64.cos(), -1.0_f64.sin()];
///
/// // Check the accuracy of the derivative.
/// assert_arrays_equal_to_decimal!(df_at_1, df_at_1_true, 16);
/// ```
///
/// #### Using other vector types
///
/// We can also use other types of vectors, such as `nalgebra::SVector`, `nalgebra::DVector`,
/// `ndarray::Array1`, or any other type of vector that implements the `linalg_traits::Vector`
/// trait.
///
/// ```
/// use linalg_traits::{Scalar, Vector};
/// use nalgebra::{dvector, DVector, SVector};
/// use ndarray::{array, Array1};
/// use numtest::*;
///
///  use numdiff::{get_vderivative, Dual};
///
/// // Define the function, f(t).
/// fn f<S: Scalar, V: Vector<S>>(t: S) -> V {
///     V::from_slice(&[t.sin(), t.cos()])
/// }
///
/// // Autogenerate the function "df" that can be used to compute the derivative of f(t) at any
/// // point t.
/// get_vderivative!(f, df);
///
/// // True derivative of f(t) at the evaluation point.
/// let df_at_1_true: Vec<f64> = vec![1.0_f64.cos(), -1.0_f64.sin()];
///
/// // nalgebra::DVector
/// let df_at_1_dvector: DVector<f64> = df::<f64, DVector<f64>>(1.0);
/// assert_arrays_equal_to_decimal!(df_at_1_dvector, df_at_1_true, 16);
///
/// // nalgebra::SVector
/// let df_at_1_svector: SVector<f64, 2> = df::<f64, SVector<f64, 2>>(1.0);
/// assert_arrays_equal_to_decimal!(df_at_1_svector, df_at_1_true, 16);
///
/// // ndarray::Array1
/// let df_at_1_array1: Array1<f64> = df::<f64, Array1<f64>>(1.0);
/// assert_arrays_equal_to_decimal!(df_at_1_array1, df_at_1_true, 16);
/// ```
#[macro_export]
macro_rules! get_vderivative {
    ($f:ident, $func_name:ident) => {
        /// Derivative of a univariate, vector-valued function `f: ℝ → ℝᵐ`.
        ///
        /// This function is generated for a specific function `f` using the
        /// `numdiff::get_vderivative!` macro.
        ///
        /// # Arguments
        ///
        /// * `x0` - Evaluation point, `x₀ ∈ ℝ`.
        ///
        /// # Returns
        ///
        /// Derivative of `f` with respect to `x`, evaluated at `x = x₀`.
        ///
        /// `(df/dx)|ₓ₌ₓ₀ ∈ ℝᵐ`
        fn $func_name<S: Scalar, V: Vector<S>>(value: S) -> V::Vectorf64 {
            let temp_value = Dual::new(value.to_f64().unwrap(), 1.0);

            let f_x0: V::VectorT<Dual> = $f(temp_value);

            let mut df = V::Vectorf64::new_with_length(f_x0.len());
            for i in 0..df.len() {
                df[i] = f_x0[i].get_dual();
            }
            df
        }
    };
}

#[cfg(test)]
mod tests {
    use crate::automatic_differentiation::dual::Dual;
    use linalg_traits::{Scalar, Vector};
    use numtest::*;

    #[test]
    fn test_vderivative() {
        fn f<S: Scalar, V: Vector<S>>(x: S) -> V {
            V::from_slice(&[x.sin(), x.cos()])
        }
        let x0 = 2.0;
        get_vderivative!(f, df);
        let df_actual = |x: f64| vec![x.cos(), -x.sin()];
        assert_arrays_equal!(df::<f64, Vec<f64>>(x0), df_actual(x0));
    }
}
