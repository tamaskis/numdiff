use crate::constants::CBRT_EPS;
use linalg_traits::Vector;

/// Derivative of a univariate, vector-valued function using the central difference approximation.
///
/// # Arguments
///
/// * `f` - Univariate, vector-valued function, $\mathbf{f}:\mathbb{R}\to\mathbb{R}^{m}$.
/// * `x0` - Evaluation point, $x_{0}\in\mathbb{R}$.
/// * `h` - Relative step size, $h\in\mathbb{R}$. Defaults to [`CBRT_EPS`].
///
/// # Returns
///
/// Derivative of $\mathbf{f}$ with respect to $x$, evaluated at $x=x_{0}$.
///
/// $$\frac{d\mathbf{f}}{dx}\bigg\rvert_{x=x_{0}}\in\mathbb{R}^{m}$$
///
/// # Note
///
/// This function performs 2 evaluations of $f(x)$.
///
/// # Example
///
/// Approximate the derivative of
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
/// use numtest::*;
///
/// use numdiff::central_difference::vderivative;
///
/// // Define the function, f(t).
/// let f = |t: f64| vec![t.sin(), t.cos()];
///
/// // Approximate the derivative of f(t) at the evaluation point.
/// let df: Vec<f64> = vderivative(&f, 1.0, None);
///
/// // True derivative of f(t) at the evaluation point.
/// let df_true: Vec<f64> = vec![1.0_f64.cos(), -1.0_f64.sin()];
///
/// // Check the accuracy of the derivative approximation.
/// assert_arrays_equal_to_decimal!(df, df_true, 10);
/// ```
///
/// #### Using other vector types
///
/// We can also use other types of vectors, such as `nalgebra::SVector`, `nalgebra::DVector`,
/// `ndarray::Array1`, or any other type of vector that implements the `linalg_traits::Vector`
/// trait.
///
/// ```
/// use nalgebra::{dvector, DVector, SVector};
/// use ndarray::{array, Array1};
/// use numtest::*;
///
/// use numdiff::central_difference::vderivative;
///
/// let df_true: Vec<f64> = vec![1.0_f64.cos(), -1.0_f64.sin()];
///
/// // nalgebra::DVector
/// let f_dvector = |t: f64| dvector![t.sin(), t.cos()];
/// let df_dvector: DVector<f64> = vderivative(&f_dvector, 1.0, None);
/// assert_arrays_equal_to_decimal!(df_dvector, df_true, 10);
///
/// // nalgebra::SVector
/// let f_svector = |t: f64| SVector::<f64, 2>::from_row_slice(&[t.sin(), t.cos()]);
/// let df_svector: SVector<f64, 2> = vderivative(&f_svector, 1.0, None);
/// assert_arrays_equal_to_decimal!(df_svector, df_true, 10);
///
/// // ndarray::Array1
/// let f_array1 = |t: f64| array![t.sin(), t.cos()];
/// let df_array1: Array1<f64> = vderivative(&f_array1, 1.0, None);
/// assert_arrays_equal_to_decimal!(df_array1, df_true, 10);
/// ```
///
/// #### Modifying the relative step size
///
/// We can also modify the relative step size. Choosing a coarser relative step size, we get a worse
/// approximation.
///
/// ```
/// use numtest::*;
///
/// use numdiff::central_difference::vderivative;
///
/// let f = |t: f64| vec![t.sin(), t.cos()];
///
/// let df: Vec<f64> = vderivative(&f, 1.0, Some(0.001));
/// let df_true: Vec<f64> = vec![1.0_f64.cos(), -1.0_f64.sin()];
///
/// assert_arrays_equal_to_decimal!(df, df_true, 6);
/// ```
pub fn vderivative<V>(f: &impl Fn(f64) -> V, x0: f64, h: Option<f64>) -> V
where
    V: Vector<f64>,
{
    // Default the relative step size to h = ε¹ᐟ³ if not specified.
    let h = h.unwrap_or(*CBRT_EPS);

    // Absolute step size.
    let dx = h * (1.0 + x0.abs());

    // Evaluate the derivative.
    (f(x0 + dx).sub(&f(x0 - dx))).div(2.0 * dx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use numtest::*;

    #[test]
    fn test_vderivative() {
        let f = |x: f64| vec![x.sin(), x.cos()];
        let x0 = 2.0;
        let df = |x: f64| vec![x.cos(), -x.sin()];
        assert_arrays_equal_to_decimal!(vderivative(&f, x0, None), df(x0), 9);
    }
}
