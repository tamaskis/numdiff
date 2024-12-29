use crate::constants::SQRT_EPS;
use linalg_traits::Vector;

/// Directional derivative of a multivariate, scalar-valued function using the forward difference
/// approximation.
///
/// # Arguments
///
/// * `f` - Multivariate, scalar-valued function, $f:\mathbb{R}^{n}\to\mathbb{R}$.
/// * `x0` - Evaluation point, $\mathbf{x}_{0}\in\mathbb{R}^{n}$.
/// * `v` - Vector defining the direction of differentiation, $\mathbf{v}\in\mathbb{R}^{n}$.
/// * `h` - Relative step size, $h\in\mathbb{R}$. Defaults to [`SQRT_EPS`].
///
/// # Returns
///
/// Directional derivative of $f$ with respect to $\mathbf{x}$ in the direction of $\mathbf{v}$,
/// evaluated at $\mathbf{x}=\mathbf{x}_{0}$.
///
/// $$\nabla_{\mathbf{v}}f(\mathbf{x}_{0})=\nabla f(\mathbf{x}\_{0})^{T}\mathbf{v}\in\mathbb{R}$$
///
/// # Note
///
/// * This function performs 2 evaluations of $f(\mathbf{x})$.
/// * This implementation does _not_ assume that $\mathbf{v}$ is a unit vector.
///
/// # Example
///
/// Approximate the directional derivative of
///
/// $$f(\mathbf{x})=x_{0}^{5}+\sin^{3}{x_{1}}$$
///
/// at $\mathbf{x}=(5,8)^{T}$ in the direction of $\mathbf{v}=(10,20)^{T}$, and compare the result
/// to the true result of
///
/// $$\nabla f_{(10,20)^{T}}\left((5,8)^{T}\right)=31250+60\sin^{2}{(8)}\cos{(8)}$$
///
/// #### Using standard vectors
///
/// ```
/// use numtest::*;
///
/// use numdiff::forward_difference::directional_derivative;
///
/// // Define the function, f(x).
/// let f = |x: &Vec<f64>| x[0].powi(5) + x[1].sin().powi(3);
///
/// // Define the evaluation point.
/// let x0 = vec![5.0, 8.0];
///
/// // Define the direction of differentiation.
/// let v = vec![10.0, 20.0];
///
/// // Approximate the directional derivative of f(x) at the evaluation point along the specified
/// // direction of differentiation.
/// let df_v: f64 = directional_derivative(&f, &x0, &v, None);
///
/// // True directional derivative of f(x) at the evaluation point along the specified direction of
/// // differentiation.
/// let df_v_true: f64 = 31250.0 + 60.0 * 8.0_f64.sin().powi(2) * 8.0_f64.cos();
///
/// // Check the accuracy of the directional derivative approximation.
/// assert_equal_to_decimal!(df_v, df_v_true, 2);
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
/// use numdiff::forward_difference::directional_derivative;
///
/// let df_v_true: f64 = 31250.0 + 60.0 * 8.0_f64.sin().powi(2) * 8.0_f64.cos();
///
/// // nalgebra::DVector
/// let f_dvector = |x: &DVector<f64>| x[0].powi(5) + x[1].sin().powi(3);
/// let x0_dvector: DVector<f64> = dvector![5.0, 8.0];
/// let v_dvector = dvector![10.0, 20.0];
/// let df_v_dvector: f64 = directional_derivative(&f_dvector, &x0_dvector, &v_dvector, None);
/// assert_equal_to_decimal!(df_v_dvector, df_v_true, 2);
///
/// // nalgebra::SVector
/// let f_svector = |x: &SVector<f64,2>| x[0].powi(5) + x[1].sin().powi(3);
/// let x0_svector: SVector<f64, 2> = SVector::from_row_slice(&[5.0, 8.0]);
/// let v_svector: SVector<f64, 2> = SVector::from_row_slice(&[10.0, 20.0]);
/// let df_v_svector: f64 = directional_derivative(&f_svector, &x0_svector, &v_svector, None);
/// assert_equal_to_decimal!(df_v_svector, df_v_true, 2);
///
/// // ndarray::Array1
/// let f_array1 = |x: &Array1<f64>| x[0].powi(5) + x[1].sin().powi(3);
/// let x0_array1: Array1<f64> = array![5.0, 8.0];
/// let v_array1: Array1<f64> = array![10.0, 20.0];
/// let df_v_array1: f64 = directional_derivative(&f_array1, &x0_array1, &v_array1, None);
/// assert_equal_to_decimal!(df_v_array1, df_v_true, 2);
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
/// use numdiff::forward_difference::directional_derivative;
///
/// let f = |x: &Vec<f64>| x[0].powi(5) + x[1].sin().powi(3);
/// let x0 = vec![5.0, 8.0];
/// let v = vec![10.0, 20.0];
///
/// let df_v: f64 = directional_derivative(&f, &x0, &v, Some(0.001));
/// let df_v_true: f64 = 31250.0 + 60.0 * 8.0_f64.sin().powi(2) * 8.0_f64.cos();
///
/// assert_equal_to_decimal!(df_v, df_v_true, -2);
/// ```
pub fn directional_derivative<V>(f: &impl Fn(&V) -> f64, x0: &V, v: &V, h: Option<f64>) -> f64
where
    V: Vector<f64>,
{
    // Default the relative step size to h = √(ε) if not specified.
    let h = h.unwrap_or(*SQRT_EPS);

    // Evaluate the directional derivative.
    (f(&x0.add(&v.mul(h))) - f(x0)) / h
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::SVector;
    use ndarray::{array, Array1};
    use numtest::*;

    #[test]
    fn test_directional_derivative_1() {
        let f = |x: &Vec<f64>| x[0].powi(2);
        let x0 = vec![2.0];
        let v = vec![0.6];
        let df = |x: &Vec<f64>, v: &Vec<f64>| 2.0 * x[0] * v[0];
        assert_equal_to_decimal!(directional_derivative(&f, &x0, &v, None), df(&x0, &v), 7);
    }

    #[test]
    fn test_directional_derivative_2() {
        let f = |x: &SVector<f64, 2>| x[0].powi(2) + x[1].powi(3);
        let x0: SVector<f64, 2> = SVector::from_slice(&[1.0, 2.0]);
        let v: SVector<f64, 2> = SVector::from_slice(&[3.0, 4.0]);
        let df = |x: &SVector<f64, 2>, v: &SVector<f64, 2>| {
            SVector::<f64, 2>::from_slice(&[2.0 * x[0], 3.0 * x[1].powi(2)]).dot(v)
        };
        assert_equal_to_decimal!(directional_derivative(&f, &x0, &v, None), df(&x0, &v), 5);
    }

    #[test]
    fn test_directional_derivative_3() {
        let f = |x: &Array1<f64>| x[0].powi(5) + x[1].sin().powi(3);
        let x0 = array![5.0, 8.0];
        let v = array![10.0, 20.0];
        let df = |x: &Array1<f64>, v: &Array1<f64>| {
            array![5.0 * x[0].powi(4), 3.0 * x[1].sin().powi(2) * x[1].cos()].dot(v)
        };
        assert_equal_to_decimal!(directional_derivative(&f, &x0, &v, None), df(&x0, &v), 2);
    }
}
