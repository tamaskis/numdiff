use crate::constants::CBRT_EPS;
use linalg_traits::Vector;

/// Second-order partial derivative of a multivariate, vector-valued function using the forward
/// difference approximation.
///
/// # Arguments
///
/// * `f` - Multivariate, vector-valued function, $\mathbf{f}:\mathbb{R}^{n}\to\mathbb{R}^{m}$.
/// * `x0` - Evaluation point, $\mathbf{x}_{0}\in\mathbb{R}^{n}$.
/// * `k` - Element of $\mathbf{x}$ to differentiate with respect to twice. Note that this uses
///   0-based indexing (e.g. $\mathbf{x}=\left(x_{0},...,x_{k},...,x_{n-1}\right)^{T}$).
/// * `h` - Relative step size, $h\in\mathbb{R}$. Defaults to [`CBRT_EPS`].
///
/// # Returns
///
/// Second-order partial derivative of $\mathbf{f}$ with respect to $x_{k}$, evaluated at
/// $\mathbf{x}=\mathbf{x}_{0}$.
///
/// $$\frac{\partial^{2}\mathbf{f}}{\partial x_{k}^{2}}\bigg\rvert_{\mathbf{x}=\mathbf{x}_{0}}\in\mathbb{R}^{m}$$
///
/// # Note
///
/// This function performs 3 evaluations of $\mathbf{f}(\mathbf{x})$.
///
/// # Examples
///
/// ## Basic Example
///
/// Approximate the second-order partial derivative of
///
/// $$\mathbf{f}(x,y)=\begin{bmatrix}x^{4}+2x^{2}y\\\\y^{3}+xy^{2}\end{bmatrix}$$
///
/// with respect to $x$ at $(x,y)=(2,1)$, and compare the result to the true result of
///
/// $$\frac{\partial^{2}\mathbf{f}}{\partial x^{2}}\bigg\rvert_{(x,y)=(2,1)}=\begin{bmatrix}12x^{2}+4y\\\\0\end{bmatrix}\bigg\rvert_{(x,y)=(2,1)}=\begin{bmatrix}52\\\\0\end{bmatrix}$$
///
/// ```
/// use numtest::*;
///
/// use numdiff::forward_difference::vpartial_derivative2;
///
/// // Define the function, f(x).
/// let f = |x: &Vec<f64>| {
///     vec![
///         x[0].powi(4) + 2.0 * x[0].powi(2) * x[1],
///         x[1].powi(3) + x[0] * x[1].powi(2),
///     ]
/// };
///
/// // Define the evaluation point.
/// let x0 = vec![2.0, 1.0];
///
/// // Define the element of the vector (using 0-based indexing) we are differentiating with
/// // respect to.
/// let k = 0;
///
/// // Approximate the second-order partial derivative of f(x) with respect to x₀.
/// let d2f: Vec<f64> = vpartial_derivative2(&f, &x0, k, None);
///
/// // Check the accuracy of the second-order partial derivative approximation.
/// assert_arrays_equal_to_decimal!(d2f, vec![52.0, 0.0], 2);
/// ```
pub fn vpartial_derivative2<V, U>(f: &impl Fn(&V) -> U, x0: &V, k: usize, h: Option<f64>) -> U
where
    V: Vector<f64>,
    U: Vector<f64>,
{
    // Copy the evaluation point so that we may modify it.
    let mut x0 = x0.clone();

    // Default the relative step size to h = ε¹ᐟ³ if not specified.
    let h = h.unwrap_or(*CBRT_EPS);

    // Evaluate and store the value of f(x₀).
    let f0 = f(&x0);

    // Original value of the evaluation point in the kth direction.
    let x0k = x0[k];

    // Absolute step size in the kth direction.
    let dxk = h * (1.0 + x0k.abs());

    // Step forward once in the kth direction.
    x0[k] = x0k + dxk;
    let f1 = f(&x0);

    // Step forward again in the kth direction.
    x0[k] = x0k + 2.0 * dxk;
    let f2 = f(&x0);

    // Evaluate the second-order partial derivative of f with respect to xₖ.
    (f2.sub(&f1.mul(2.0)).add(&f0)).div(dxk.powi(2))
}

#[cfg(test)]
mod tests {
    use super::*;
    use numtest::*;

    #[test]
    fn test_vpartial_derivative2() {
        let f = |x: &Vec<f64>| {
            vec![
                x[0].powi(4) + 2.0 * x[0].powi(2) * x[1],
                x[1].powi(3) + x[0] * x[1].powi(2),
            ]
        };
        let x0 = vec![2.0, 1.0];
        assert_arrays_equal_to_decimal!(vpartial_derivative2(&f, &x0, 0, None), [52.0, 0.0], 2);
        assert_arrays_equal_to_decimal!(
            vpartial_derivative2(&f, &x0, 1, None),
            [0.0, 6.0 * x0[1] + 2.0 * x0[0]],
            2
        );
    }
}
