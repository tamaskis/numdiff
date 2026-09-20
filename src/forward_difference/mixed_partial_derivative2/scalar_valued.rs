use crate::constants::CBRT_EPS;
use linalg_traits::Vector;

/// Mixed second-order partial derivative of a multivariate, scalar-valued function using the
/// forward difference approximation.
///
/// # Arguments
///
/// * `f` - Multivariate, scalar-valued function, $f:\mathbb{R}^{n}\to\mathbb{R}$.
/// * `x0` - Evaluation point, $\mathbf{x}_{0}\in\mathbb{R}^{n}$.
/// * `i` - First element of $\mathbf{x}$ to differentiate with respect to. Note that this uses
///   0-based indexing (e.g. $\mathbf{x}=\left(x_{0},...,x_{i},...,x_{n-1}\right)^{T}$).
/// * `j` - Second element of $\mathbf{x}$ to differentiate with respect to. Note that this uses
///   0-based indexing (e.g. $\mathbf{x}=\left(x_{0},...,x_{j},...,x_{n-1}\right)^{T}$).
/// * `h` - Relative step size, $h\in\mathbb{R}$. Defaults to [`CBRT_EPS`].
///
/// # Returns
///
/// Mixed second-order partial derivative of $f$ with respect to $x_{i}$ and $x_{j}$, evaluated at
/// $\mathbf{x}=\mathbf{x}_{0}$.
///
/// $$\frac{\partial^{2}f}{\partial x_{i}\partial x_{j}}\bigg\rvert_{\mathbf{x}=\mathbf{x}_{0}}\in\mathbb{R}$$
///
/// # Note
///
/// This function performs 4 evaluations of $f(\mathbf{x})$.
///
/// # Examples
///
/// ## Basic Example
///
/// Approximate the mixed second-order partial derivative of
///
/// $$f(x,y)=x^{4}+2x^{2}y+y^{3}$$
///
/// with respect to $x$ and $y$ at $(x,y)=(2,1)$, and compare the result to the true result of
///
/// $$\frac{\partial^{2}f}{\partial x\partial y}\bigg\rvert_{(x,y)=(2,1)}=4x\bigg\rvert_{(x,y)=(2,1)}=8$$
///
/// ```
/// use numtest::*;
///
/// use numdiff::forward_difference::mixed_spartial_derivative2;
///
/// // Define the function, f(x).
/// let f = |x: &Vec<f64>| x[0].powi(4) + 2.0 * x[0].powi(2) * x[1] + x[1].powi(3);
///
/// // Define the evaluation point.
/// let x0 = vec![2.0, 1.0];
///
/// // Approximate the mixed second-order partial derivative with respect to x₀ and x₁.
/// let d2f: f64 = mixed_spartial_derivative2(&f, &x0, 0, 1, None);
///
/// // Check the accuracy of the mixed second-order partial derivative approximation.
/// assert_equal_to_decimal!(d2f, 8.0, 3);
/// ```
pub fn mixed_spartial_derivative2<V>(
    f: &impl Fn(&V) -> f64,
    x0: &V,
    i: usize,
    j: usize,
    h: Option<f64>,
) -> f64
where
    V: Vector<f64>,
{
    // Copy the evaluation point so that we may modify it.
    let mut x0 = x0.clone();

    // Default the relative step size to h = ε¹ᐟ³ if not specified.
    let h = h.unwrap_or(*CBRT_EPS);

    // Evaluate and store the value of f(x₀).
    let f0 = f(&x0);

    // Original values of the evaluation point in the ith and jth directions.
    let x0i = x0[i];
    let x0j = x0[j];

    // Absolute step sizes in the ith and jth directions.
    let dxi = h * (1.0 + x0i.abs());
    let dxj = h * (1.0 + x0j.abs());

    // Step forward in the ith direction only.
    x0[i] = x0i + dxi;
    let fi = f(&x0);
    x0[i] = x0i;

    // Step forward in the jth direction only.
    x0[j] = x0j + dxj;
    let fj = f(&x0);
    x0[j] = x0j;

    // Step forward in both the ith and jth directions.
    x0[i] += dxi;
    x0[j] += dxj;
    let fij = f(&x0);

    // Evaluate the mixed second-order partial derivative of f with respect to xᵢ and xⱼ.
    (fij - fi - fj + f0) / (dxi * dxj)
}

#[cfg(test)]
mod tests {
    use super::*;
    use numtest::*;

    #[test]
    fn test_mixed_spartial_derivative2_basic() {
        let f = |x: &Vec<f64>| x[0].powi(4) + 2.0 * x[0].powi(2) * x[1] + x[1].powi(3);
        let x0 = vec![2.0, 1.0];
        assert_equal_to_decimal!(mixed_spartial_derivative2(&f, &x0, 0, 1, None), 8.0, 3);
        assert_equal_to_decimal!(mixed_spartial_derivative2(&f, &x0, 1, 0, None), 8.0, 3);
    }

    #[test]
    fn test_mixed_spartial_derivative2_same_index() {
        // When i == j, the mixed partial derivative reduces to the second-order partial
        // derivative.
        let f = |x: &Vec<f64>| x[0].powi(4) + 2.0 * x[0].powi(2) * x[1] + x[1].powi(3);
        let x0 = vec![2.0, 1.0];
        assert_equal_to_decimal!(mixed_spartial_derivative2(&f, &x0, 0, 0, None), 52.0, 2);
        assert_equal_to_decimal!(mixed_spartial_derivative2(&f, &x0, 1, 1, None), 6.0, 3);
    }
}
