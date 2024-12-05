/// TODO.
#[macro_export]
macro_rules! get_gradient {
    ($func_name:ident, $generic_func:ident) => {
        /// Gradient of a multivariate, scalar-valued function using the forward difference approximation.
        ///
        /// # Arguments
        ///
        /// * `f` - Multivariate, scalar-valued function, $f:\mathbb{R}^{n}\to\mathbb{R}$.
        /// * `x0` - Evaluation point, $\mathbf{x}_{0}\in\mathbb{R}^{n}$.
        ///
        /// # Returns
        ///
        /// Gradient of $f$ with respect to $\mathbf{x}$, evaluated at $\mathbf{x}=\mathbf{x}_{0}$.
        ///
        /// $$\nabla f(\mathbf{x}_{0})\in\mathbb{R}^{n}$$
        ///
        /// // TODO more docs
        fn $func_name<S, V>(value: S) -> DMatrix<f64>
        where
            S: Scalar,
            V: Vector<S>,
        {
            let temp_value = Dual::new(value.to_f64().unwrap(), 1.0);

            // TODO: need a method to create a vector of dual numbers
            // TODO: need a method to take the dual portion of a vector of dual numbers

            let result = $generic_func(temp_value);

            let mut result_f64 = result.new_vector_f64();
            for i in 0..result_f64.len() {
                result_f64[i] = result[i].get_dual();
            }
            result_f64
        }
    };
}

#[cfg(test)]
mod tests {
    use crate::automatic_differentiation::dual::Dual;
    use linalg_traits::{Scalar, Vector};
    use nalgebra::{dvector, DVector};
    use numtest::*;

    #[test]
    fn test_vderivative() {
        fn f<S: Scalar>(x: S) -> DVector<S> {
            dvector![x.sin(), x.cos()]
        }
        let x0 = 2.0;
        get_vderivative!(df, f);
        let df_actual = |x: f64| vec![x.cos(), -x.sin()];
        assert_arrays_equal!(df(x0), df_actual(x0));
    }
}
