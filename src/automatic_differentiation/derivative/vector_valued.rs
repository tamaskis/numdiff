/// TODO.
#[macro_export]
macro_rules! get_vderivative {
    ($func_name:ident, $generic_func:ident) => {
        /// Derivative of a univariate, scalar-valued function using forward-mode automatic
        /// differentiation.
        ///
        /// # Arguments
        ///
        /// * `f` - Univariate, scalar-valued function, $f:\mathbb{R}\to\mathbb{R}$.
        /// * `x0` - Evaluation point, $x_{0}\in\mathbb{R}$.
        ///
        /// # Returns
        ///
        /// Derivative of $f$ with respect to $x$, evaluated at $x=x_{0}$.
        ///
        /// $$\frac{df}{dx}\bigg\rvert_{x=x_{0}}\in\mathbb{R}$$
        ///
        /// // TODO more docs
        fn $func_name<S: Scalar>(value: S) -> DVector<f64> {
            // Cast the value to a different concrete type (T -> ConcreteTypeB)
            let temp_value = Dual::new(value.to_f64().unwrap(), 1.0);

            // Call the passed-in generic function with a reference to the new concrete type
            let result = $generic_func(temp_value); // Now using ConcreteTypeB

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
    use super::*;
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
