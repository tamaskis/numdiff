/// TODO.
#[macro_export]
macro_rules! get_sderivative {
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
        fn $func_name<S: Scalar>(value: S) -> f64 {
            let temp_value = Dual::new(value.to_f64().unwrap(), 1.0);
            let result = $generic_func(temp_value);
            result.get_dual()
        }
    };
}

#[cfg(test)]
mod tests {
    use crate::automatic_differentiation::dual::Dual;
    use crate::test_utils;
    use linalg_traits::Scalar;
    use numtest::*;

    pub fn func<S: Scalar>(x: S) -> S {
        x.powi(2)
    }

    #[test]
    fn example_3() {
        get_sderivative!(derivative, func);
        println!("{:?}", derivative(5.0));
        assert_eq!(derivative(5.0), 10.0);
    }

    #[test]
    fn test_sderivative_polynomial() {
        // assert_equal_to_decimal!(
        //     let f = |
        //     sderivative(&|_x: f64| 1.0, 2.0, None),
        //     test_utils::polyi_deriv(0, 2.0),
        //     16
        // );

        // Test #2.
        fn f2<S: Scalar>(x: S) -> S {
            x
        }
        get_sderivative!(df2, f2);
        assert_eq!(df2(2.0), test_utils::polyi_deriv(1, 2.0));

        // Test #3.
        fn f3<S: Scalar>(x: S) -> S {
            x.powi(2)
        }
        get_sderivative!(df3, f3);
        assert_eq!(df3(2.0), test_utils::polyi_deriv(2, 2.0));
        // assert_equal_to_decimal!(
        //     sderivative(&|x: f64| x.powi(2), 2.0, None),
        //     test_utils::polyi_deriv(2, 2.0),
        //     11
        // );
        // assert_equal_to_decimal!(
        //     sderivative(&|x: f64| x.powi(3), 2.0, None),
        //     test_utils::polyi_deriv(3, 2.0),
        //     9
        // );
        // assert_equal_to_decimal!(
        //     sderivative(&|x: f64| x.powi(4), 2.0, None),
        //     test_utils::polyi_deriv(4, 2.0),
        //     8
        // );
        // assert_equal_to_decimal!(
        //     sderivative(&|x: f64| x.powi(7), 2.0, None),
        //     test_utils::polyi_deriv(7, 2.0),
        //     6
        // );
        // assert_equal_to_decimal!(
        //     sderivative(&|x: f64| x.powi(-1), 2.0, None),
        //     test_utils::polyi_deriv(-1, 2.0),
        //     10
        // );
        // assert_equal_to_decimal!(
        //     sderivative(&|x: f64| x.powi(-2), 2.0, None),
        //     test_utils::polyi_deriv(-2, 2.0),
        //     10
        // );
        // assert_equal_to_decimal!(
        //     sderivative(&|x: f64| x.powi(-3), 2.0, None),
        //     test_utils::polyi_deriv(-3, 2.0),
        //     10
        // );
        // assert_equal_to_decimal!(
        //     sderivative(&|x: f64| x.powi(-7), 2.0, None),
        //     test_utils::polyi_deriv(-7, 2.0),
        //     10
        // );
        // assert_equal_to_decimal!(
        //     sderivative(&|x: f64| x.powf(1.0 / 3.0), 2.0, None),
        //     test_utils::polyf_deriv(1.0 / 3.0, 2.0),
        //     16
        // );
        // assert_equal_to_decimal!(
        //     sderivative(&|x: f64| x.powf(7.0 / 3.0), 2.0, None),
        //     test_utils::polyf_deriv(7.0 / 3.0, 2.0),
        //     10
        // );
        // assert_equal_to_decimal!(
        //     sderivative(&|x: f64| x.powf(-1.0 / 3.0), 2.0, None),
        //     test_utils::polyf_deriv(-1.0 / 3.0, 2.0),
        //     11
        // );
        // assert_equal_to_decimal!(
        //     sderivative(&|x: f64| x.powf(-7.0 / 3.0), 2.0, None),
        //     test_utils::polyf_deriv(-7.0 / 3.0, 2.0),
        //     10
        // );
    }
}
