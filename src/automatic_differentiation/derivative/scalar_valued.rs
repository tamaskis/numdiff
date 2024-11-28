/// Get a function that returns the derivative of the provided univariate, scalar-valued function.
///
/// The derivative is computed using forward-mode automatic differentiatino.
///
/// # Arguments
///
/// * `f` - Univariate, scalar-valued function, $f:\mathbb{R}\to\mathbb{R}$.
/// * `func_name` - Name of the function that will return the derivative of $f(x)$ at any point
///                 $x\in\mathbb{R}$.
///
/// # Note
///
/// `f` cannot be defined as closure. It must be defined as a function.
///
/// # Example
///
/// Approximate the derivative of
///
/// $$f(x)=x^{3}$$
///
/// at $x=2$, and compare the result to the true result of $f'(2)=12$.
///
/// ```
/// use linalg_traits::Scalar;
/// use numtest::*;
///
/// use numdiff::{get_sderivative, Dual};
///
/// // Define the function, f(x).
/// fn f<S: Scalar>(x: S) -> S {
///     x.powi(3)
/// }
///
/// // Autogenerate the function "df" that can be used to compute the derivative of f(x) at any
/// // point x.
/// get_sderivative!(f, df);
///
/// // Compute the derivative of f(x) at the evaluation point, x = 2.
/// let df_at_2: f64 = df(2.0);
///
/// // Check the accuracy of the derivative.
/// assert_equal_to_decimal!(df_at_2, 12.0, 16);
/// ```
#[macro_export]
macro_rules! get_sderivative {
    ($f:ident, $func_name:ident) => {
        /// Derivative of some univariate, scalar-valued function $f:\mathbb{R}\to\mathbb{R}$.
        ///
        /// # Arguments
        ///
        /// * `x0` - Evaluation point, $x_{0}\in\mathbb{R}$.
        ///
        /// # Returns
        ///
        /// Derivative of $f$ with respect to $x$, evaluated at $x=x_{0}$.
        ///
        /// $$\frac{df}{dx}\bigg\rvert_{x=x_{0}}\in\mathbb{R}$$
        fn $func_name<S: Scalar>(x0: S) -> f64 {
            let x0 = Dual::new(x0.to_f64().unwrap(), 1.0);
            let result = $f(x0);
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

    // TODO: test for all methods implemented on Scalar type
    // TODO: add optional dependency to trig
    // TODO: add product, quotient, and chain rule tests for other differentiation methods
    #[test]
    fn test_product_rule() {
        fn f<S: Scalar>(x: S) -> S {
            x.powi(3)
        }
        fn g<S: Scalar>(x: S) -> S {
            x.sin()
        }
        fn h<S: Scalar>(x: S) -> S {
            f(x) * g(x)
        }
        get_sderivative!(h, dh);
        let dh_true = |x: f64| 3.0 * x.powi(2) * x.sin() + x.cos() * x.powi(3);
        assert_eq!(dh(-1.5), dh_true(-1.5));
        assert_eq!(dh(1.5), dh_true(1.5));
    }

    #[test]
    fn test_quotient_rule() {
        fn f<S: Scalar>(x: S) -> S {
            x.powi(3)
        }
        fn g<S: Scalar>(x: S) -> S {
            x.sin()
        }
        fn h<S: Scalar>(x: S) -> S {
            f(x) / g(x)
        }
        get_sderivative!(h, dh);
        let dh_true = |x: f64| (3.0 * x.powi(2) * x.sin() - x.powi(3) * x.cos()) / x.sin().powi(2);
        assert_eq!(dh(-1.5), dh_true(-1.5));
        assert_eq!(dh(1.5), dh_true(1.5));
    }

    #[test]
    fn test_chain_rule_one_composition() {
        fn f<S: Scalar>(x: S) -> S {
            x.powi(3)
        }
        fn g<S: Scalar>(x: S) -> S {
            x.sin()
        }
        fn h<S: Scalar>(x: S) -> S {
            g(f(x))
        }
        get_sderivative!(h, dh);
        let dh_true = |x: f64| 3.0 * x.powi(2) * (x.powi(3)).cos();
        assert_eq!(dh(-1.5), dh_true(-1.5));
        assert_eq!(dh(1.5), dh_true(1.5));
    }
    #[test]
    fn test_chain_rule_two_compositions() {
        fn f<S: Scalar>(x: S) -> S {
            x.powi(3)
        }
        fn g<S: Scalar>(x: S) -> S {
            x.sin()
        }
        fn h<S: Scalar>(x: S) -> S {
            S::new(5.0) / x.powi(2)
        }
        fn j<S: Scalar>(x: S) -> S {
            h(g(f(x)))
        }
        get_sderivative!(j, dj);
        let dj_true = |x: f64| (-30.0 * x.powi(2) * (x.powi(3)).cos()) / (x.powi(3)).sin().powi(3);
        assert_equal_to_decimal!(dj(1.5), dj_true(1.5), 12);
    }

    #[test]
    fn test_sderivative_polynomial() {
        // Test #1.
        fn f1<S: Scalar>(_x: S) -> S {
            S::one()
        }
        get_sderivative!(f1, df1);
        assert_eq!(df1(2.0), test_utils::polyi_deriv(0, 2.0));

        // Test #2.
        fn f2<S: Scalar>(x: S) -> S {
            x
        }
        get_sderivative!(f2, df2);
        assert_eq!(df2(2.0), test_utils::polyi_deriv(1, 2.0));

        // Test #3.
        fn f3<S: Scalar>(x: S) -> S {
            x.powi(2)
        }
        get_sderivative!(f3, df3);
        assert_eq!(df3(2.0), test_utils::polyi_deriv(2, 2.0));

        // Test #4.
        fn f4<S: Scalar>(x: S) -> S {
            x.powi(3)
        }
        get_sderivative!(f4, df4);
        assert_eq!(df4(2.0), test_utils::polyi_deriv(3, 2.0));

        // Test #5.
        fn f5<S: Scalar>(x: S) -> S {
            x.powi(4)
        }
        get_sderivative!(f5, df5);
        assert_eq!(df5(2.0), test_utils::polyi_deriv(4, 2.0));

        // Test #6.
        fn f6<S: Scalar>(x: S) -> S {
            x.powi(5)
        }
        get_sderivative!(f6, df6);
        assert_eq!(df6(2.0), test_utils::polyi_deriv(5, 2.0));

        // Test #7.
        fn f7<S: Scalar>(x: S) -> S {
            x.powi(6)
        }
        get_sderivative!(f7, df7);
        assert_eq!(df7(2.0), test_utils::polyi_deriv(6, 2.0));

        // Test #8.
        fn f8<S: Scalar>(x: S) -> S {
            x.powi(7)
        }
        get_sderivative!(f8, df8);
        assert_eq!(df8(2.0), test_utils::polyi_deriv(7, 2.0));

        // Test #9.
        fn f9<S: Scalar>(x: S) -> S {
            x.powi(-1)
        }
        get_sderivative!(f9, df9);
        assert_eq!(df9(2.0), test_utils::polyi_deriv(-1, 2.0));

        // Test #10.
        fn f10<S: Scalar>(x: S) -> S {
            x.powi(-2)
        }
        get_sderivative!(f10, df10);
        assert_eq!(df10(2.0), test_utils::polyi_deriv(-2, 2.0));

        // Test #11.
        fn f11<S: Scalar>(x: S) -> S {
            x.powi(-3)
        }
        get_sderivative!(f11, df11);
        assert_eq!(df11(2.0), test_utils::polyi_deriv(-3, 2.0));

        // Test #12.
        fn f12<S: Scalar>(x: S) -> S {
            x.powi(-7)
        }
        get_sderivative!(f12, df12);
        assert_eq!(df12(2.0), test_utils::polyi_deriv(-7, 2.0));

        // Test #13.
        fn f13<S: Scalar>(x: S) -> S {
            x.powf(S::new(1.0 / 3.0))
        }
        get_sderivative!(f13, df13);
        assert_eq!(df13(2.0), test_utils::polyf_deriv(1.0 / 3.0, 2.0));

        // Test #14.
        fn f14<S: Scalar>(x: S) -> S {
            x.powf(S::new(7.0 / 3.0))
        }
        get_sderivative!(f14, df14);
        assert_equal_to_decimal!(df14(2.0), test_utils::polyf_deriv(7.0 / 3.0, 2.0), 15);

        // Test #15.
        fn f15<S: Scalar>(x: S) -> S {
            x.powf(S::new(-1.0 / 3.0))
        }
        get_sderivative!(f15, df15);
        assert_eq!(df15(2.0), test_utils::polyf_deriv(-1.0 / 3.0, 2.0));

        // Test #16.
        fn f16<S: Scalar>(x: S) -> S {
            x.powf(S::new(-7.0 / 3.0))
        }
        get_sderivative!(f16, df16);
        assert_eq!(df16(2.0), test_utils::polyf_deriv(-7.0 / 3.0, 2.0));
    }

    #[test]
    fn test_sderivative_square_root() {
        fn f<S: Scalar>(x: S) -> S {
            x.sqrt()
        }
        get_sderivative!(f, df);
        assert_eq!(df(0.5), test_utils::sqrt_deriv(0.5));
        assert_eq!(df(1.5), test_utils::sqrt_deriv(1.5));
    }

    #[test]
    fn test_sderivative_exponential() {
        fn f<S: Scalar>(x: S) -> S {
            x.exp()
        }
        get_sderivative!(f, df);
        assert_eq!(df(-1.0), test_utils::exp_deriv(-1.0));
        assert_eq!(df(0.0), test_utils::exp_deriv(0.0));
        assert_eq!(df(1.0), test_utils::exp_deriv(1.0));
    }

    #[test]
    fn test_sderivative_power() {
        fn f<S: Scalar>(x: S) -> S {
            S::new(5.0).powf(x)
        }
        get_sderivative!(f, df);
        assert_eq!(df(-1.0), test_utils::power_deriv(5.0, -1.0));
        assert_eq!(df(0.0), test_utils::power_deriv(5.0, 0.0));
        assert_equal_to_decimal!(df(1.0), test_utils::power_deriv(5.0, 1.0), 14);
    }

    #[test]
    fn test_sderivative_natural_logarithm() {
        fn f<S: Scalar>(x: S) -> S {
            x.ln()
        }
        get_sderivative!(f, df);
        assert_eq!(df(0.5), test_utils::ln_deriv(0.5));
        assert_eq!(df(1.0), test_utils::ln_deriv(1.0));
        assert_eq!(df(1.5), test_utils::ln_deriv(1.5));
    }

    #[test]
    fn test_sderivative_base_10_logarithm() {
        fn f<S: Scalar>(x: S) -> S {
            x.log10()
        }
        get_sderivative!(f, df);
        assert_eq!(df(0.5), test_utils::log10_deriv(0.5));
        assert_eq!(df(1.0), test_utils::log10_deriv(1.0));
        assert_eq!(df(1.5), test_utils::log10_deriv(1.5));
    }
}
