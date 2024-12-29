/// Get a function that returns the derivative of the provided univariate, scalar-valued function.
///
/// The derivative is computed using forward-mode automatic differentiation.
///
/// # Arguments
///
/// * `f` - Univariate, scalar-valued function, $f:\mathbb{R}\to\mathbb{R}$.
/// * `func_name` - Name of the function that will return the derivative of $f(x)$ at any point
///                 $x\in\mathbb{R}$.
///
/// # Warning
///
/// `f` cannot be defined as closure. It must be defined as a function.
///
/// # Note
///
/// The function produced by this macro will perform 1 evaluation of $f(x)$ to evaluate its
/// derivative.
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
        /// Derivative of a univariate, scalar-valued function `f: ℝ → ℝ`.
        ///
        /// This function is generated for a specific function `f` using the
        /// `numdiff::get_sderivative!` macro.
        ///
        /// # Arguments
        ///
        /// `x0` - Evaluation point, `x₀ ∈ ℝ`.
        ///
        /// # Returns
        ///
        /// Derivative of `f` with respect to `x`, evaluated at `x = x₀`.
        ///
        /// `(df/dx)|ₓ₌ₓ₀ ∈ ℝ`
        fn $func_name<S: Scalar>(x0: S) -> f64 {
            // Step forward in the dual direction.
            let x0 = Dual::new(x0.to_f64().unwrap(), 1.0);

            // Evaluate the function at the dual number.
            let f_x0 = $f(x0);

            // Derivative of f with respect to x evaluated at x = x₀.
            f_x0.get_dual()
        }
    };
}

#[cfg(test)]
mod tests {
    use crate::automatic_differentiation::dual::Dual;
    use crate::test_utils;
    use linalg_traits::Scalar;
    use numtest::*;
    use std::f64::consts::PI;

    #[cfg(feature = "trig")]
    use trig::Trig;

    #[test]
    fn test_product_rule() {
        // f(x) and f'(x).
        fn f<S: Scalar>(x: S) -> S {
            x.powi(3)
        }
        fn df<S: Scalar>(x: S) -> S {
            S::new(3.0) * x.powi(2)
        }

        // g(x) and g'(x)
        fn g<S: Scalar>(x: S) -> S {
            x.sin()
        }
        fn dg<S: Scalar>(x: S) -> S {
            x.cos()
        }

        // h(x) = f(x)g(x) and h'(x) = f'(x)g(x) + f(x)g'(x).
        fn h<S: Scalar>(x: S) -> S {
            f(x) * g(x)
        }
        fn dh<S: Scalar>(x: S) -> S {
            df(x) * g(x) + f(x) * dg(x)
        }

        // h'(x) obtained via forward-mode automatic differentiation.
        get_sderivative!(h, dh_autodiff);

        // Test autodiff h'(x) against true h'(x).
        assert_eq!(dh_autodiff(-1.5), dh(-1.5));
        assert_eq!(dh_autodiff(1.5), dh(1.5));
    }

    #[test]
    fn test_quotient_rule() {
        // f(x) and f'(x).
        fn f<S: Scalar>(x: S) -> S {
            x.powi(3)
        }
        fn df<S: Scalar>(x: S) -> S {
            S::new(3.0) * x.powi(2)
        }

        // g(x) and g'(x)
        fn g<S: Scalar>(x: S) -> S {
            x.sin()
        }
        fn dg<S: Scalar>(x: S) -> S {
            x.cos()
        }

        // h(x) = f(x) / g(x) and h'(x) = (g(x)f'(x) - f(x)g'(x)) / (g(x))².
        fn h<S: Scalar>(x: S) -> S {
            f(x) / g(x)
        }
        fn dh<S: Scalar>(x: S) -> S {
            (g(x) * df(x) - f(x) * dg(x)) / g(x).powi(2)
        }

        // h'(x) obtained via forward-mode automatic differentiation.
        get_sderivative!(h, dh_autodiff);

        // Test autodiff h'(x) against true h'(x).
        assert_eq!(dh_autodiff(-1.5), dh(-1.5));
        assert_eq!(dh_autodiff(1.5), dh(1.5));
    }

    #[test]
    fn test_chain_rule_one_composition() {
        // f(x) and f'(x).
        fn f<S: Scalar>(x: S) -> S {
            x.powi(3)
        }
        fn df<S: Scalar>(x: S) -> S {
            S::new(3.0) * x.powi(2)
        }

        // g(x) and g'(x)
        fn g<S: Scalar>(x: S) -> S {
            x.sin()
        }
        fn dg<S: Scalar>(x: S) -> S {
            x.cos()
        }

        // h(x) = g(f(x)) and h'(x) = [g'(f(x))][f'(x)].
        fn h<S: Scalar>(x: S) -> S {
            g(f(x))
        }
        fn dh<S: Scalar>(x: S) -> S {
            dg(f(x)) * df(x)
        }

        // h'(x) obtained via forward-mode automatic differentiation.
        get_sderivative!(h, dh_autodiff);

        // Test autodiff h'(x) against true h'(x).
        assert_eq!(dh_autodiff(-1.5), dh(-1.5));
        assert_eq!(dh_autodiff(1.5), dh(1.5));
    }

    #[test]
    fn test_chain_rule_two_compositions() {
        // f(x) and f'(x).
        fn f<S: Scalar>(x: S) -> S {
            x.powi(3)
        }
        fn df<S: Scalar>(x: S) -> S {
            S::new(3.0) * x.powi(2)
        }

        // g(x) and g'(x)
        fn g<S: Scalar>(x: S) -> S {
            x.sin()
        }
        fn dg<S: Scalar>(x: S) -> S {
            x.cos()
        }

        // h(x) and h'(x).
        fn h<S: Scalar>(x: S) -> S {
            S::new(5.0) / x.powi(2)
        }
        fn dh<S: Scalar>(x: S) -> S {
            S::new(-10.0) / x.powi(3)
        }

        // j(x) = h(g(f(x))) and j'(x) = [h'(g(f(x)))][g'(f(x))][f'(x)].
        fn j<S: Scalar>(x: S) -> S {
            h(g(f(x)))
        }
        fn dj<S: Scalar>(x: S) -> S {
            dh(g(f(x))) * dg(f(x)) * df(x)
        }

        // j'(x) obtained via forward-mode automatic differentiation.
        get_sderivative!(j, dj_autodiff);

        // Test autodiff j'(x) against true j'(x).
        assert_equal_to_decimal!(dj_autodiff(1.5), dj(1.5), 12);
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

    #[test]
    fn test_sderivative_sine() {
        fn f<S: Scalar>(x: S) -> S {
            x.sin()
        }
        get_sderivative!(f, df);
        assert_eq!(df(0.0), test_utils::sin_deriv(0.0));
        assert_eq!(df(PI / 4.0), test_utils::sin_deriv(PI / 4.0));
        assert_eq!(df(PI / 2.0), test_utils::sin_deriv(PI / 2.0));
        assert_eq!(df(3.0 * PI / 4.0), test_utils::sin_deriv(3.0 * PI / 4.0));
        assert_eq!(df(PI), test_utils::sin_deriv(PI));
        assert_eq!(df(5.0 * PI / 4.0), test_utils::sin_deriv(5.0 * PI / 4.0));
        assert_eq!(df(3.0 * PI / 2.0), test_utils::sin_deriv(3.0 * PI / 2.0));
        assert_eq!(df(7.0 * PI / 4.0), test_utils::sin_deriv(7.0 * PI / 4.0));
        assert_eq!(df(2.0 * PI), test_utils::sin_deriv(2.0 * PI));
    }

    #[test]
    fn test_sderivative_cosine() {
        fn f<S: Scalar>(x: S) -> S {
            x.cos()
        }
        get_sderivative!(f, df);
        assert_eq!(df(0.0), test_utils::cos_deriv(0.0));
        assert_eq!(df(PI / 4.0), test_utils::cos_deriv(PI / 4.0));
        assert_eq!(df(PI / 2.0), test_utils::cos_deriv(PI / 2.0));
        assert_eq!(df(3.0 * PI / 4.0), test_utils::cos_deriv(3.0 * PI / 4.0));
        assert_eq!(df(PI), test_utils::cos_deriv(PI));
        assert_eq!(df(5.0 * PI / 4.0), test_utils::cos_deriv(5.0 * PI / 4.0));
        assert_eq!(df(3.0 * PI / 2.0), test_utils::cos_deriv(3.0 * PI / 2.0));
        assert_eq!(df(7.0 * PI / 4.0), test_utils::cos_deriv(7.0 * PI / 4.0));
        assert_eq!(df(2.0 * PI), test_utils::cos_deriv(2.0 * PI));
    }

    #[test]
    #[cfg(feature = "trig")]
    fn test_sderivative_tangent() {
        fn f<S: Scalar>(x: S) -> S {
            x.tan()
        }
        get_sderivative!(f, df);
        assert_eq!(df(0.0), test_utils::tan_deriv(0.0));
        assert_eq!(df(PI / 4.0), test_utils::tan_deriv(PI / 4.0));
        assert_eq!(df(3.0 * PI / 4.0), test_utils::tan_deriv(3.0 * PI / 4.0));
        assert_eq!(df(PI), test_utils::tan_deriv(PI));
        assert_eq!(df(5.0 * PI / 4.0), test_utils::tan_deriv(5.0 * PI / 4.0));
        assert_eq!(df(7.0 * PI / 4.0), test_utils::tan_deriv(7.0 * PI / 4.0));
        assert_eq!(df(2.0 * PI), test_utils::tan_deriv(2.0 * PI));
    }

    #[test]
    #[cfg(feature = "trig")]
    fn test_sderivative_cosecant() {
        fn f<S: Scalar + Trig>(x: S) -> S {
            x.csc()
        }
        get_sderivative!(f, df);
        assert_equal_to_decimal!(df(PI / 4.0), test_utils::csc_deriv(PI / 4.0), 15);
        assert_equal_to_decimal!(
            df(3.0 * PI / 4.0),
            test_utils::csc_deriv(3.0 * PI / 4.0),
            15
        );
        assert_equal_to_decimal!(
            df(5.0 * PI / 4.0),
            test_utils::csc_deriv(5.0 * PI / 4.0),
            15
        );
        assert_equal_to_decimal!(
            df(7.0 * PI / 4.0),
            test_utils::csc_deriv(7.0 * PI / 4.0),
            15
        );
    }

    #[test]
    #[cfg(feature = "trig")]
    fn test_sderivative_secant() {
        fn f<S: Scalar + Trig>(x: S) -> S {
            x.sec()
        }
        get_sderivative!(f, df);
        assert_eq!(df(0.0), test_utils::sec_deriv(0.0));
        assert_eq!(df(PI / 4.0), test_utils::sec_deriv(PI / 4.0));
        assert_eq!(df(3.0 * PI / 4.0), test_utils::sec_deriv(3.0 * PI / 4.0));
        assert_eq!(df(PI), test_utils::sec_deriv(PI));
        assert_eq!(df(5.0 * PI / 4.0), test_utils::sec_deriv(5.0 * PI / 4.0));
        assert_eq!(df(7.0 * PI / 4.0), test_utils::sec_deriv(7.0 * PI / 4.0));
        assert_eq!(df(2.0 * PI), test_utils::sec_deriv(2.0 * PI));
    }

    #[test]
    #[cfg(feature = "trig")]
    fn test_sderivative_cotangent() {
        fn f<S: Scalar + Trig>(x: S) -> S {
            x.cot()
        }
        get_sderivative!(f, df);
        assert_equal_to_decimal!(df(PI / 4.0), test_utils::cot_deriv(PI / 4.0), 15);
        assert_equal_to_decimal!(df(PI / 2.0), test_utils::cot_deriv(PI / 2.0), 16);
        assert_eq!(df(3.0 * PI / 4.0), test_utils::cot_deriv(3.0 * PI / 4.0));
        assert_eq!(df(5.0 * PI / 4.0), test_utils::cot_deriv(5.0 * PI / 4.0));
        assert_equal_to_decimal!(
            df(3.0 * PI / 2.0),
            test_utils::cot_deriv(3.0 * PI / 2.0),
            15
        );
        assert_eq!(df(7.0 * PI / 4.0), test_utils::cot_deriv(7.0 * PI / 4.0));
    }

    #[test]
    fn test_sderivative_inverse_sine() {
        fn f<S: Scalar>(x: S) -> S {
            x.asin()
        }
        get_sderivative!(f, df);
        assert_eq!(df(-0.5), test_utils::asin_deriv(-0.5));
        assert_eq!(df(0.0), test_utils::asin_deriv(0.0));
        assert_eq!(df(0.5), test_utils::asin_deriv(0.5));
    }

    #[test]
    fn test_sderivative_inverse_cosine() {
        fn f<S: Scalar>(x: S) -> S {
            x.acos()
        }
        get_sderivative!(f, df);
        assert_eq!(df(-0.5), test_utils::acos_deriv(-0.5));
        assert_eq!(df(0.0), test_utils::acos_deriv(0.0));
        assert_eq!(df(0.5), test_utils::acos_deriv(0.5));
    }

    #[test]
    fn test_sderivative_inverse_tangent() {
        fn f<S: Scalar>(x: S) -> S {
            x.atan()
        }
        get_sderivative!(f, df);
        assert_eq!(df(-1.5), test_utils::atan_deriv(-1.5));
        assert_eq!(df(-1.0), test_utils::atan_deriv(-1.0));
        assert_eq!(df(-0.5), test_utils::atan_deriv(-0.5));
        assert_eq!(df(0.0), test_utils::atan_deriv(0.0));
        assert_eq!(df(0.5), test_utils::atan_deriv(0.5));
        assert_eq!(df(1.0), test_utils::atan_deriv(1.0));
        assert_eq!(df(1.5), test_utils::atan_deriv(1.5));
    }

    #[test]
    #[cfg(feature = "trig")]
    fn test_sderivative_inverse_cosecant() {
        fn f<S: Scalar + Trig>(x: S) -> S {
            x.acsc()
        }
        get_sderivative!(f, df);
        assert_eq!(df(-1.5), test_utils::acsc_deriv(-1.5));
        assert_eq!(df(1.5), test_utils::acsc_deriv(1.5));
    }

    #[test]
    #[cfg(feature = "trig")]
    fn test_sderivative_inverse_secant() {
        fn f<S: Scalar + Trig>(x: S) -> S {
            x.asec()
        }
        get_sderivative!(f, df);
        assert_eq!(df(-1.5), test_utils::asec_deriv(-1.5));
        assert_eq!(df(1.5), test_utils::asec_deriv(1.5));
    }

    #[test]
    #[cfg(feature = "trig")]
    fn test_sderivative_inverse_cotangent() {
        fn f<S: Scalar + Trig>(x: S) -> S {
            x.acot()
        }
        get_sderivative!(f, df);
        assert_equal_to_decimal!(df(-1.5), test_utils::acot_deriv(-1.5), 16);
        assert_eq!(df(-1.0), test_utils::acot_deriv(-1.0));
        assert_eq!(df(-0.5), test_utils::acot_deriv(-0.5));
        assert_eq!(df(0.5), test_utils::acot_deriv(0.5));
        assert_eq!(df(1.0), test_utils::acot_deriv(1.0));
        assert_equal_to_decimal!(df(1.5), test_utils::acot_deriv(1.5), 16);
    }

    #[test]
    fn test_sderivative_hyperbolic_sine() {
        fn f<S: Scalar>(x: S) -> S {
            x.sinh()
        }
        get_sderivative!(f, df);
        assert_eq!(df(-1.0), test_utils::sinh_deriv(-1.0));
        assert_eq!(df(0.0), test_utils::sinh_deriv(0.0));
        assert_eq!(df(1.0), test_utils::sinh_deriv(1.0));
    }

    #[test]
    fn test_sderivative_hyperbolic_cosine() {
        fn f<S: Scalar>(x: S) -> S {
            x.cosh()
        }
        get_sderivative!(f, df);
        assert_eq!(df(-1.0), test_utils::cosh_deriv(-1.0));
        assert_eq!(df(0.0), test_utils::cosh_deriv(0.0));
        assert_eq!(df(1.0), test_utils::cosh_deriv(1.0));
    }

    #[test]
    #[cfg(feature = "trig")]
    fn test_sderivative_hyperbolic_tangent() {
        fn f<S: Scalar>(x: S) -> S {
            x.tanh()
        }
        get_sderivative!(f, df);
        assert_eq!(df(-1.0), test_utils::tanh_deriv(-1.0));
        assert_eq!(df(0.0), test_utils::tanh_deriv(0.0));
        assert_eq!(df(1.0), test_utils::tanh_deriv(1.0));
    }

    #[test]
    #[cfg(feature = "trig")]
    fn test_sderivative_hyperbolic_cosecant() {
        fn f<S: Scalar + Trig>(x: S) -> S {
            x.csch()
        }
        get_sderivative!(f, df);
        assert_eq!(df(-1.0), test_utils::csch_deriv(-1.0));
        assert_eq!(df(1.0), test_utils::csch_deriv(1.0));
    }

    #[test]
    #[cfg(feature = "trig")]
    fn test_sderivative_hyperbolic_secant() {
        fn f<S: Scalar + Trig>(x: S) -> S {
            x.sech()
        }
        get_sderivative!(f, df);
        assert_equal_to_decimal!(df(-1.0), test_utils::sech_deriv(-1.0), 16);
        assert_eq!(df(0.0), test_utils::sech_deriv(0.0));
        assert_equal_to_decimal!(df(1.0), test_utils::sech_deriv(1.0), 16);
    }

    #[test]
    #[cfg(feature = "trig")]
    fn test_sderivative_hyperbolic_cotangent() {
        fn f<S: Scalar + Trig>(x: S) -> S {
            x.coth()
        }
        get_sderivative!(f, df);
        assert_equal_to_decimal!(df(-1.0), test_utils::coth_deriv(-1.0), 16);
        assert_equal_to_decimal!(df(1.0), test_utils::coth_deriv(1.0), 16);
    }

    #[test]
    fn test_sderivative_inverse_hyperbolic_sine() {
        fn f<S: Scalar>(x: S) -> S {
            x.asinh()
        }
        get_sderivative!(f, df);
        assert_eq!(df(-1.5), test_utils::asinh_deriv(-1.5));
        assert_eq!(df(-1.0), test_utils::asinh_deriv(-1.0));
        assert_eq!(df(-0.5), test_utils::asinh_deriv(-0.5));
        assert_eq!(df(0.0), test_utils::asinh_deriv(0.0));
        assert_eq!(df(0.5), test_utils::asinh_deriv(0.5));
        assert_eq!(df(1.0), test_utils::asinh_deriv(1.0));
        assert_eq!(df(1.5), test_utils::asinh_deriv(1.5));
    }

    #[test]
    fn test_sderivative_inverse_hyperbolic_cosine() {
        fn f<S: Scalar>(x: S) -> S {
            x.acosh()
        }
        get_sderivative!(f, df);
        assert_eq!(df(1.5), test_utils::acosh_deriv(1.5));
    }

    #[test]
    fn test_sderivative_inverse_hyperbolic_tangent() {
        fn f<S: Scalar>(x: S) -> S {
            x.atanh()
        }
        get_sderivative!(f, df);
        assert_eq!(df(-0.5), test_utils::atanh_deriv(-0.5));
        assert_eq!(df(0.0), test_utils::atanh_deriv(0.0));
        assert_eq!(df(0.5), test_utils::atanh_deriv(0.5));
    }

    #[test]
    #[cfg(feature = "trig")]
    fn test_sderivative_inverse_hyperbolic_cosecant() {
        fn f<S: Scalar + Trig>(x: S) -> S {
            x.acsch()
        }
        get_sderivative!(f, df);
        assert_equal_to_decimal!(df(-1.5), test_utils::acsch_deriv(-1.5), 16);
        assert_eq!(df(-1.0), test_utils::acsch_deriv(-1.0));
        assert_eq!(df(-0.5), test_utils::acsch_deriv(-0.5));
        assert_eq!(df(0.5), test_utils::acsch_deriv(0.5));
        assert_eq!(df(1.0), test_utils::acsch_deriv(1.0));
        assert_equal_to_decimal!(df(1.5), test_utils::acsch_deriv(1.5), 16);
    }

    #[test]
    #[cfg(feature = "trig")]
    fn test_sderivative_inverse_hyperbolic_secant() {
        fn f<S: Scalar + Trig>(x: S) -> S {
            x.asech()
        }
        get_sderivative!(f, df);
        assert_eq!(df(0.5), test_utils::asech_deriv(0.5));
    }

    #[test]
    #[cfg(feature = "trig")]
    fn test_sderivative_inverse_hyperbolic_cotangent() {
        fn f<S: Scalar + Trig>(x: S) -> S {
            x.acoth()
        }
        get_sderivative!(f, df);
        assert_equal_to_decimal!(df(-1.5), test_utils::acoth_deriv(-1.5), 16);
        assert_equal_to_decimal!(df(1.5), test_utils::acoth_deriv(1.5), 16);
    }
}
