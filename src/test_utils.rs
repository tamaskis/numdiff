use crate::automatic_differentiation::hyper_dual::hyper_dual::HyperDual;
use linalg_traits::RealField;
use numtest::*;

// ---------------------------------
// First-order derivative functions.
// ---------------------------------

pub(crate) fn polyi_deriv(n: i32, x: f64) -> f64 {
    <f64 as From<i32>>::from(n) * x.powi(n - 1)
}

pub(crate) fn polyf_deriv(n: f64, x: f64) -> f64 {
    n * x.powf(n - 1.0)
}

pub(crate) fn sqrt_deriv(x: f64) -> f64 {
    1.0 / (2.0 * x.sqrt())
}

pub(crate) fn power_deriv(b: f64, x: f64) -> f64 {
    b.powf(x) * b.ln()
}

pub(crate) fn exp_deriv(x: f64) -> f64 {
    x.exp()
}

pub(crate) fn ln_deriv(x: f64) -> f64 {
    1.0 / x
}

pub(crate) fn log10_deriv(x: f64) -> f64 {
    1.0 / (x * 10.0f64.ln())
}

pub(crate) fn sin_deriv(x: f64) -> f64 {
    x.cos()
}

pub(crate) fn cos_deriv(x: f64) -> f64 {
    -x.sin()
}

pub(crate) fn tan_deriv(x: f64) -> f64 {
    x.sec().powi(2)
}

pub(crate) fn csc_deriv(x: f64) -> f64 {
    -x.csc() * x.tan()
}

pub(crate) fn sec_deriv(x: f64) -> f64 {
    x.sec() * x.tan()
}

pub(crate) fn cot_deriv(x: f64) -> f64 {
    -x.csc().powi(2)
}

pub(crate) fn asin_deriv(x: f64) -> f64 {
    1.0 / (1.0 - x.powi(2)).sqrt()
}

pub(crate) fn acos_deriv(x: f64) -> f64 {
    -1.0 / (1.0 - x.powi(2)).sqrt()
}

pub(crate) fn atan_deriv(x: f64) -> f64 {
    1.0 / (1.0 + x.powi(2))
}

pub(crate) fn acsc_deriv(x: f64) -> f64 {
    -1.0 / (x.abs() * (x.powi(2) - 1.0).sqrt())
}

pub(crate) fn asec_deriv(x: f64) -> f64 {
    1.0 / (x.abs() * (x.powi(2) - 1.0).sqrt())
}

pub(crate) fn acot_deriv(x: f64) -> f64 {
    -1.0 / (1.0 + x.powi(2))
}

pub(crate) fn sinh_deriv(x: f64) -> f64 {
    x.cosh()
}

pub(crate) fn cosh_deriv(x: f64) -> f64 {
    x.sinh()
}

pub(crate) fn tanh_deriv(x: f64) -> f64 {
    x.sech().powi(2)
}

pub(crate) fn csch_deriv(x: f64) -> f64 {
    -x.csch() * x.coth()
}

pub(crate) fn sech_deriv(x: f64) -> f64 {
    -x.sech() * x.tanh()
}

pub(crate) fn coth_deriv(x: f64) -> f64 {
    -x.csch().powi(2)
}

pub(crate) fn asinh_deriv(x: f64) -> f64 {
    1.0 / (1.0 + x.powi(2)).sqrt()
}

pub(crate) fn acosh_deriv(x: f64) -> f64 {
    1.0 / (x.powi(2) - 1.0).sqrt()
}

pub(crate) fn atanh_deriv(x: f64) -> f64 {
    1.0 / (1.0 - x.powi(2))
}

pub(crate) fn acsch_deriv(x: f64) -> f64 {
    -1.0 / (x.abs() * (x.powi(2) + 1.0).sqrt())
}

pub(crate) fn asech_deriv(x: f64) -> f64 {
    -1.0 / (x * (1.0 - x.powi(2)).sqrt())
}

pub(crate) fn acoth_deriv(x: f64) -> f64 {
    1.0 / (1.0 - x.powi(2))
}

// ----------------------------------
// Second-order derivative functions.
// ----------------------------------

pub(crate) fn polyi_deriv2(n: i32, x: f64) -> f64 {
    <f64 as From<i32>>::from(n) * <f64 as From<i32>>::from(n - 1) * x.powi(n - 2)
}

pub(crate) fn polyf_deriv2(n: f64, x: f64) -> f64 {
    n * (n - 1.0) * x.powf(n - 2.0)
}

pub(crate) fn sqrt_deriv2(x: f64) -> f64 {
    -1.0 / (4.0 * x.powf(1.5))
}

pub(crate) fn power_deriv2(b: f64, x: f64) -> f64 {
    b.powf(x) * b.ln().powi(2)
}

pub(crate) fn exp_deriv2(x: f64) -> f64 {
    x.exp()
}

pub(crate) fn ln_deriv2(x: f64) -> f64 {
    -1.0 / x.powi(2)
}

pub(crate) fn log10_deriv2(x: f64) -> f64 {
    -1.0 / (x.powi(2) * 10.0f64.ln())
}

pub(crate) fn sin_deriv2(x: f64) -> f64 {
    -x.sin()
}

pub(crate) fn cos_deriv2(x: f64) -> f64 {
    -x.cos()
}

pub(crate) fn tan_deriv2(x: f64) -> f64 {
    2.0 * x.sec().powi(2) * x.tan()
}

pub(crate) fn csc_deriv2(x: f64) -> f64 {
    x.csc() * (x.csc().powi(2) + x.cot().powi(2))
}

pub(crate) fn sec_deriv2(x: f64) -> f64 {
    x.sec() * (x.sec().powi(2) + x.tan().powi(2))
}

pub(crate) fn cot_deriv2(x: f64) -> f64 {
    2.0 * x.csc().powi(2) * x.cot()
}

pub(crate) fn asin_deriv2(x: f64) -> f64 {
    x / (1.0 - x.powi(2)).powf(1.5)
}

pub(crate) fn acos_deriv2(x: f64) -> f64 {
    -x / (1.0 - x.powi(2)).powf(1.5)
}

pub(crate) fn atan_deriv2(x: f64) -> f64 {
    -2.0 * x / (1.0 + x.powi(2)).powi(2)
}

pub(crate) fn acsc_deriv2(x: f64) -> f64 {
    (2.0 * x.powi(2) - 1.0) / (x * x.abs() * (x.powi(2) - 1.0).powf(1.5))
}

pub(crate) fn asec_deriv2(x: f64) -> f64 {
    -(2.0 * x.powi(2) - 1.0) / (x * x.abs() * (x.powi(2) - 1.0).powf(1.5))
}

pub(crate) fn acot_deriv2(x: f64) -> f64 {
    2.0 * x / (1.0 + x.powi(2)).powi(2)
}

pub(crate) fn sinh_deriv2(x: f64) -> f64 {
    x.sinh()
}

pub(crate) fn cosh_deriv2(x: f64) -> f64 {
    x.cosh()
}

pub(crate) fn tanh_deriv2(x: f64) -> f64 {
    -2.0 * x.sech().powi(2) * x.tanh()
}

pub(crate) fn csch_deriv2(x: f64) -> f64 {
    x.csch() * (x.csch().powi(2) + x.coth().powi(2))
}

pub(crate) fn sech_deriv2(x: f64) -> f64 {
    x.sech() * (x.tanh().powi(2) - x.sech().powi(2))
}

pub(crate) fn coth_deriv2(x: f64) -> f64 {
    2.0 * x.csch().powi(2) * x.coth()
}

pub(crate) fn asinh_deriv2(x: f64) -> f64 {
    -x / (1.0 + x.powi(2)).powf(1.5)
}

pub(crate) fn acosh_deriv2(x: f64) -> f64 {
    -x / (x.powi(2) - 1.0).powf(1.5)
}

pub(crate) fn atanh_deriv2(x: f64) -> f64 {
    2.0 * x / (1.0 - x.powi(2)).powi(2)
}

pub(crate) fn acsch_deriv2(x: f64) -> f64 {
    (2.0 * x.powi(2) + 1.0) / (x * x.abs() * (x.powi(2) + 1.0).powf(1.5))
}

pub(crate) fn asech_deriv2(x: f64) -> f64 {
    -(2.0 * x.powi(2) - 1.0) / (x.powi(2) * (1.0 - x.powi(2)).powf(1.5))
}

pub(crate) fn acoth_deriv2(x: f64) -> f64 {
    2.0 * x / (1.0 - x.powi(2)).powi(2)
}

pub(crate) fn assert_hyper_dual_close(left: HyperDual, right: HyperDual, decimal: i32) {
    assert_arrays_equal_to_decimal!(
        [left.get_a(), left.get_b(), left.get_c(), left.get_d()],
        [right.get_a(), right.get_b(), right.get_c(), right.get_d()],
        decimal
    );
}
