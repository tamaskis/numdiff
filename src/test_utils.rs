#[cfg(test)]
#[cfg(feature = "trig")]
use trig::Trig;

// ---------------------------------
// First-order derivative functions.
// ---------------------------------

#[cfg(test)]
pub(crate) fn polyi_deriv(n: i32, x: f64) -> f64 {
    <f64 as From<i32>>::from(n) * x.powi(n - 1)
}

#[cfg(test)]
pub(crate) fn polyf_deriv(n: f64, x: f64) -> f64 {
    n * x.powf(n - 1.0)
}

#[cfg(test)]
pub(crate) fn sqrt_deriv(x: f64) -> f64 {
    1.0 / (2.0 * x.sqrt())
}

#[cfg(test)]
pub(crate) fn power_deriv(b: f64, x: f64) -> f64 {
    b.powf(x) * b.ln()
}

#[cfg(test)]
pub(crate) fn exp_deriv(x: f64) -> f64 {
    x.exp()
}

#[cfg(test)]
pub(crate) fn ln_deriv(x: f64) -> f64 {
    1.0 / x
}

#[cfg(test)]
pub(crate) fn log10_deriv(x: f64) -> f64 {
    1.0 / (x * 10.0f64.ln())
}

#[cfg(test)]
pub(crate) fn sin_deriv(x: f64) -> f64 {
    x.cos()
}

#[cfg(test)]
pub(crate) fn cos_deriv(x: f64) -> f64 {
    -x.sin()
}

#[cfg(test)]
#[cfg(feature = "trig")]
pub(crate) fn tan_deriv(x: f64) -> f64 {
    x.sec().powi(2)
}

#[cfg(test)]
#[cfg(feature = "trig")]
pub(crate) fn csc_deriv(x: f64) -> f64 {
    -x.csc() * x.tan()
}

#[cfg(test)]
#[cfg(feature = "trig")]
pub(crate) fn sec_deriv(x: f64) -> f64 {
    x.sec() * x.tan()
}

#[cfg(test)]
#[cfg(feature = "trig")]
pub(crate) fn cot_deriv(x: f64) -> f64 {
    -x.csc().powi(2)
}

#[cfg(test)]
pub(crate) fn asin_deriv(x: f64) -> f64 {
    1.0 / (1.0 - x.powi(2)).sqrt()
}

#[cfg(test)]
pub(crate) fn acos_deriv(x: f64) -> f64 {
    -1.0 / (1.0 - x.powi(2)).sqrt()
}

#[cfg(test)]
pub(crate) fn atan_deriv(x: f64) -> f64 {
    1.0 / (1.0 + x.powi(2))
}

#[cfg(all(test, feature = "trig"))]
pub(crate) fn acsc_deriv(x: f64) -> f64 {
    -1.0 / (x.abs() * (x.powi(2) - 1.0).sqrt())
}

#[cfg(all(test, feature = "trig"))]
pub(crate) fn asec_deriv(x: f64) -> f64 {
    1.0 / (x.abs() * (x.powi(2) - 1.0).sqrt())
}

#[cfg(all(test, feature = "trig"))]
pub(crate) fn acot_deriv(x: f64) -> f64 {
    -1.0 / (1.0 + x.powi(2))
}

#[cfg(test)]
pub(crate) fn sinh_deriv(x: f64) -> f64 {
    x.cosh()
}

#[cfg(test)]
pub(crate) fn cosh_deriv(x: f64) -> f64 {
    x.sinh()
}

#[cfg(test)]
#[cfg(feature = "trig")]
pub(crate) fn tanh_deriv(x: f64) -> f64 {
    x.sech().powi(2)
}

#[cfg(test)]
#[cfg(feature = "trig")]
pub(crate) fn csch_deriv(x: f64) -> f64 {
    -x.csch() * x.coth()
}

#[cfg(test)]
#[cfg(feature = "trig")]
pub(crate) fn sech_deriv(x: f64) -> f64 {
    -x.sech() * x.tanh()
}

#[cfg(test)]
#[cfg(feature = "trig")]
pub(crate) fn coth_deriv(x: f64) -> f64 {
    -x.csch().powi(2)
}

#[cfg(test)]
pub(crate) fn asinh_deriv(x: f64) -> f64 {
    1.0 / (1.0 + x.powi(2)).sqrt()
}

#[cfg(test)]
pub(crate) fn acosh_deriv(x: f64) -> f64 {
    1.0 / (x.powi(2) - 1.0).sqrt()
}

#[cfg(test)]
pub(crate) fn atanh_deriv(x: f64) -> f64 {
    1.0 / (1.0 - x.powi(2))
}

#[cfg(all(test, feature = "trig"))]
pub(crate) fn acsch_deriv(x: f64) -> f64 {
    -1.0 / (x.abs() * (x.powi(2) + 1.0).sqrt())
}

#[cfg(all(test, feature = "trig"))]
pub(crate) fn asech_deriv(x: f64) -> f64 {
    -1.0 / (x * (1.0 - x.powi(2)).sqrt())
}

#[cfg(all(test, feature = "trig"))]
pub(crate) fn acoth_deriv(x: f64) -> f64 {
    1.0 / (1.0 - x.powi(2))
}

// ----------------------------------
// Second-order derivative functions.
// ----------------------------------

#[cfg(test)]
pub(crate) fn polyi_deriv2(n: i32, x: f64) -> f64 {
    <f64 as From<i32>>::from(n) * <f64 as From<i32>>::from(n - 1) * x.powi(n - 2)
}

#[cfg(test)]
pub(crate) fn polyf_deriv2(n: f64, x: f64) -> f64 {
    n * (n - 1.0) * x.powf(n - 2.0)
}

#[cfg(test)]
pub(crate) fn sqrt_deriv2(x: f64) -> f64 {
    -1.0 / (4.0 * x.powf(1.5))
}

#[cfg(test)]
pub(crate) fn power_deriv2(b: f64, x: f64) -> f64 {
    b.powf(x) * b.ln().powi(2)
}

#[cfg(test)]
pub(crate) fn exp_deriv2(x: f64) -> f64 {
    x.exp()
}

#[cfg(test)]
pub(crate) fn ln_deriv2(x: f64) -> f64 {
    -1.0 / x.powi(2)
}

#[cfg(test)]
pub(crate) fn log10_deriv2(x: f64) -> f64 {
    -1.0 / (x.powi(2) * 10.0f64.ln())
}

#[cfg(test)]
pub(crate) fn sin_deriv2(x: f64) -> f64 {
    -x.sin()
}

#[cfg(test)]
pub(crate) fn cos_deriv2(x: f64) -> f64 {
    -x.cos()
}

#[cfg(test)]
#[cfg(feature = "trig")]
pub(crate) fn tan_deriv2(x: f64) -> f64 {
    2.0 * x.sec().powi(2) * x.tan()
}

#[cfg(test)]
#[cfg(feature = "trig")]
pub(crate) fn csc_deriv2(x: f64) -> f64 {
    x.csc() * (x.csc().powi(2) + x.cot().powi(2))
}

#[cfg(test)]
#[cfg(feature = "trig")]
pub(crate) fn sec_deriv2(x: f64) -> f64 {
    x.sec() * (x.sec().powi(2) + x.tan().powi(2))
}

#[cfg(test)]
#[cfg(feature = "trig")]
pub(crate) fn cot_deriv2(x: f64) -> f64 {
    2.0 * x.csc().powi(2) * x.cot()
}

#[cfg(test)]
pub(crate) fn asin_deriv2(x: f64) -> f64 {
    x / (1.0 - x.powi(2)).powf(1.5)
}

#[cfg(test)]
pub(crate) fn acos_deriv2(x: f64) -> f64 {
    -x / (1.0 - x.powi(2)).powf(1.5)
}

#[cfg(test)]
pub(crate) fn atan_deriv2(x: f64) -> f64 {
    -2.0 * x / (1.0 + x.powi(2)).powi(2)
}

#[cfg(all(test, feature = "trig"))]
pub(crate) fn acsc_deriv2(x: f64) -> f64 {
    (2.0 * x.powi(2) - 1.0) / (x.powi(2) * x.abs() * (x.powi(2) - 1.0).powf(1.5))
}

#[cfg(all(test, feature = "trig"))]
pub(crate) fn asec_deriv2(x: f64) -> f64 {
    -(2.0 * x.powi(2) - 1.0) / (x.powi(2) * x.abs() * (x.powi(2) - 1.0).powf(1.5))
}

#[cfg(all(test, feature = "trig"))]
pub(crate) fn acot_deriv2(x: f64) -> f64 {
    2.0 * x / (1.0 + x.powi(2)).powi(2)
}

#[cfg(test)]
pub(crate) fn sinh_deriv2(x: f64) -> f64 {
    x.sinh()
}

#[cfg(test)]
pub(crate) fn cosh_deriv2(x: f64) -> f64 {
    x.cosh()
}

#[cfg(test)]
#[cfg(feature = "trig")]
pub(crate) fn tanh_deriv2(x: f64) -> f64 {
    -2.0 * x.sech().powi(2) * x.tanh()
}

#[cfg(test)]
#[cfg(feature = "trig")]
pub(crate) fn csch_deriv2(x: f64) -> f64 {
    x.csch() * (x.csch().powi(2) + x.coth().powi(2))
}

#[cfg(test)]
#[cfg(feature = "trig")]
pub(crate) fn sech_deriv2(x: f64) -> f64 {
    x.sech() * (x.sech().powi(2) - x.tanh().powi(2))
}

#[cfg(test)]
#[cfg(feature = "trig")]
pub(crate) fn coth_deriv2(x: f64) -> f64 {
    2.0 * x.csch().powi(2) * x.coth()
}

#[cfg(test)]
pub(crate) fn asinh_deriv2(x: f64) -> f64 {
    -x / (1.0 + x.powi(2)).powf(1.5)
}

#[cfg(test)]
pub(crate) fn acosh_deriv2(x: f64) -> f64 {
    -x / (x.powi(2) - 1.0).powf(1.5)
}

#[cfg(test)]
pub(crate) fn atanh_deriv2(x: f64) -> f64 {
    2.0 * x / (1.0 - x.powi(2)).powi(2)
}

#[cfg(all(test, feature = "trig"))]
pub(crate) fn acsch_deriv2(x: f64) -> f64 {
    (2.0 * x.powi(2) + 1.0) / (x.powi(2) * x.abs() * (x.powi(2) + 1.0).powf(1.5))
}

#[cfg(all(test, feature = "trig"))]
pub(crate) fn asech_deriv2(x: f64) -> f64 {
    -(2.0 * x.powi(2) - 1.0) / (x.powi(2) * (1.0 - x.powi(2)).powf(1.5))
}

#[cfg(all(test, feature = "trig"))]
pub(crate) fn acoth_deriv2(x: f64) -> f64 {
    2.0 * x / (1.0 - x.powi(2)).powi(2)
}
