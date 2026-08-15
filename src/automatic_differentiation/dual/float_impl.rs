use crate::automatic_differentiation::dual::dual::Dual;
use num_traits::{Float, One, Zero};
use std::cmp::Ordering;
use std::f64::consts::{LN_2, LN_10};
use std::num::FpCategory;
use std::ops::Neg;

// -------------------------------
// Implementing num_traits::Float.
// -------------------------------
// https://docs.rs/num-traits/latest/num_traits/float/trait.Float.html
//
// pub trait Float: Num + Copy + NumCast + PartialOrd + Neg<Output = Self> {
// [+] Show 60 methods
// }

// Only perform comparisons on the real part.
//  --> This is primarily to support numerical methods where we want to check convergence on the
//      actual function evaluation, and NOT its derivative.
impl PartialOrd for Dual {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.real.partial_cmp(&other.real)
    }
}

impl Neg for Dual {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Dual::new(-self.real, -self.dual)
    }
}

impl Float for Dual {
    fn nan() -> Self {
        Dual::new(f64::nan(), f64::nan())
    }

    fn infinity() -> Self {
        Dual::new(f64::infinity(), f64::infinity())
    }

    fn neg_infinity() -> Self {
        Dual::new(f64::neg_infinity(), f64::neg_infinity())
    }

    fn neg_zero() -> Self {
        Dual::new(f64::neg_zero(), f64::neg_zero())
    }

    fn min_value() -> Self {
        Dual::new(f64::min_value(), f64::min_value())
    }

    fn min_positive_value() -> Self {
        Dual::new(f64::min_positive_value(), f64::min_positive_value())
    }

    fn max_value() -> Self {
        Dual::new(f64::max_value(), f64::max_value())
    }

    fn is_nan(self) -> bool {
        self.real.is_nan()
    }

    fn is_infinite(self) -> bool {
        self.real.is_infinite()
    }

    fn is_finite(self) -> bool {
        self.real.is_finite()
    }

    fn is_normal(self) -> bool {
        self.real.is_normal()
    }

    fn classify(self) -> FpCategory {
        self.real.classify()
    }

    fn floor(self) -> Self {
        Dual::from_real(self.real.floor())
    }

    fn ceil(self) -> Self {
        Dual::from_real(self.real.ceil())
    }

    fn round(self) -> Self {
        Dual::from_real(self.real.round())
    }

    fn trunc(self) -> Self {
        Dual::from_real(self.real.trunc())
    }

    fn fract(self) -> Self {
        Dual::new(self.real.fract(), self.dual)
    }

    fn abs(self) -> Self {
        Dual::new(self.real.abs(), self.dual * self.real.signum())
    }

    fn signum(self) -> Self {
        Dual::new(self.real.signum(), f64::zero())
    }

    fn is_sign_positive(self) -> bool {
        self.real.is_sign_positive()
    }

    fn is_sign_negative(self) -> bool {
        self.real.is_sign_negative()
    }

    fn mul_add(self, a: Self, b: Self) -> Self {
        Dual::new(
            self.real.mul_add(a.real, b.real),
            self.real * a.dual + self.dual * a.real + b.dual,
        )
    }

    fn recip(self) -> Self {
        Dual::new(self.real.recip(), -self.dual / self.real.powi(2))
    }

    fn powi(self, n: i32) -> Self {
        if n == 0 {
            Dual::one()
        } else {
            Dual::new(
                self.real.powi(n),
                <f64 as From<i32>>::from(n) * self.dual * self.real.powi(n - 1),
            )
        }
    }

    // Numerically-stable version.
    fn powf(self, n: Self) -> Self {
        (self.ln() * n).exp()
    }

    fn sqrt(self) -> Self {
        let sqrt_re = self.real.sqrt();
        Dual::new(sqrt_re, self.dual / (2.0 * sqrt_re))
    }

    fn exp(self) -> Self {
        let exp_re = self.real.exp();
        Dual::new(exp_re, exp_re * self.dual)
    }

    fn exp2(self) -> Self {
        let exp2_re = self.real.exp2();
        Dual::new(exp2_re, exp2_re * LN_2 * self.dual)
    }

    fn ln(self) -> Self {
        Dual::new(self.real.ln(), self.dual / self.real)
    }

    fn log(self, base: Self) -> Self {
        Dual::new(
            self.real.log(base.real),
            self.dual / (self.real * base.real.ln()),
        )
    }

    fn log2(self) -> Self {
        Dual::new(self.real.ln() / LN_2, self.dual / (self.real * LN_2))
    }

    fn log10(self) -> Self {
        Dual::new(self.real.ln() / LN_10, self.dual / (self.real * LN_10))
    }

    fn max(self, other: Self) -> Self {
        if self.real > other.real { self } else { other }
    }

    fn min(self, other: Self) -> Self {
        if self.real < other.real { self } else { other }
    }

    #[allow(deprecated)]
    fn abs_sub(self, other: Self) -> Self {
        if self.real > other.real {
            self - other
        } else {
            Self::zero()
        }
    }

    fn cbrt(self) -> Self {
        let cbrt_re = self.real.cbrt();
        Dual::new(cbrt_re, self.dual / (3.0 * cbrt_re.powi(2)))
    }

    fn hypot(self, other: Self) -> Self {
        let hypot_real = (self.real.powi(2) + other.real.powi(2)).sqrt();
        Dual::new(
            hypot_real,
            (self.real * self.dual + other.real * other.dual) / hypot_real,
        )
    }

    fn sin(self) -> Self {
        Dual::new(self.real.sin(), self.real.cos() * self.dual)
    }

    fn cos(self) -> Self {
        Dual::new(self.real.cos(), -self.real.sin() * self.dual)
    }

    fn tan(self) -> Self {
        let re_tan = self.real.tan();
        Dual::new(re_tan, self.dual / (self.real.cos().powi(2)))
    }

    fn asin(self) -> Self {
        Dual::new(
            self.real.asin(),
            self.dual / (1.0 - self.real.powi(2)).sqrt(),
        )
    }

    fn acos(self) -> Self {
        Dual::new(
            self.real.acos(),
            -self.dual / (1.0 - self.real.powi(2)).sqrt(),
        )
    }

    fn atan(self) -> Self {
        Dual::new(self.real.atan(), self.dual / (1.0 + self.real.powi(2)))
    }

    fn atan2(self, other: Self) -> Self {
        Dual::new(
            self.real.atan2(other.real),
            (self.dual * other.real - self.real * other.dual)
                / (self.real.powi(2) + other.real.powi(2)),
        )
    }

    fn sin_cos(self) -> (Self, Self) {
        (
            Dual::new(self.real.sin(), self.real.cos() * self.dual),
            Dual::new(self.real.cos(), -self.real.sin() * self.dual),
        )
    }

    fn exp_m1(self) -> Self {
        let exp_re = self.real.exp();
        Dual::new(exp_re - 1.0, self.dual * exp_re)
    }

    fn ln_1p(self) -> Self {
        Dual::new(self.real.ln_1p(), self.dual / (1.0 + self.real))
    }

    fn sinh(self) -> Self {
        Dual::new(self.real.sinh(), self.dual * self.real.cosh())
    }

    fn cosh(self) -> Self {
        Dual::new(self.real.cosh(), self.dual * self.real.sinh())
    }

    fn tanh(self) -> Self {
        let tanh_re = self.real.tanh();
        Dual::new(tanh_re, self.dual * (1.0 - tanh_re.powi(2)))
    }

    fn asinh(self) -> Self {
        Dual::new(
            self.real.asinh(),
            self.dual / (self.real.powi(2) + 1.0).sqrt(),
        )
    }

    fn acosh(self) -> Self {
        Dual::new(
            self.real.acosh(),
            self.dual / (self.real.powi(2) - 1.0).sqrt(),
        )
    }

    fn atanh(self) -> Self {
        Dual::new(self.real.atanh(), self.dual / (1.0 - self.real.powi(2)))
    }

    // This method is really irrelevant, but we need to implement it anyway to satisfy the Float
    // trait.
    fn integer_decode(self) -> (u64, i16, i8) {
        self.real.integer_decode()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numtest::*;
    use std::f64::consts::{E, FRAC_PI_4, FRAC_PI_6};

    #[test]
    fn test_partial_ord() {
        // Check <.
        assert!(Dual::new(1.0, 2.0) < Dual::new(3.0, 4.0));
        assert!(Dual::new(1.0, 4.0) < Dual::new(3.0, 2.0));
        assert!(Dual::new(-3.0, -4.0) < Dual::new(-1.0, -2.0));
        assert!(Dual::new(-3.0, -2.0) < Dual::new(-1.0, -4.0));

        // Check >.
        assert!(Dual::new(3.0, 4.0) > Dual::new(1.0, 2.0));
        assert!(Dual::new(3.0, 2.0) > Dual::new(1.0, 4.0));
        assert!(Dual::new(-1.0, -2.0) > Dual::new(-3.0, -4.0));
        assert!(Dual::new(-1.0, -4.0) > Dual::new(-3.0, -2.0));

        // Check <=.
        assert!(Dual::new(0.0, 2.0) <= Dual::new(1.0, 2.0));
        assert!(Dual::new(1.0, 2.0) <= Dual::new(1.0, 2.0));

        // Check >=.
        assert!(Dual::new(2.0, 2.0) >= Dual::new(1.0, 2.0));
        assert!(Dual::new(1.0, 2.0) >= Dual::new(1.0, 2.0));
    }

    #[test]
    fn test_neg() {
        assert_eq!(-Dual::new(1.0, 2.0), Dual::new(-1.0, -2.0));
        assert_eq!(-Dual::new(1.0, -2.0), Dual::new(-1.0, 2.0));
        assert_eq!(-Dual::new(-1.0, 2.0), Dual::new(1.0, -2.0));
        assert_eq!(-Dual::new(-1.0, -2.0), Dual::new(1.0, 2.0));
    }

    #[test]
    fn test_nan() {
        let num = Dual::nan();
        assert!(num.get_real().is_nan());
        assert!(num.get_dual().is_nan());
    }

    #[test]
    fn test_infinity() {
        let num = Dual::infinity();
        assert!(num.get_real().is_infinite() & (num.get_real() > 0.0));
        assert!(num.get_dual().is_infinite() & (num.get_dual() > 0.0));
    }

    #[test]
    fn test_neg_infinity() {
        let num = Dual::neg_infinity();
        assert!(num.get_real().is_infinite() & (num.get_real() < 0.0));
        assert!(num.get_dual().is_infinite() & (num.get_dual() < 0.0));
    }

    #[test]
    fn test_neg_zero() {
        let num = Dual::neg_zero();
        assert!(num.get_real().is_zero());
        assert!(num.get_dual().is_zero());
    }

    #[test]
    fn test_min_value() {
        let num = Dual::min_value();
        assert!(num.get_real() == f64::MIN);
        assert!(num.get_dual() == f64::MIN);
    }

    #[test]
    fn test_min_positive_value() {
        let num = Dual::min_positive_value();
        assert!(num.get_real() == f64::MIN_POSITIVE);
        assert!(num.get_dual() == f64::MIN_POSITIVE);
    }

    #[test]
    fn test_max_value() {
        let num = Dual::max_value();
        assert!(num.get_real() == f64::MAX);
        assert!(num.get_dual() == f64::MAX);
    }

    #[test]
    fn test_is_nan() {
        assert!(Dual::nan().is_nan());
        assert!(Dual::from_real(f64::NAN).is_nan());
        assert!(!Dual::new(0.0, f64::NAN).is_nan());
        assert!(!Dual::from_real(0.0).is_nan());
    }

    #[test]
    fn test_is_infinite() {
        assert!(Dual::infinity().is_infinite());
        assert!(Dual::neg_infinity().is_infinite());
        assert!(Dual::from_real(f64::INFINITY).is_infinite());
        assert!(Dual::from_real(f64::NEG_INFINITY).is_infinite());
        assert!(!Dual::new(0.0, f64::INFINITY).is_infinite());
        assert!(!Dual::new(0.0, f64::NEG_INFINITY).is_infinite());
        assert!(!Dual::from_real(0.0).is_infinite());
    }

    #[test]
    fn test_is_finite() {
        assert!(!Dual::infinity().is_finite());
        assert!(!Dual::neg_infinity().is_finite());
        assert!(!Dual::from_real(f64::INFINITY).is_finite());
        assert!(!Dual::from_real(f64::NEG_INFINITY).is_finite());
        assert!(Dual::new(0.0, f64::INFINITY).is_finite());
        assert!(Dual::new(0.0, f64::NEG_INFINITY).is_finite());
        assert!(Dual::from_real(0.0).is_finite());
    }

    /// # References
    ///
    /// * <https://docs.rs/num-traits/latest/num_traits/float/trait.Float.html#tymethod.is_normal>
    ///
    /// # Note
    ///
    /// For each of these tests, we use a dual part of `f64::NAN` to ensure that `is_normal` is only
    /// checking the real part.
    #[test]
    fn test_is_normal() {
        // Normal (for these checks we use a not-normal dual part to ensure that only the real part
        // is being checked).
        assert!(Dual::new(1.0, f64::NAN).is_normal());
        assert!(Dual::new(f64::MIN_POSITIVE, f64::NAN).is_normal());
        assert!(Dual::new(f64::MAX, f64::NAN).is_normal());

        // Not normal (for these checks we use a normal dual part to ensure that only the real part
        // is being checked).
        assert!(!Dual::new(0.0, 1.0).is_normal()); // Zero.
        assert!(!Dual::new(f64::NAN, 1.0).is_normal()); // NaN.
        assert!(!Dual::new(f64::INFINITY, 1.0).is_normal()); // Infinite.
        assert!(!Dual::new(f64::NEG_INFINITY, 1.0).is_normal()); // Infinite.
        assert!(!Dual::new(1.0e-308_f64, 1.0).is_normal()); // Subnormal (between 0 and f64::MIN).
    }

    #[test]
    fn test_classify() {
        // Normal (for these checks we use a not-normal dual part to ensure that only the real part
        // is being checked).
        assert_eq!(Dual::new(1.0, f64::NAN).classify(), FpCategory::Normal);
        assert_eq!(
            Dual::new(f64::MIN_POSITIVE, f64::NAN).classify(),
            FpCategory::Normal
        );
        assert_eq!(Dual::new(f64::MAX, f64::NAN).classify(), FpCategory::Normal);

        // Not normal (for these checks we use a normal dual part to ensure that only the real part
        // is being checked).
        assert_eq!(Dual::new(0.0, 1.0).classify(), FpCategory::Zero);
        assert_eq!(Dual::new(f64::NAN, 1.0).classify(), FpCategory::Nan);
        assert_eq!(
            Dual::new(f64::INFINITY, 1.0).classify(),
            FpCategory::Infinite
        );
        assert_eq!(
            Dual::new(f64::NEG_INFINITY, 1.0).classify(),
            FpCategory::Infinite
        );
        assert_eq!(
            Dual::new(1.0e-308_f64, 1.0).classify(),
            FpCategory::Subnormal
        );
    }

    #[test]
    fn test_floor() {
        assert_eq!(Dual::new(2.7, 2.7).floor(), Dual::from_real(2.0));
        assert_eq!(Dual::new(-2.7, -2.7).floor(), Dual::from_real(-3.0));
    }

    #[test]
    fn test_ceil() {
        assert_eq!(Dual::new(2.7, 2.7).ceil(), Dual::from_real(3.0));
        assert_eq!(Dual::new(-2.7, -2.7).ceil(), Dual::from_real(-2.0));
    }

    #[test]
    fn test_round() {
        assert_eq!(Dual::new(2.3, 2.3).round(), Dual::from_real(2.0));
        assert_eq!(Dual::new(2.7, 2.7).round(), Dual::from_real(3.0));
        assert_eq!(Dual::new(-2.7, -2.7).round(), Dual::from_real(-3.0));
        assert_eq!(Dual::new(-2.3, -2.3).round(), Dual::from_real(-2.0));
    }

    #[test]
    fn test_trunc() {
        assert_eq!(Dual::new(2.3, 2.3).trunc(), Dual::from_real(2.0));
        assert_eq!(Dual::new(2.7, 2.7).trunc(), Dual::from_real(2.0));
        assert_eq!(Dual::new(-2.7, -2.7).trunc(), Dual::from_real(-2.0));
        assert_eq!(Dual::new(-2.3, -2.3).trunc(), Dual::from_real(-2.0));
    }

    #[test]
    fn test_fract() {
        assert_eq!(Dual::new(2.5, 2.5).fract(), Dual::new(0.5, 2.5));
        assert_eq!(Dual::new(-2.5, -2.5).fract(), Dual::new(-0.5, -2.5));
    }

    #[test]
    fn test_abs() {
        assert_eq!(Dual::new(1.0, 2.0).abs(), Dual::new(1.0, 2.0));
        assert_eq!(Dual::new(-1.0, -2.0).abs(), Dual::new(1.0, 2.0));
        assert_eq!(Dual::new(-1.0, 2.0).abs(), Dual::new(1.0, -2.0));
    }

    #[test]
    fn test_signum() {
        assert_eq!(Dual::new(2.0, 4.0).signum(), Dual::from_real(1.0));
        assert_eq!(Dual::new(-2.0, -4.0).signum(), Dual::from_real(-1.0));
    }

    #[test]
    fn test_is_sign_positive() {
        assert!(Dual::new(2.0, -4.0).is_sign_positive());
        assert!(!Dual::new(-2.0, 4.0).is_sign_positive());
    }

    #[test]
    fn test_is_sign_negative() {
        assert!(Dual::new(-2.0, 4.0).is_sign_negative());
        assert!(!Dual::new(2.0, -4.0).is_sign_negative());
    }

    #[test]
    fn test_mul_add() {
        let a = Dual::new(1.0, 3.0);
        let b = Dual::new(-2.0, 5.0);
        let c = Dual::new(10.0, -4.0);
        assert_eq!(c.mul_add(a, b), (c * a) + b);
    }

    #[test]
    fn test_recip() {
        assert_eq!(Dual::new(2.0, -5.0).recip(), Dual::new(0.5, 1.25));
    }

    #[test]
    fn test_powi() {
        assert_eq!(Dual::new(2.0, -5.0).powi(0), Dual::from_real(1.0));
        assert_eq!(Dual::new(2.0, -5.0).powi(1), Dual::new(2.0, -5.0));
        assert_eq!(Dual::new(2.0, -5.0).powi(2), Dual::new(4.0, -20.0));
        assert_eq!(Dual::new(2.0, -5.0).powi(3), Dual::new(8.0, -60.0));
    }

    #[test]
    fn test_powf() {
        // Test against powi for integer powers.
        assert_eq!(
            Dual::new(2.0, -5.0).powf(Dual::from_real(0.0)),
            Dual::new(2.0, -5.0).powi(0)
        );
        assert_eq!(
            Dual::new(2.0, -5.0).powf(Dual::from_real(1.0)),
            Dual::new(2.0, -5.0).powi(1)
        );
        assert_eq!(
            Dual::new(2.0, -5.0).powf(Dual::from_real(2.0)),
            Dual::new(2.0, -5.0).powi(2)
        );
        assert_equal_to_decimal!(
            Dual::new(2.0, -5.0).powf(Dual::from_real(3.0)),
            Dual::new(2.0, -5.0).powi(3),
            14
        );

        // Test against sqrt.
        assert_equal_to_decimal!(
            Dual::new(2.0, -5.0).powf(Dual::from_real(0.5)),
            Dual::new(2.0, -5.0).sqrt(),
            15
        );

        // Test against cbrt.
        assert_equal_to_decimal!(
            Dual::new(2.0, -5.0).powf(Dual::from_real(1.0 / 3.0)),
            Dual::new(2.0, -5.0).cbrt(),
            15
        );

        // Spot check.
        assert_eq!(
            Dual::new(2.0, -5.0).powf(Dual::new(-5.0, 4.0)),
            Dual::new(0.03125, 0.4772683975699932)
        );
    }

    #[test]
    fn test_sqrt() {
        assert_eq!(Dual::new(4.0, 25.0).sqrt(), Dual::new(2.0, 6.25));
    }

    #[test]
    fn test_exp() {
        assert_eq!(
            Dual::new(2.0, -3.0).exp(),
            Dual::new(2.0.exp(), -3.0 * 2.0.exp())
        );
    }

    #[test]
    fn test_exp2() {
        assert_eq!(
            Dual::new(2.0, -3.0).exp2(),
            Dual::new(2.0.exp2(), -8.317766166719343)
        );
    }

    #[test]
    fn test_ln() {
        assert_eq!(Dual::new(5.0, 8.0).ln(), Dual::new(5.0.ln(), 8.0 / 5.0));
    }

    #[test]
    fn test_log() {
        assert_eq!(
            Dual::new(5.0, 8.0).log(Dual::from_real(4.5)),
            Dual::new(5.0.log(4.5), 1.0637750447080176)
        );
    }

    #[test]
    fn test_log2() {
        assert_eq!(
            Dual::new(5.0, 8.0).log2(),
            Dual::new(5.0.log2(), 2.3083120654223412)
        );
    }

    #[test]
    fn test_log10() {
        assert_equal_to_decimal!(
            Dual::new(5.0, 8.0).log10(),
            Dual::new(5.0.log10(), 0.6948711710452028),
            16
        );
    }

    #[test]
    fn test_max() {
        assert_eq!(
            Dual::new(1.0, 2.0).max(Dual::new(3.0, 4.0)),
            Dual::new(3.0, 4.0)
        );
        assert_eq!(
            Dual::new(3.0, 2.0).max(Dual::new(1.0, 4.0)),
            Dual::new(3.0, 2.0)
        );
        assert_eq!(
            Dual::new(3.0, 4.0).max(Dual::new(1.0, 2.0)),
            Dual::new(3.0, 4.0)
        );
        assert_eq!(
            Dual::new(-1.0, 2.0).max(Dual::new(-3.0, 4.0)),
            Dual::new(-1.0, 2.0)
        );
    }

    #[test]
    fn test_min() {
        assert_eq!(
            Dual::new(1.0, 2.0).min(Dual::new(3.0, 4.0)),
            Dual::new(1.0, 2.0)
        );
        assert_eq!(
            Dual::new(3.0, 2.0).min(Dual::new(1.0, 4.0)),
            Dual::new(1.0, 4.0)
        );
        assert_eq!(
            Dual::new(3.0, 4.0).min(Dual::new(1.0, 2.0)),
            Dual::new(1.0, 2.0)
        );
        assert_eq!(
            Dual::new(-1.0, 2.0).min(Dual::new(-3.0, 4.0)),
            Dual::new(-3.0, 4.0)
        );
    }

    #[test]
    fn test_abs_sub() {
        assert_eq!(
            Dual::new(4.0, 5.0).abs_sub(Dual::new(2.0, 8.0)),
            Dual::new(2.0, -3.0)
        );
    }

    #[test]
    fn test_cbrt() {
        assert_eq!(Dual::new(8.0, 27.0).cbrt(), Dual::new(2.0, 2.25));
    }

    #[test]
    fn test_hypot() {
        // Spot check.
        assert_eq!(
            Dual::new(1.0, 2.0).hypot(Dual::new(3.0, 4.0)),
            Dual::new(3.1622776601683795, 4.427188724235731)
        );

        // Check parity with Euclidian norm.
        assert_eq!(
            Dual::new(1.0, 2.0).hypot(Dual::new(3.0, 4.0)),
            (Dual::new(1.0, 2.0).powi(2) + Dual::new(3.0, 4.0).powi(2)).sqrt()
        );
    }

    #[test]
    fn test_sin() {
        assert_eq!(Dual::new(FRAC_PI_6, 2.0).sin(), Dual::new(0.5, 3.0.sqrt()));
    }

    #[test]
    fn test_cos() {
        assert_eq!(
            Dual::new(FRAC_PI_6, 2.0).cos(),
            Dual::new(3.0.sqrt() / 2.0, -1.0)
        );
    }

    #[test]
    fn test_tan() {
        assert_equal_to_decimal!(
            Dual::new(FRAC_PI_6, 2.0).tan(),
            Dual::new(3.0.sqrt() / 3.0, 8.0 / 3.0),
            15
        );
    }

    #[test]
    fn test_asin() {
        assert_equal_to_decimal!(
            Dual::new(0.5, 3.0).asin(),
            Dual::new(FRAC_PI_6, 3.0 / 0.75.sqrt()),
            16
        );
    }

    #[test]
    fn test_asin_near_domain_edges() {
        for x in [-0.99_f64, -0.9, 0.9, 0.99] {
            let dual = 1.7;
            assert_equal_to_decimal!(
                Dual::new(x, dual).asin(),
                Dual::new(x.asin(), dual / (1.0 - x.powi(2)).sqrt()),
                13
            );
        }
    }

    #[test]
    fn test_asin_out_of_domain_nan() {
        assert!(Dual::new(1.0001, 2.0).asin().get_real().is_nan());
        assert!(Dual::new(-1.0001, 2.0).asin().get_real().is_nan());
    }

    #[test]
    fn test_acos() {
        assert_equal_to_decimal!(
            Dual::new(3.0.sqrt() / 2.0, 3.0).acos(),
            Dual::new(FRAC_PI_6, -6.0),
            15
        );
    }

    #[test]
    fn test_acos_near_domain_edges() {
        for x in [-0.99_f64, -0.9, 0.9, 0.99] {
            let dual = -1.3;
            assert_equal_to_decimal!(
                Dual::new(x, dual).acos(),
                Dual::new(x.acos(), -dual / (1.0 - x.powi(2)).sqrt()),
                13
            );
        }
    }

    #[test]
    fn test_acos_out_of_domain_nan() {
        assert!(Dual::new(1.0001, 2.0).acos().get_real().is_nan());
        assert!(Dual::new(-1.0001, 2.0).acos().get_real().is_nan());
    }

    #[test]
    fn test_atan() {
        assert_eq!(Dual::new(1.0, 3.0).atan(), Dual::new(FRAC_PI_4, 1.5));
    }

    #[test]
    fn test_atan2() {
        let x = Dual::new(3.0, 2.0);
        let y = Dual::new(-3.0, 5.0);
        let angle_expected = Dual::new(-FRAC_PI_4, 7.0 / 6.0);
        assert_eq!(y.atan2(x), angle_expected);
    }

    #[test]
    fn test_sin_cos() {
        let (sin, cos) = Dual::new(FRAC_PI_6, 2.0).sin_cos();
        assert_eq!(sin, Dual::new(0.5, 3.0.sqrt()));
        assert_eq!(cos, Dual::new(3.0.sqrt() / 2.0, -1.0));
    }

    #[test]
    fn test_exp_m1() {
        assert_eq!(
            Dual::new(3.0, 5.0).exp_m1(),
            Dual::new(3.0, 5.0).exp() - Dual::one()
        );
    }

    #[test]
    fn test_ln_1p() {
        assert_eq!(
            Dual::new(3.0, 5.0).ln_1p(),
            (Dual::new(3.0, 5.0) + Dual::one()).ln()
        );
    }

    #[test]
    fn test_sinh() {
        assert_equal_to_decimal!(
            Dual::new(1.0, 2.0).sinh(),
            Dual::new(((E * E) - 1.0) / (2.0 * E), ((E * E) + 1.0) / E),
            15
        );
    }

    #[test]
    fn test_cosh() {
        assert_equal_to_decimal!(
            Dual::new(1.0, 2.0).cosh(),
            Dual::new(((E * E) + 1.0) / (2.0 * E), ((E * E) - 1.0) / E),
            15
        );
    }

    #[test]
    fn test_tanh() {
        assert_equal_to_decimal!(
            Dual::new(1.0, 2.0).tanh(),
            Dual::new(
                (1.0 - E.powi(-2)) / (1.0 + E.powi(-2)),
                2.0 * ((2.0 * E) / (E.powi(2) + 1.0)).powi(2)
            ),
            15
        );
    }

    #[test]
    fn test_asinh() {
        assert_eq!(Dual::new(1.0, 2.0).sinh().asinh(), Dual::new(1.0, 2.0));
    }

    #[test]
    fn test_acosh() {
        assert_eq!(Dual::new(1.0, 2.0).cosh().acosh(), Dual::new(1.0, 2.0));
    }

    #[test]
    fn test_acosh_near_domain_boundary() {
        for x in [1.0001_f64, 1.01, 2.0] {
            let dual = 1.2;
            assert_equal_to_decimal!(
                Dual::new(x, dual).acosh(),
                Dual::new(x.acosh(), dual / ((x - 1.0).sqrt() * (x + 1.0).sqrt())),
                11
            );
        }
    }

    #[test]
    fn test_acosh_out_of_domain_nan() {
        assert!(Dual::new(0.9999, 2.0).acosh().get_real().is_nan());
    }

    #[test]
    fn test_atanh() {
        assert_equal_to_decimal!(Dual::new(1.0, 2.0).tanh().atanh(), Dual::new(1.0, 2.0), 16);
    }

    #[test]
    fn test_atanh_near_domain_edges() {
        for x in [-0.99_f64, -0.9, 0.9, 0.99] {
            let dual = -0.8;
            assert_equal_to_decimal!(
                Dual::new(x, dual).atanh(),
                Dual::new(x.atanh(), dual / (1.0 - x.powi(2))),
                12
            );
        }
    }

    #[test]
    fn test_atanh_out_of_domain_nan() {
        assert!(Dual::new(1.0001, 2.0).atanh().get_real().is_nan());
        assert!(Dual::new(-1.0001, 2.0).atanh().get_real().is_nan());
    }

    #[test]
    fn test_integer_decode() {
        assert_eq!(
            Dual::new(1.2345e-5, 6.789e-7).integer_decode(),
            (1.2345e-5).integer_decode()
        );
    }
}
