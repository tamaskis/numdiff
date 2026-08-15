//! Implements [`linalg_traits::real_field::RealFieldBase`] for [`crate::Dual`].

#![allow(clippy::used_underscore_items)]

use crate::automatic_differentiation::dual::dual::Dual;
use linalg_traits::real_field::RealFieldBase;
use linalg_traits::verify_trait_implemented;
use std::f64::consts::{LN_2, LN_10};
use std::num::FpCategory;

const _: bool = verify_trait_implemented!(Dual: RealFieldBase);

impl RealFieldBase for Dual {
    #[inline]
    fn _from_f64(value: f64) -> Self {
        Self::from_real(value)
    }

    #[inline]
    fn _to_f64(&self) -> f64 {
        self.real
    }

    #[inline]
    fn _neg(self) -> Self {
        Dual::new(-self.real, -self.dual)
    }

    #[inline]
    fn _add(self, rhs: Self) -> Self {
        Dual::new(self.real + rhs.real, self.dual + rhs.dual)
    }

    #[inline]
    fn _sub(self, rhs: Self) -> Self {
        Dual::new(self.real - rhs.real, self.dual - rhs.dual)
    }

    #[inline]
    fn _mul(self, rhs: Self) -> Self {
        Dual::new(
            self.real * rhs.real,
            self.dual * rhs.real + self.real * rhs.dual,
        )
    }

    #[inline]
    fn _div(self, rhs: Self) -> Self {
        Dual::new(
            self.real / rhs.real,
            (self.dual * rhs.real - self.real * rhs.dual) / rhs.real.powi(2),
        )
    }

    #[inline]
    fn _rem(self, rhs: Self) -> Self {
        Dual::new(
            self.real % rhs.real,
            self.dual - (self.real / rhs.real).trunc() * rhs.dual,
        )
    }

    #[inline]
    fn _eq(self, rhs: Self) -> bool {
        self.real == rhs.real && self.dual == rhs.dual
    }

    #[inline]
    fn _partial_cmp(self, rhs: Self) -> Option<std::cmp::Ordering> {
        // Only the real part is compared. This is primarily to support numerical methods where we
        // want to check convergence on the actual function evaluation, and NOT its derivative.
        self.real.partial_cmp(&rhs.real)
    }

    #[inline]
    fn _zero() -> Self {
        Dual::from_real(0.0)
    }

    #[inline]
    fn _one() -> Self {
        Dual::from_real(1.0)
    }

    #[inline]
    fn _abs(&self) -> Self {
        Dual::new(self.real.abs(), self.dual * self.real.signum())
    }

    #[inline]
    fn _hypot(self, other: Self) -> Self {
        let hypot_real = (self.real.powi(2) + other.real.powi(2)).sqrt();
        Dual::new(
            hypot_real,
            (self.real * self.dual + other.real * other.dual) / hypot_real,
        )
    }

    #[inline]
    fn _recip(self) -> Self {
        Dual::new(self.real.recip(), -self.dual / self.real.powi(2))
    }

    #[inline]
    fn _mul_add(self, a: Self, b: Self) -> Self {
        Dual::new(
            self.real.mul_add(a.real, b.real),
            self.real * a.dual + self.dual * a.real + b.dual,
        )
    }

    #[inline]
    fn _sqrt(self) -> Self {
        let sqrt_re = self.real.sqrt();
        Dual::new(sqrt_re, self.dual / (2.0 * sqrt_re))
    }

    #[inline]
    fn _cbrt(self) -> Self {
        let cbrt_re = self.real.cbrt();
        Dual::new(cbrt_re, self.dual / (3.0 * cbrt_re.powi(2)))
    }

    #[inline]
    fn _powi(self, n: i32) -> Self {
        if n == 0 {
            Self::_one()
        } else {
            Dual::new(
                self.real.powi(n),
                <f64 as From<i32>>::from(n) * self.dual * self.real.powi(n - 1),
            )
        }
    }

    // Numerically-stable version.
    #[inline]
    fn _powf(self, n: Self) -> Self {
        self._ln()._mul(n)._exp()
    }

    #[inline]
    fn _exp(self) -> Self {
        let exp_re = self.real.exp();
        Dual::new(exp_re, exp_re * self.dual)
    }

    #[inline]
    fn _exp2(self) -> Self {
        let exp2_re = self.real.exp2();
        Dual::new(exp2_re, exp2_re * LN_2 * self.dual)
    }

    #[inline]
    fn _exp_m1(self) -> Self {
        let exp_re = self.real.exp();
        Dual::new(exp_re - 1.0, self.dual * exp_re)
    }

    #[inline]
    fn _ln(self) -> Self {
        Dual::new(self.real.ln(), self.dual / self.real)
    }

    #[inline]
    fn _ln_1p(self) -> Self {
        Dual::new(self.real.ln_1p(), self.dual / (1.0 + self.real))
    }

    #[inline]
    fn _log(self, base: Self) -> Self {
        Dual::new(
            self.real.log(base.real),
            self.dual / (self.real * base.real.ln()),
        )
    }

    #[inline]
    fn _log2(self) -> Self {
        Dual::new(self.real.ln() / LN_2, self.dual / (self.real * LN_2))
    }

    #[inline]
    fn _log10(self) -> Self {
        Dual::new(self.real.ln() / LN_10, self.dual / (self.real * LN_10))
    }

    #[inline]
    fn _sin(self) -> Self {
        Dual::new(self.real.sin(), self.real.cos() * self.dual)
    }

    #[inline]
    fn _cos(self) -> Self {
        Dual::new(self.real.cos(), -self.real.sin() * self.dual)
    }

    #[inline]
    fn _sin_cos(self) -> (Self, Self) {
        (self._sin(), self._cos())
    }

    #[inline]
    fn _tan(self) -> Self {
        let re_tan = self.real.tan();
        Dual::new(re_tan, self.dual / (self.real.cos().powi(2)))
    }

    #[inline]
    fn _asin(self) -> Self {
        Dual::new(
            self.real.asin(),
            self.dual / (1.0 - self.real.powi(2)).sqrt(),
        )
    }

    #[inline]
    fn _acos(self) -> Self {
        Dual::new(
            self.real.acos(),
            -self.dual / (1.0 - self.real.powi(2)).sqrt(),
        )
    }

    #[inline]
    fn _atan(self) -> Self {
        Dual::new(self.real.atan(), self.dual / (1.0 + self.real.powi(2)))
    }

    #[inline]
    fn _atan2(self, other: Self) -> Self {
        Dual::new(
            self.real.atan2(other.real),
            (self.dual * other.real - self.real * other.dual)
                / (self.real.powi(2) + other.real.powi(2)),
        )
    }

    #[inline]
    fn _sinh(self) -> Self {
        Dual::new(self.real.sinh(), self.dual * self.real.cosh())
    }

    #[inline]
    fn _cosh(self) -> Self {
        Dual::new(self.real.cosh(), self.dual * self.real.sinh())
    }

    #[inline]
    fn _tanh(self) -> Self {
        let tanh_re = self.real.tanh();
        Dual::new(tanh_re, self.dual * (1.0 - tanh_re.powi(2)))
    }

    #[inline]
    fn _asinh(self) -> Self {
        Dual::new(
            self.real.asinh(),
            self.dual / (self.real.powi(2) + 1.0).sqrt(),
        )
    }

    #[inline]
    fn _acosh(self) -> Self {
        Dual::new(
            self.real.acosh(),
            self.dual / (self.real.powi(2) - 1.0).sqrt(),
        )
    }

    #[inline]
    fn _atanh(self) -> Self {
        Dual::new(self.real.atanh(), self.dual / (1.0 - self.real.powi(2)))
    }

    #[inline]
    fn _floor(self) -> Self {
        Dual::from_real(self.real.floor())
    }

    #[inline]
    fn _ceil(self) -> Self {
        Dual::from_real(self.real.ceil())
    }

    #[inline]
    fn _round(self) -> Self {
        Dual::from_real(self.real.round())
    }

    #[inline]
    fn _trunc(self) -> Self {
        Dual::from_real(self.real.trunc())
    }

    #[inline]
    fn _fract(self) -> Self {
        Dual::new(self.real.fract(), self.dual)
    }

    #[inline]
    fn _copysign(self, sign: Self) -> Self {
        if self.real.is_sign_negative() == sign.real.is_sign_negative() {
            self
        } else {
            self._neg()
        }
    }

    #[inline]
    fn _min(self, other: Self) -> Self {
        if self.real < other.real { self } else { other }
    }

    #[inline]
    fn _max(self, other: Self) -> Self {
        if self.real > other.real { self } else { other }
    }

    #[inline]
    fn _clamp(self, min: Self, max: Self) -> Self {
        if self.real < min.real {
            min
        } else if self.real > max.real {
            max
        } else {
            self
        }
    }

    #[inline]
    fn _is_nan(self) -> bool {
        self.real.is_nan()
    }

    #[inline]
    fn _is_infinite(self) -> bool {
        self.real.is_infinite()
    }

    #[inline]
    fn _is_finite(&self) -> bool {
        self.real.is_finite()
    }

    #[inline]
    fn _is_subnormal(self) -> bool {
        self.real.is_subnormal()
    }

    #[inline]
    fn _is_normal(self) -> bool {
        self.real.is_normal()
    }

    #[inline]
    fn _classify(self) -> FpCategory {
        self.real.classify()
    }

    #[inline]
    fn _is_sign_positive(&self) -> bool {
        self.real.is_sign_positive()
    }

    #[inline]
    fn _is_sign_negative(&self) -> bool {
        self.real.is_sign_negative()
    }

    #[inline]
    fn _next_up(self) -> Self {
        Dual::new(self.real.next_up(), self.dual)
    }

    #[inline]
    fn _next_down(self) -> Self {
        Dual::new(self.real.next_down(), self.dual)
    }

    #[inline]
    fn _epsilon() -> Self {
        Dual::from_real(f64::EPSILON)
    }

    #[inline]
    fn _bits() -> usize {
        64
    }

    #[inline]
    fn _min_positive() -> Self {
        Dual::from_real(f64::MIN_POSITIVE)
    }

    #[inline]
    fn _max_positive() -> Self {
        Dual::from_real(f64::MAX)
    }

    #[inline]
    fn _min_value() -> Option<Self> {
        Some(Dual::from_real(f64::MIN))
    }

    #[inline]
    fn _max_value() -> Option<Self> {
        Some(Dual::from_real(f64::MAX))
    }

    #[inline]
    fn _nan() -> Self {
        Dual::new(f64::NAN, f64::NAN)
    }

    #[inline]
    fn _infinity() -> Self {
        Dual::new(f64::INFINITY, f64::INFINITY)
    }

    #[inline]
    fn _as_slice(&self) -> &[f64] {
        // SAFETY: `Dual` is `#[repr(C)]` with two contiguous `f64` fields (`real` and `dual`), so
        // reinterpreting a reference to `self` as a two-element `f64` slice is valid.
        unsafe { std::slice::from_raw_parts(std::ptr::from_ref(self).cast::<f64>(), 2) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numtest::*;
    use std::f64::consts::{E, FRAC_PI_4, FRAC_PI_6, PI};

    // NOTE: These tests call the underscore-prefixed `RealFieldBase` methods directly (rather than
    // the public `RealField` wrapper methods) because several of the public wrapper methods are
    // conditionally re-routed to `nalgebra`/`faer`-provided trait methods when those features are
    // enabled (see `linalg_traits::RealField`'s documentation), which would otherwise require
    // feature-specific imports in these tests.

    #[test]
    fn test_partial_ord() {
        assert!(Dual::new(1.0, 2.0) < Dual::new(3.0, 4.0));
        assert!(Dual::new(1.0, 4.0) < Dual::new(3.0, 2.0));
        assert!(Dual::new(3.0, 4.0) > Dual::new(1.0, 2.0));
        assert!(Dual::new(0.0, 2.0) <= Dual::new(1.0, 2.0));
        assert!(Dual::new(2.0, 2.0) >= Dual::new(1.0, 2.0));
    }

    #[test]
    fn test_neg() {
        assert_eq!(-Dual::new(1.0, 2.0), Dual::new(-1.0, -2.0));
    }

    #[test]
    fn test_as_slice() {
        assert_eq!(Dual::new(1.0, 2.0)._as_slice(), &[1.0, 2.0]);
    }

    #[test]
    fn test_nan() {
        let num = Dual::_nan();
        assert!(num.get_real().is_nan());
        assert!(num.get_dual().is_nan());
    }

    #[test]
    fn test_infinity() {
        let num = Dual::_infinity();
        assert!(num.get_real().is_infinite() & (num.get_real() > 0.0));
        assert!(num.get_dual().is_infinite() & (num.get_dual() > 0.0));
    }

    #[test]
    fn test_is_nan() {
        assert!(Dual::_nan()._is_nan());
        assert!(Dual::from_real(f64::NAN)._is_nan());
        assert!(!Dual::new(0.0, f64::NAN)._is_nan());
    }

    #[test]
    fn test_is_infinite() {
        assert!(Dual::_infinity()._is_infinite());
        assert!(!Dual::new(0.0, f64::INFINITY)._is_infinite());
    }

    #[test]
    fn test_is_finite() {
        assert!(!Dual::_infinity()._is_finite());
        assert!(Dual::new(0.0, f64::INFINITY)._is_finite());
    }

    #[test]
    fn test_is_normal() {
        assert!(Dual::new(1.0, f64::NAN)._is_normal());
        assert!(!Dual::new(0.0, 1.0)._is_normal());
    }

    #[test]
    fn test_classify() {
        assert_eq!(Dual::new(1.0, f64::NAN)._classify(), FpCategory::Normal);
        assert_eq!(Dual::new(0.0, 1.0)._classify(), FpCategory::Zero);
    }

    #[test]
    fn test_floor() {
        assert_eq!(Dual::new(2.7, 2.7)._floor(), Dual::from_real(2.0));
    }

    #[test]
    fn test_ceil() {
        assert_eq!(Dual::new(2.7, 2.7)._ceil(), Dual::from_real(3.0));
    }

    #[test]
    fn test_round() {
        assert_eq!(Dual::new(2.7, 2.7)._round(), Dual::from_real(3.0));
    }

    #[test]
    fn test_trunc() {
        assert_eq!(Dual::new(2.7, 2.7)._trunc(), Dual::from_real(2.0));
    }

    #[test]
    fn test_fract() {
        assert_eq!(Dual::new(2.5, 2.5)._fract(), Dual::new(0.5, 2.5));
    }

    #[test]
    fn test_abs() {
        assert_eq!(Dual::new(1.0, 2.0)._abs(), Dual::new(1.0, 2.0));
        assert_eq!(Dual::new(-1.0, -2.0)._abs(), Dual::new(1.0, 2.0));
        assert_eq!(Dual::new(-1.0, 2.0)._abs(), Dual::new(1.0, -2.0));
    }

    #[test]
    fn test_copysign() {
        assert_eq!(
            Dual::new(1.0, 2.0)._copysign(Dual::new(-1.0, 0.0)),
            Dual::new(-1.0, -2.0)
        );
        assert_eq!(
            Dual::new(1.0, 2.0)._copysign(Dual::new(1.0, 0.0)),
            Dual::new(1.0, 2.0)
        );
    }

    #[test]
    fn test_is_sign_positive() {
        assert!(Dual::new(2.0, -4.0)._is_sign_positive());
        assert!(!Dual::new(-2.0, 4.0)._is_sign_positive());
    }

    #[test]
    fn test_is_sign_negative() {
        assert!(Dual::new(-2.0, 4.0)._is_sign_negative());
        assert!(!Dual::new(2.0, -4.0)._is_sign_negative());
    }

    #[test]
    fn test_mul_add() {
        let a = Dual::new(1.0, 3.0);
        let b = Dual::new(-2.0, 5.0);
        let c = Dual::new(10.0, -4.0);
        assert_eq!(c._mul_add(a, b), (c * a) + b);
    }

    #[test]
    fn test_recip() {
        assert_eq!(Dual::new(2.0, -5.0)._recip(), Dual::new(0.5, 1.25));
    }

    #[test]
    fn test_powi() {
        assert_eq!(Dual::new(2.0, -5.0)._powi(0), Dual::from_real(1.0));
        assert_eq!(Dual::new(2.0, -5.0)._powi(1), Dual::new(2.0, -5.0));
        assert_eq!(Dual::new(2.0, -5.0)._powi(2), Dual::new(4.0, -20.0));
    }

    #[test]
    fn test_powf() {
        assert_eq!(
            Dual::new(2.0, -5.0)._powf(Dual::from_real(2.0)),
            Dual::new(2.0, -5.0)._powi(2)
        );
        assert_equal_to_decimal!(
            Dual::new(2.0, -5.0)._powf(Dual::from_real(0.5)),
            Dual::new(2.0, -5.0)._sqrt(),
            15
        );
    }

    #[test]
    fn test_sqrt() {
        assert_eq!(Dual::new(4.0, 25.0)._sqrt(), Dual::new(2.0, 6.25));
    }

    #[test]
    fn test_cbrt() {
        assert_eq!(Dual::new(8.0, 27.0)._cbrt(), Dual::new(2.0, 2.25));
    }

    #[test]
    fn test_exp() {
        assert_eq!(
            Dual::new(2.0, -3.0)._exp(),
            Dual::new(2.0f64.exp(), -3.0 * 2.0f64.exp())
        );
    }

    #[test]
    fn test_exp2() {
        assert_eq!(
            Dual::new(2.0, -3.0)._exp2(),
            Dual::new(2.0f64.exp2(), -8.317766166719343)
        );
    }

    #[test]
    fn test_exp_m1() {
        assert_eq!(
            Dual::new(3.0, 5.0)._exp_m1(),
            Dual::new(3.0, 5.0)._exp()._sub(Dual::_one())
        );
    }

    #[test]
    fn test_ln() {
        assert_eq!(Dual::new(5.0, 8.0)._ln(), Dual::new(5.0f64.ln(), 8.0 / 5.0));
    }

    #[test]
    fn test_ln_1p() {
        assert_eq!(
            Dual::new(3.0, 5.0)._ln_1p(),
            (Dual::new(3.0, 5.0)._add(Dual::_one()))._ln()
        );
    }

    #[test]
    fn test_log() {
        assert_eq!(
            Dual::new(5.0, 8.0)._log(Dual::from_real(4.5)),
            Dual::new(5.0f64.log(4.5), 1.0637750447080176)
        );
    }

    #[test]
    fn test_log2() {
        assert_eq!(
            Dual::new(5.0, 8.0)._log2(),
            Dual::new(5.0f64.log2(), 2.3083120654223412)
        );
    }

    #[test]
    fn test_log10() {
        assert_equal_to_decimal!(
            Dual::new(5.0, 8.0)._log10(),
            Dual::new(5.0f64.log10(), 0.6948711710452028),
            16
        );
    }

    #[test]
    fn test_max() {
        assert_eq!(
            Dual::new(1.0, 2.0)._max(Dual::new(3.0, 4.0)),
            Dual::new(3.0, 4.0)
        );
    }

    #[test]
    fn test_min() {
        assert_eq!(
            Dual::new(1.0, 2.0)._min(Dual::new(3.0, 4.0)),
            Dual::new(1.0, 2.0)
        );
    }

    #[test]
    fn test_clamp() {
        assert_eq!(
            Dual::new(5.0, 1.0)._clamp(Dual::from_real(0.0), Dual::from_real(3.0)),
            Dual::from_real(3.0)
        );
        assert_eq!(
            Dual::new(-5.0, 1.0)._clamp(Dual::from_real(0.0), Dual::from_real(3.0)),
            Dual::from_real(0.0)
        );
        assert_eq!(
            Dual::new(1.0, 1.0)._clamp(Dual::from_real(0.0), Dual::from_real(3.0)),
            Dual::new(1.0, 1.0)
        );
    }

    #[test]
    fn test_hypot() {
        assert_eq!(
            Dual::new(1.0, 2.0)._hypot(Dual::new(3.0, 4.0)),
            Dual::new(3.1622776601683795, 4.427188724235731)
        );
        assert_eq!(
            Dual::new(1.0, 2.0)._hypot(Dual::new(3.0, 4.0)),
            Dual::new(1.0, 2.0)
                ._powi(2)
                ._add(Dual::new(3.0, 4.0)._powi(2))
                ._sqrt()
        );
    }

    #[test]
    fn test_sin() {
        assert_eq!(
            Dual::new(FRAC_PI_6, 2.0)._sin(),
            Dual::new(0.5, 3.0f64.sqrt())
        );
    }

    #[test]
    fn test_cos() {
        assert_eq!(
            Dual::new(FRAC_PI_6, 2.0)._cos(),
            Dual::new(3.0f64.sqrt() / 2.0, -1.0)
        );
    }

    #[test]
    fn test_sin_cos() {
        let (sin, cos) = Dual::new(FRAC_PI_6, 2.0)._sin_cos();
        assert_eq!(sin, Dual::new(0.5, 3.0f64.sqrt()));
        assert_eq!(cos, Dual::new(3.0f64.sqrt() / 2.0, -1.0));
    }

    #[test]
    fn test_tan() {
        assert_equal_to_decimal!(
            Dual::new(FRAC_PI_6, 2.0)._tan(),
            Dual::new(3.0f64.sqrt() / 3.0, 8.0 / 3.0),
            15
        );
    }

    #[test]
    fn test_asin() {
        assert_equal_to_decimal!(
            Dual::new(0.5, 3.0)._asin(),
            Dual::new(FRAC_PI_6, 3.0 / 0.75f64.sqrt()),
            16
        );
    }

    #[test]
    fn test_asin_out_of_domain_nan() {
        assert!(Dual::new(1.0001, 2.0)._asin().get_real().is_nan());
    }

    #[test]
    fn test_acos() {
        assert_equal_to_decimal!(
            Dual::new(3.0f64.sqrt() / 2.0, 3.0)._acos(),
            Dual::new(FRAC_PI_6, -6.0),
            15
        );
    }

    #[test]
    fn test_atan() {
        assert_eq!(Dual::new(1.0, 3.0)._atan(), Dual::new(FRAC_PI_4, 1.5));
    }

    #[test]
    fn test_atan2() {
        let x = Dual::new(3.0, 2.0);
        let y = Dual::new(-3.0, 5.0);
        assert_eq!(y._atan2(x), Dual::new(-FRAC_PI_4, 7.0 / 6.0));
    }

    #[test]
    fn test_sinh() {
        assert_equal_to_decimal!(
            Dual::new(1.0, 2.0)._sinh(),
            Dual::new(((E * E) - 1.0) / (2.0 * E), ((E * E) + 1.0) / E),
            15
        );
    }

    #[test]
    fn test_cosh() {
        assert_equal_to_decimal!(
            Dual::new(1.0, 2.0)._cosh(),
            Dual::new(((E * E) + 1.0) / (2.0 * E), ((E * E) - 1.0) / E),
            15
        );
    }

    #[test]
    fn test_tanh() {
        assert_equal_to_decimal!(
            Dual::new(1.0, 2.0)._tanh(),
            Dual::new(
                (1.0 - E.powi(-2)) / (1.0 + E.powi(-2)),
                2.0 * ((2.0 * E) / (E.powi(2) + 1.0)).powi(2)
            ),
            15
        );
    }

    #[test]
    fn test_asinh() {
        assert_eq!(Dual::new(1.0, 2.0)._sinh()._asinh(), Dual::new(1.0, 2.0));
    }

    #[test]
    fn test_acosh() {
        assert_eq!(Dual::new(1.0, 2.0)._cosh()._acosh(), Dual::new(1.0, 2.0));
    }

    #[test]
    fn test_atanh() {
        assert_equal_to_decimal!(
            Dual::new(1.0, 2.0)._tanh()._atanh(),
            Dual::new(1.0, 2.0),
            16
        );
    }

    #[test]
    fn test_extended_trig_from_real_field_defaults() {
        // These are provided entirely by `linalg_traits::RealFieldBase`'s default implementations,
        // in terms of the primitives implemented above.
        let num = Dual::new(FRAC_PI_6, 2.0);
        assert_equal_to_decimal!(num._csc(), Dual::_one()._div(num._sin()), 14);
        assert_equal_to_decimal!(num._sec(), Dual::_one()._div(num._cos()), 14);
        assert_equal_to_decimal!(num._cot(), num._cos()._div(num._sin()), 14);
        assert_equal_to_decimal!(num._sind(), num._to_radians()._sin(), 14);
        assert_equal_to_decimal!(num._acot(), num._recip()._atan(), 14);
    }

    #[test]
    fn test_to_degrees_and_to_radians() {
        assert_equal_to_decimal!(
            Dual::new(180.0, 2.0)._to_radians(),
            Dual::new(PI, PI / 90.0),
            15
        );
        assert_equal_to_decimal!(
            Dual::new(PI, 2.0)._to_degrees(),
            Dual::new(180.0, 360.0 / PI),
            14
        );
    }
}
