//! Implements [`linalg_traits::real_field::RealFieldBase`] for [`crate::HyperDual`].

#![allow(clippy::used_underscore_items)]

use crate::automatic_differentiation::hyper_dual::hyper_dual::HyperDual;
use linalg_traits::real_field::RealFieldBase;
use linalg_traits::verify_trait_implemented;
use std::f64::consts::{LN_2, LN_10};
use std::num::FpCategory;

const _: bool = verify_trait_implemented!(HyperDual: RealFieldBase);

impl RealFieldBase for HyperDual {
    #[inline]
    fn _from_f64(value: f64) -> Self {
        Self::from_real(value)
    }

    #[inline]
    fn _to_f64(&self) -> f64 {
        self.a
    }

    #[inline]
    fn _neg(self) -> Self {
        HyperDual::new(-self.a, -self.b, -self.c, -self.d)
    }

    #[inline]
    fn _add(self, rhs: Self) -> Self {
        HyperDual::new(
            self.a + rhs.a,
            self.b + rhs.b,
            self.c + rhs.c,
            self.d + rhs.d,
        )
    }

    #[inline]
    fn _sub(self, rhs: Self) -> Self {
        HyperDual::new(
            self.a - rhs.a,
            self.b - rhs.b,
            self.c - rhs.c,
            self.d - rhs.d,
        )
    }

    #[inline]
    fn _mul(self, rhs: Self) -> Self {
        HyperDual::new(
            self.a * rhs.a,
            self.b * rhs.a + self.a * rhs.b,
            self.c * rhs.a + self.a * rhs.c,
            self.d * rhs.a + self.b * rhs.c + self.c * rhs.b + self.a * rhs.d,
        )
    }

    #[inline]
    fn _div(self, rhs: Self) -> Self {
        self.bivariate_map(
            rhs,
            |x, y| x / y,
            |_, y| 1.0 / y,
            |x, y| -x / y._powi(2),
            |_, _| 0.0,
            |_, y| -1.0 / y._powi(2),
            |x, y| 2.0 * x / y._powi(3),
        )
    }

    #[inline]
    fn _rem(self, rhs: Self) -> Self {
        let q = (self.a / rhs.a)._trunc();
        HyperDual::new(
            self.a % rhs.a,
            self.b - q * rhs.b,
            self.c - q * rhs.c,
            self.d - q * rhs.d,
        )
    }

    #[inline]
    fn _eq(self, rhs: Self) -> bool {
        self.a == rhs.a && self.b == rhs.b && self.c == rhs.c && self.d == rhs.d
    }

    #[inline]
    fn _partial_cmp(self, rhs: Self) -> Option<std::cmp::Ordering> {
        // Only the real part is compared. This is primarily to support numerical methods where we
        // want to check convergence on the actual function evaluation, and NOT its derivatives.
        self.a.partial_cmp(&rhs.a)
    }

    #[inline]
    fn _zero() -> Self {
        HyperDual::new(0.0, 0.0, 0.0, 0.0)
    }

    #[inline]
    fn _one() -> Self {
        HyperDual::new(1.0, 0.0, 0.0, 0.0)
    }

    #[inline]
    fn _abs(&self) -> Self {
        let sign = self.a.signum();
        HyperDual::new(self.a._abs(), self.b * sign, self.c * sign, self.d * sign)
    }

    #[inline]
    fn _hypot(self, other: Self) -> Self {
        self.bivariate_map(
            other,
            |x, y| (x._powi(2) + y._powi(2))._sqrt(),
            |x, y| {
                let r = (x._powi(2) + y._powi(2))._sqrt();
                x / r
            },
            |x, y| {
                let r = (x._powi(2) + y._powi(2))._sqrt();
                y / r
            },
            |x, y| {
                let r3 = (x._powi(2) + y._powi(2))._powf(1.5);
                y._powi(2) / r3
            },
            |x, y| {
                let r3 = (x._powi(2) + y._powi(2))._powf(1.5);
                -x * y / r3
            },
            |x, y| {
                let r3 = (x._powi(2) + y._powi(2))._powf(1.5);
                x._powi(2) / r3
            },
        )
    }

    #[inline]
    fn _recip(self) -> Self {
        self.univariate_map(f64::recip, |x| -1.0 / x._powi(2), |x| 2.0 / x._powi(3))
    }

    #[inline]
    fn _mul_add(self, a: Self, b: Self) -> Self {
        self._mul(a)._add(b)
    }

    #[inline]
    fn _sqrt(self) -> Self {
        self.univariate_map(
            f64::sqrt,
            |x| 1.0 / (2.0 * x._sqrt()),
            |x| -1.0 / (4.0 * x._powf(1.5)),
        )
    }

    #[inline]
    fn _cbrt(self) -> Self {
        self.univariate_map(
            f64::cbrt,
            |x| 1.0 / (3.0 * x._cbrt()._powi(2)),
            |x| -2.0 / (9.0 * x._cbrt()._powi(5)),
        )
    }

    #[inline]
    fn _powi(self, n: i32) -> Self {
        if n == 0 {
            Self::_one()
        } else {
            self.univariate_map(
                |x| x._powi(n),
                |x| <f64 as From<i32>>::from(n) * x._powi(n - 1),
                |x| <f64 as From<i32>>::from(n) * <f64 as From<i32>>::from(n - 1) * x._powi(n - 2),
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
        self.univariate_map(f64::exp, f64::exp, f64::exp)
    }

    #[inline]
    fn _exp2(self) -> Self {
        self.univariate_map(
            f64::exp2,
            |x| LN_2 * x._exp2(),
            |x| LN_2._powi(2) * x._exp2(),
        )
    }

    #[inline]
    fn _exp_m1(self) -> Self {
        self.univariate_map(f64::exp_m1, f64::exp, f64::exp)
    }

    #[inline]
    fn _ln(self) -> Self {
        self.univariate_map(f64::ln, |x| 1.0 / x, |x| -1.0 / x._powi(2))
    }

    #[inline]
    fn _ln_1p(self) -> Self {
        self.univariate_map(
            f64::ln_1p,
            |x| 1.0 / (1.0 + x),
            |x| -1.0 / (1.0 + x)._powi(2),
        )
    }

    #[inline]
    fn _log(self, base: Self) -> Self {
        self._ln()._div(base._ln())
    }

    #[inline]
    fn _log2(self) -> Self {
        self.univariate_map(
            f64::log2,
            |x| 1.0 / (x * LN_2),
            |x| -1.0 / (x._powi(2) * LN_2),
        )
    }

    #[inline]
    fn _log10(self) -> Self {
        self.univariate_map(
            f64::log10,
            |x| 1.0 / (x * LN_10),
            |x| -1.0 / (x._powi(2) * LN_10),
        )
    }

    #[inline]
    fn _sin(self) -> Self {
        self.univariate_map(f64::sin, f64::cos, |x| -x._sin())
    }

    #[inline]
    fn _cos(self) -> Self {
        self.univariate_map(f64::cos, |x| -x._sin(), |x| -x._cos())
    }

    #[inline]
    fn _sin_cos(self) -> (Self, Self) {
        (self._sin(), self._cos())
    }

    #[inline]
    fn _tan(self) -> Self {
        self.univariate_map(
            f64::tan,
            |x| 1.0 / x._cos()._powi(2),
            |x| 2.0 * x._tan() / x._cos()._powi(2),
        )
    }

    #[inline]
    fn _asin(self) -> Self {
        self.univariate_map(
            f64::asin,
            |x| 1.0 / (1.0 - x._powi(2))._sqrt(),
            |x| x / (1.0 - x._powi(2))._powf(1.5),
        )
    }

    #[inline]
    fn _acos(self) -> Self {
        self.univariate_map(
            f64::acos,
            |x| -1.0 / (1.0 - x._powi(2))._sqrt(),
            |x| -x / (1.0 - x._powi(2))._powf(1.5),
        )
    }

    #[inline]
    fn _atan(self) -> Self {
        self.univariate_map(
            f64::atan,
            |x| 1.0 / (1.0 + x._powi(2)),
            |x| -2.0 * x / (1.0 + x._powi(2))._powi(2),
        )
    }

    #[inline]
    fn _atan2(self, other: Self) -> Self {
        self.bivariate_map(
            other,
            f64::atan2,
            |x, y| y / (x._powi(2) + y._powi(2)),
            |x, y| -x / (x._powi(2) + y._powi(2)),
            |x, y| -2.0 * x * y / (x._powi(2) + y._powi(2))._powi(2),
            |x, y| (x._powi(2) - y._powi(2)) / (x._powi(2) + y._powi(2))._powi(2),
            |x, y| 2.0 * x * y / (x._powi(2) + y._powi(2))._powi(2),
        )
    }

    #[inline]
    fn _sinh(self) -> Self {
        self.univariate_map(f64::sinh, f64::cosh, f64::sinh)
    }

    #[inline]
    fn _cosh(self) -> Self {
        self.univariate_map(f64::cosh, f64::sinh, f64::cosh)
    }

    #[inline]
    fn _tanh(self) -> Self {
        self.univariate_map(
            f64::tanh,
            |x| 1.0 - x._tanh()._powi(2),
            |x| -2.0 * x._tanh() * (1.0 - x._tanh()._powi(2)),
        )
    }

    #[inline]
    fn _asinh(self) -> Self {
        self.univariate_map(
            f64::asinh,
            |x| 1.0 / (x._powi(2) + 1.0)._sqrt(),
            |x| -x / (x._powi(2) + 1.0)._powf(1.5),
        )
    }

    #[inline]
    fn _acosh(self) -> Self {
        self.univariate_map(
            f64::acosh,
            |x| 1.0 / (x._powi(2) - 1.0)._sqrt(),
            |x| -x / (x._powi(2) - 1.0)._powf(1.5),
        )
    }

    #[inline]
    fn _atanh(self) -> Self {
        self.univariate_map(
            f64::atanh,
            |x| 1.0 / (1.0 - x._powi(2)),
            |x| 2.0 * x / (1.0 - x._powi(2))._powi(2),
        )
    }

    #[inline]
    fn _floor(self) -> Self {
        HyperDual::from_real(self.a._floor())
    }

    #[inline]
    fn _ceil(self) -> Self {
        HyperDual::from_real(self.a._ceil())
    }

    #[inline]
    fn _round(self) -> Self {
        HyperDual::from_real(self.a._round())
    }

    #[inline]
    fn _trunc(self) -> Self {
        HyperDual::from_real(self.a._trunc())
    }

    #[inline]
    fn _fract(self) -> Self {
        HyperDual::new(self.a._fract(), self.b, self.c, self.d)
    }

    #[inline]
    fn _copysign(self, sign: Self) -> Self {
        if self.a._is_sign_negative() == sign.a._is_sign_negative() {
            self
        } else {
            self._neg()
        }
    }

    #[inline]
    fn _min(self, other: Self) -> Self {
        if self.a < other.a { self } else { other }
    }

    #[inline]
    fn _max(self, other: Self) -> Self {
        if self.a > other.a { self } else { other }
    }

    #[inline]
    fn _clamp(self, min: Self, max: Self) -> Self {
        if self.a < min.a {
            min
        } else if self.a > max.a {
            max
        } else {
            self
        }
    }

    #[inline]
    fn _is_nan(self) -> bool {
        self.a._is_nan()
    }

    #[inline]
    fn _is_infinite(self) -> bool {
        self.a._is_infinite()
    }

    #[inline]
    fn _is_finite(&self) -> bool {
        self.a._is_finite()
    }

    #[inline]
    fn _is_subnormal(self) -> bool {
        self.a.is_subnormal()
    }

    #[inline]
    fn _is_normal(self) -> bool {
        self.a._is_normal()
    }

    #[inline]
    fn _classify(self) -> FpCategory {
        self.a._classify()
    }

    #[inline]
    fn _is_sign_positive(&self) -> bool {
        self.a._is_sign_positive()
    }

    #[inline]
    fn _is_sign_negative(&self) -> bool {
        self.a._is_sign_negative()
    }

    #[inline]
    fn _next_up(self) -> Self {
        HyperDual::new(self.a.next_up(), self.b, self.c, self.d)
    }

    #[inline]
    fn _next_down(self) -> Self {
        HyperDual::new(self.a.next_down(), self.b, self.c, self.d)
    }

    #[inline]
    fn _epsilon() -> Self {
        HyperDual::from_real(f64::EPSILON)
    }

    #[inline]
    fn _bits() -> usize {
        64
    }

    #[inline]
    fn _min_positive() -> Self {
        HyperDual::from_real(f64::MIN_POSITIVE)
    }

    #[inline]
    fn _max_positive() -> Self {
        HyperDual::from_real(f64::MAX)
    }

    #[inline]
    fn _min_value() -> Option<Self> {
        Some(HyperDual::from_real(f64::MIN))
    }

    #[inline]
    fn _max_value() -> Option<Self> {
        Some(HyperDual::from_real(f64::MAX))
    }

    #[inline]
    fn _nan() -> Self {
        HyperDual::new(f64::NAN, f64::NAN, f64::NAN, f64::NAN)
    }

    #[inline]
    fn _infinity() -> Self {
        HyperDual::new(f64::INFINITY, f64::INFINITY, f64::INFINITY, f64::INFINITY)
    }

    #[inline]
    fn _as_slice(&self) -> &[f64] {
        // SAFETY: `HyperDual` is `#[repr(C)]` with four contiguous `f64` fields (`a`, `b`, `c`, and
        // `d`), so reinterpreting a reference to `self` as a four-element `f64` slice is valid.
        unsafe { std::slice::from_raw_parts(std::ptr::from_ref(self).cast::<f64>(), 4) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::assert_hyper_dual_close;
    use std::f64::consts::{E, FRAC_PI_4, FRAC_PI_6};

    #[test]
    fn test_partial_ord() {
        assert!(HyperDual::new(1.0, 2.0, 3.0, 4.0) < HyperDual::new(3.0, 4.0, 5.0, 6.0));
        assert!(HyperDual::new(3.0, 4.0, 5.0, 6.0) > HyperDual::new(1.0, 2.0, 3.0, 4.0));
        assert!(HyperDual::new(0.0, 2.0, 3.0, 4.0) <= HyperDual::new(1.0, 2.0, 3.0, 4.0));
        assert!(HyperDual::new(2.0, 2.0, 3.0, 4.0) >= HyperDual::new(1.0, 2.0, 3.0, 4.0));
    }

    #[test]
    fn test_neg() {
        assert_eq!(
            -HyperDual::new(1.0, 2.0, 3.0, 4.0),
            HyperDual::new(-1.0, -2.0, -3.0, -4.0)
        );
    }

    #[test]
    fn test_as_slice() {
        assert_eq!(
            HyperDual::new(1.0, 2.0, 3.0, 4.0)._as_slice(),
            &[1.0, 2.0, 3.0, 4.0]
        );
    }

    #[test]
    fn test_nan() {
        let num = HyperDual::_nan();
        assert!(num.get_a()._is_nan());
        assert!(num.get_b()._is_nan());
        assert!(num.get_c()._is_nan());
        assert!(num.get_d()._is_nan());
    }

    #[test]
    fn test_infinity() {
        let num = HyperDual::_infinity();
        assert!(num.get_a()._is_infinite() & (num.get_a() > 0.0));
        assert!(num.get_d()._is_infinite() & (num.get_d() > 0.0));
    }

    #[test]
    fn test_is_nan() {
        assert!(HyperDual::_nan()._is_nan());
        assert!(HyperDual::from_real(f64::NAN)._is_nan());
        assert!(!HyperDual::new(0.0, f64::NAN, f64::NAN, f64::NAN)._is_nan());
    }

    #[test]
    fn test_is_infinite() {
        assert!(HyperDual::_infinity()._is_infinite());
        assert!(!HyperDual::new(0.0, f64::INFINITY, f64::INFINITY, f64::INFINITY)._is_infinite());
    }

    #[test]
    fn test_is_finite() {
        assert!(!HyperDual::_infinity()._is_finite());
        assert!(HyperDual::new(0.0, f64::INFINITY, f64::INFINITY, f64::INFINITY)._is_finite());
    }

    #[test]
    fn test_is_normal() {
        assert!(HyperDual::new(1.0, f64::NAN, f64::NAN, f64::NAN)._is_normal());
        assert!(!HyperDual::new(0.0, 1.0, 1.0, 1.0)._is_normal());
    }

    #[test]
    fn test_classify() {
        assert_eq!(
            HyperDual::new(1.0, f64::NAN, f64::NAN, f64::NAN)._classify(),
            FpCategory::Normal
        );
        assert_eq!(
            HyperDual::new(0.0, 1.0, 1.0, 1.0)._classify(),
            FpCategory::Zero
        );
    }

    #[test]
    fn test_floor() {
        assert_eq!(
            HyperDual::new(2.7, 2.7, -2.7, 5.0)._floor(),
            HyperDual::from_real(2.0)
        );
    }

    #[test]
    fn test_ceil() {
        assert_eq!(
            HyperDual::new(2.7, 2.7, -2.7, 5.0)._ceil(),
            HyperDual::from_real(3.0)
        );
    }

    #[test]
    fn test_round() {
        assert_eq!(
            HyperDual::new(2.7, 2.7, -2.7, 5.0)._round(),
            HyperDual::from_real(3.0)
        );
    }

    #[test]
    fn test_trunc() {
        assert_eq!(
            HyperDual::new(2.7, 2.7, -2.7, 5.0)._trunc(),
            HyperDual::from_real(2.0)
        );
    }

    #[test]
    fn test_fract() {
        assert_eq!(
            HyperDual::new(2.5, 2.5, 3.5, 4.5)._fract(),
            HyperDual::new(0.5, 2.5, 3.5, 4.5)
        );
    }

    #[test]
    fn test_abs() {
        assert_eq!(
            HyperDual::new(-1.0, -2.0, -3.0, -4.0)._abs(),
            HyperDual::new(1.0, 2.0, 3.0, 4.0)
        );
    }

    #[test]
    fn test_copysign() {
        assert_eq!(
            HyperDual::new(1.0, 2.0, 3.0, 4.0)._copysign(HyperDual::new(-1.0, 0.0, 0.0, 0.0)),
            HyperDual::new(-1.0, -2.0, -3.0, -4.0)
        );
        assert_eq!(
            HyperDual::new(1.0, 2.0, 3.0, 4.0)._copysign(HyperDual::new(1.0, 0.0, 0.0, 0.0)),
            HyperDual::new(1.0, 2.0, 3.0, 4.0)
        );
    }

    #[test]
    fn test_is_sign_positive() {
        assert!(HyperDual::new(2.0, -4.0, 7.0, -8.0)._is_sign_positive());
        assert!(!HyperDual::new(-2.0, 4.0, -7.0, 8.0)._is_sign_positive());
    }

    #[test]
    fn test_is_sign_negative() {
        assert!(HyperDual::new(-2.0, 4.0, -7.0, 8.0)._is_sign_negative());
        assert!(!HyperDual::new(2.0, -4.0, 7.0, -8.0)._is_sign_negative());
    }

    #[test]
    fn test_mul_add() {
        let a = HyperDual::new(1.0, 3.0, -2.0, 4.0);
        let b = HyperDual::new(-2.0, 5.0, 7.0, -3.0);
        let c = HyperDual::new(10.0, -4.0, 6.0, 8.0);
        assert_eq!(c._mul_add(a, b), (c * a) + b);
    }

    #[test]
    fn test_recip() {
        assert_hyper_dual_close(
            HyperDual::new(2.0, -5.0, 7.0, 11.0)._recip(),
            HyperDual::new(0.5, 1.25, -1.75, -11.5),
            14,
        );
    }

    #[test]
    fn test_powi() {
        let x = HyperDual::new(2.0, -5.0, 7.0, 11.0);
        assert_eq!(x._powi(0), HyperDual::new(1.0, 0.0, 0.0, 0.0));
        assert_eq!(x._powi(1), x);
        assert_hyper_dual_close(x._powi(2), HyperDual::new(4.0, -20.0, 28.0, -26.0), 14);
    }

    #[test]
    fn test_powf() {
        let x = HyperDual::new(2.0, -5.0, 7.0, 11.0);
        assert_hyper_dual_close(x._powf(HyperDual::from_real(2.0)), x._powi(2), 13);
        assert_hyper_dual_close(x._powf(HyperDual::from_real(0.5)), x._sqrt(), 13);
    }

    #[test]
    fn test_sqrt() {
        assert_hyper_dual_close(
            HyperDual::new(4.0, 25.0, -3.0, 7.0)._sqrt(),
            HyperDual::new(2.0, 6.25, -0.75, 4.09375),
            14,
        );
    }

    #[test]
    fn test_cbrt() {
        assert_hyper_dual_close(
            HyperDual::new(8.0, 27.0, -4.0, 5.0)._cbrt(),
            HyperDual::new(2.0, 2.25, -1.0 / 3.0, 7.0 / 6.0),
            13,
        );
    }

    #[test]
    fn test_exp() {
        assert_hyper_dual_close(
            HyperDual::new(2.0, -3.0, 5.0, 7.0)._exp(),
            HyperDual::new(
                7.38905609893065,
                -22.16716829679195,
                36.945280494653254,
                -59.11244879144519,
            ),
            14,
        );
    }

    #[test]
    fn test_exp2() {
        assert_hyper_dual_close(
            HyperDual::new(2.0, -3.0, 5.0, 7.0)._exp2(),
            HyperDual::new(
                4.0,
                -8.317766166719343,
                13.862943611198906,
                -9.419059779413612,
            ),
            14,
        );
    }

    #[test]
    fn test_exp_m1() {
        let x = HyperDual::new(3.0, 5.0, -2.0, 4.0);
        assert_hyper_dual_close(x._exp_m1(), x._exp() - HyperDual::_one(), 14);
    }

    #[test]
    fn test_ln() {
        assert_hyper_dual_close(
            HyperDual::new(5.0, 8.0, -3.0, 7.0)._ln(),
            HyperDual::new(1.6094379124341003, 1.6, -0.6, 2.36),
            13,
        );
    }

    #[test]
    fn test_ln_1p() {
        let x = HyperDual::new(3.0, 5.0, -2.0, 4.0);
        assert_hyper_dual_close(x._ln_1p(), (x + HyperDual::_one())._ln(), 14);
    }

    #[test]
    fn test_log() {
        let x = HyperDual::new(5.0, 8.0, -3.0, 7.0);
        let base = HyperDual::new(4.5, 2.0, 6.0, -5.0);
        assert_hyper_dual_close(x._log(base), x._ln() / base._ln(), 14);
    }

    #[test]
    fn test_log2() {
        assert_hyper_dual_close(
            HyperDual::new(5.0, 8.0, -3.0, 7.0)._log2(),
            HyperDual::new(
                2.321928094887362,
                2.3083120654223412,
                -0.865617024533378,
                3.4047602964979533,
            ),
            14,
        );
    }

    #[test]
    fn test_log10() {
        assert_hyper_dual_close(
            HyperDual::new(5.0, 8.0, -3.0, 7.0)._log10(),
            HyperDual::new(
                0.6989700043360189,
                0.6948711710452028,
                -0.26057668914195103,
                1.0249349772916743,
            ),
            14,
        );
    }

    #[test]
    fn test_max() {
        assert_eq!(
            HyperDual::new(1.0, 2.0, 3.0, 4.0)._max(HyperDual::new(3.0, 4.0, 5.0, 6.0)),
            HyperDual::new(3.0, 4.0, 5.0, 6.0)
        );
    }

    #[test]
    fn test_min() {
        assert_eq!(
            HyperDual::new(1.0, 2.0, 3.0, 4.0)._min(HyperDual::new(3.0, 4.0, 5.0, 6.0)),
            HyperDual::new(1.0, 2.0, 3.0, 4.0)
        );
    }

    #[test]
    fn test_clamp() {
        assert_eq!(
            HyperDual::new(5.0, 1.0, 0.0, 0.0)
                ._clamp(HyperDual::from_real(0.0), HyperDual::from_real(3.0)),
            HyperDual::from_real(3.0)
        );
    }

    #[test]
    fn test_hypot() {
        let x = HyperDual::new(1.0, 2.0, -3.0, 5.0);
        let y = HyperDual::new(3.0, 4.0, 7.0, -11.0);
        assert_hyper_dual_close(
            x._hypot(y),
            HyperDual::new(
                3.1622776601683795,
                4.427188724235731,
                5.692099788303082,
                -9.866306299725345,
            ),
            14,
        );
        let norm = (x._powi(2) + y._powi(2))._sqrt();
        assert_hyper_dual_close(x._hypot(y), norm, 14);
    }

    #[test]
    fn test_sin() {
        assert_hyper_dual_close(
            HyperDual::new(FRAC_PI_6, 2.0, -3.0, 5.0)._sin(),
            HyperDual::new(
                0.5,
                1.7320508075688772,
                -2.598076211353316,
                7.330127018922193,
            ),
            14,
        );
    }

    #[test]
    fn test_cos() {
        assert_hyper_dual_close(
            HyperDual::new(FRAC_PI_6, 2.0, -3.0, 5.0)._cos(),
            HyperDual::new(0.8660254037844386, -1.0, 1.5, 2.696152422706632),
            14,
        );
    }

    #[test]
    fn test_sin_cos() {
        let x = HyperDual::new(FRAC_PI_6, 2.0, -3.0, 5.0);
        let (sin, cos) = x._sin_cos();
        assert_hyper_dual_close(sin, x._sin(), 14);
        assert_hyper_dual_close(cos, x._cos(), 14);
    }

    #[test]
    fn test_tan() {
        assert_hyper_dual_close(
            HyperDual::new(FRAC_PI_6, 2.0, -3.0, 5.0)._tan(),
            HyperDual::new(
                0.5773502691896258,
                2.666666666666667,
                -4.0,
                -2.5709376403673474,
            ),
            14,
        );
    }

    #[test]
    fn test_asin() {
        assert_hyper_dual_close(
            HyperDual::new(0.5, 3.0, -2.0, 7.0)._asin(),
            HyperDual::new(
                0.5235987755982988,
                3.4641016151377553,
                -2.3094010767585034,
                3.4641016151377553,
            ),
            14,
        );
    }

    #[test]
    fn test_asin_out_of_domain_nan() {
        assert!(
            HyperDual::new(1.0001, 2.0, -3.0, 4.0)
                ._asin()
                .get_a()
                ._is_nan()
        );
    }

    #[test]
    fn test_acos() {
        assert_hyper_dual_close(
            HyperDual::new(3.0_f64._sqrt() / 2.0, 3.0, -2.0, 7.0)._acos(),
            HyperDual::new(FRAC_PI_6, -6.0, 4.0, 27.56921938165303),
            13,
        );
    }

    #[test]
    fn test_atan() {
        assert_hyper_dual_close(
            HyperDual::new(1.0, 3.0, -2.0, 7.0)._atan(),
            HyperDual::new(FRAC_PI_4, 1.5, -1.0, 6.5),
            14,
        );
    }

    #[test]
    fn test_atan2() {
        assert_hyper_dual_close(
            HyperDual::new(-3.0, 5.0, 7.0, -6.0)._atan2(HyperDual::new(3.0, 2.0, -1.0, 4.0)),
            HyperDual::new(-FRAC_PI_4, 7.0 / 6.0, 1.0, 31.0 / 18.0),
            14,
        );
    }

    #[test]
    fn test_sinh() {
        assert_hyper_dual_close(
            HyperDual::new(1.0, 2.0, 1.5, -0.75)._sinh(),
            HyperDual::new(
                ((E * E) - 1.0) / (2.0 * E),
                ((E * E) + 1.0) / E,
                2.3146209522228656,
                2.3682931048199714,
            ),
            15,
        );
    }

    #[test]
    fn test_cosh() {
        assert_hyper_dual_close(
            HyperDual::new(1.0, 2.0, -3.0, 4.0)._cosh(),
            HyperDual::new(
                1.5430806348152437,
                2.3504023872876028,
                -3.525603580931404,
                -4.557679034316257,
            ),
            14,
        );
    }

    #[test]
    fn test_tanh() {
        assert_hyper_dual_close(
            HyperDual::new(1.0, 2.0, -3.0, 4.0)._tanh(),
            HyperDual::new(
                0.7615941559557649,
                0.8399486832280523,
                -1.2599230248420783,
                5.518097417151452,
            ),
            14,
        );
    }

    #[test]
    fn test_asinh() {
        assert_hyper_dual_close(
            HyperDual::new(1.0, 2.0, -3.0, 4.0)._sinh()._asinh(),
            HyperDual::new(1.0, 2.0, -3.0, 4.0),
            13,
        );
    }

    #[test]
    fn test_acosh() {
        assert_hyper_dual_close(
            HyperDual::new(1.0, 2.0, -3.0, 4.0)._cosh()._acosh(),
            HyperDual::new(1.0, 2.0, -3.0, 4.0),
            13,
        );
    }

    #[test]
    fn test_atanh() {
        assert_hyper_dual_close(
            HyperDual::new(0.5, 2.0, -3.0, 4.0)._tanh()._atanh(),
            HyperDual::new(0.5, 2.0, -3.0, 4.0),
            12,
        );
    }

    #[test]
    fn test_extended_trig_from_real_field_defaults() {
        let num = HyperDual::new(FRAC_PI_6, 2.0, -3.0, 5.0);
        assert_hyper_dual_close(num._csc(), HyperDual::_one()._div(num._sin()), 12);
        assert_hyper_dual_close(num._sec(), HyperDual::_one()._div(num._cos()), 12);
        assert_hyper_dual_close(num._cot(), num._cos()._div(num._sin()), 12);
        assert_hyper_dual_close(num._sind(), num._to_radians()._sin(), 12);
    }
}
