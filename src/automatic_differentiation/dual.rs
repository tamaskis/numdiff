use num_traits::{Float, Num, NumCast, One, ToPrimitive, Zero};
use std::cmp::Ordering;
use std::num::FpCategory;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};

#[cfg(feature = "trig")]
use trig::Trig;

/// TODO: notes on other crates
///     --> num-dual: panics on a lot of methods implemented for num_traits::Float; this is NOT good
///     --> num-dual: depends on nalgebra
///     --> num-dual: fairly intrusive (makes you write everything with custom types)
///     --> autodj: fairly intrusive (makes you write everything with custom types)
///     --> autodiff: fairly intrusive (makes you write everything with custom types)
///
/// TODO:
///     --> make sure argument names match original definitions
///     --> put in cascading order of trait implementations
///     --> implement trig trait
///     --> unit test everything
///     --> rename real and dual parts to x and dx?
///     --> check that the derived PartialEq is good; probably just manually implement?
///     --> get rid of all "re" and "im" refs

/// TODO: document.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Dual {
    /// Real part of the dual number.
    real: f64,

    /// Dual part of the dual number.
    dual: f64,
}

impl Dual {
    /// Constructor.
    ///
    /// # Arguments
    ///
    /// * `real` - Real part.
    /// * `dual` - Dual part.
    ///
    /// # Returns
    ///
    /// Dual number.
    pub fn new(real: f64, dual: f64) -> Dual {
        Dual { real, dual }
    }

    /// TODO
    pub fn get_real(self) -> f64 {
        self.real
    }

    /// TODO
    pub fn get_dual(self) -> f64 {
        self.dual
    }
}

// TODO: note one restriction: can't do assignment operations on an f64
// TODO: unit test everything

// --------------------------------
// Implementing num_traits::NumOps.
// --------------------------------
// https://docs.rs/num-traits/latest/num_traits/trait.NumOps.html
//
// pub trait NumOps<Rhs = Self, Output = Self>:
//     Add<Rhs, Output = Output>
//     + Sub<Rhs, Output = Output>
//     + Mul<Rhs, Output = Output>
//     + Div<Rhs, Output = Output>
//     + Rem<Rhs, Output = Output>
// {
// }

// Dual + Dual.
impl Add for Dual {
    type Output = Dual;
    fn add(self, other: Dual) -> Dual {
        Dual::new(self.real + other.real, self.dual + other.dual)
    }
}

// Dual - Dual.
impl Sub for Dual {
    type Output = Dual;
    fn sub(self, other: Dual) -> Dual {
        Dual::new(self.real - other.real, self.dual - other.dual)
    }
}

// Dual * Dual.
impl Mul for Dual {
    type Output = Dual;
    fn mul(self, other: Dual) -> Dual {
        Dual::new(
            self.real * other.real,
            self.dual * other.real + self.real * other.dual,
        )
    }
}

// Dual / Dual.
impl Div for Dual {
    type Output = Dual;
    fn div(self, other: Dual) -> Dual {
        Dual::new(
            self.real / other.real,
            (self.dual * other.real - self.real * other.dual) / other.real.powi(2),
        )
    }
}

// Remainder of Dual / Dual.
impl Rem for Dual {
    type Output = Self;
    fn rem(self, rhs: Self) -> Self::Output {
        Dual::new(
            self.real % rhs.real,
            self.dual - (self.real / rhs.real).floor() * rhs.dual,
        )
    }
}

// -----------------------------
// Implementing num_traits::Num.
// -----------------------------
// https://docs.rs/num-traits/latest/num_traits/trait.Num.html
//
// pub trait Num: PartialEq + Zero + One + NumOps {
//     type FromStrRadixErr;
//
//     // Required method
//     fn from_str_radix(
//         str: &str,
//         radix: u32
//     ) -> Result<Self, Self::FromStrRadixErr>;
// }

impl Zero for Dual {
    fn zero() -> Self {
        Dual::new(0.0, 0.0)
    }
    fn is_zero(&self) -> bool {
        self.real.is_zero() && self.dual.is_zero()
    }
}

impl One for Dual {
    fn one() -> Self {
        Dual::new(1.0, 0.0)
    }
}

impl Num for Dual {
    type FromStrRadixErr = <f64 as Num>::FromStrRadixErr;
    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        f64::from_str_radix(str, radix).map(|re| Dual::new(re, 0.0))
    }
}

// -------------------------------------
// Implementing num_traits::ToPrimitive.
// -------------------------------------
// https://docs.rs/num-traits/latest/num_traits/cast/trait.ToPrimitive.html
//
// pub trait ToPrimitive {
//     // Required methods
//     fn to_i64(&self) -> Option<i64>;
//     fn to_u64(&self) -> Option<u64>;
//
//     // Provided methods
//     ...
//     fn to_f64(&self) -> Option<f64> { ... }
// }

impl ToPrimitive for Dual {
    fn to_i64(&self) -> Option<i64> {
        self.real.to_i64()
    }
    fn to_u64(&self) -> Option<u64> {
        self.real.to_u64()
    }
    fn to_f64(&self) -> Option<f64> {
        Some(self.real)
    }
}

// ---------------------------------
// Implementing num_traits::NumCast.
// ---------------------------------
// https://docs.rs/num-traits/latest/num_traits/cast/trait.NumCast.html
//
// pub trait NumCast: Sized + ToPrimitive {
//     // Required method
//     fn from<T: ToPrimitive>(n: T) -> Option<Self>;
// }

impl NumCast for Dual {
    fn from<T: ToPrimitive>(n: T) -> Option<Self> {
        n.to_f64().map(|re| Dual::new(re, 0.0))
    }
}

// -------------------------------
// Implementing num_traits::Float.
// -------------------------------
// https://docs.rs/num-traits/latest/num_traits/float/trait.Float.html
//
// pub trait Float: Num + Copy + NumCast + PartialOrd + Neg<Output = Self> {
// [+] Show 60 methods
// }

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
        Dual::new(f64::infinity(), f64::zero())
    }

    fn neg_infinity() -> Self {
        Dual::new(f64::neg_infinity(), f64::zero())
    }

    fn neg_zero() -> Self {
        Dual::new(f64::neg_zero(), f64::zero())
    }

    fn min_value() -> Self {
        Dual::new(f64::min_value(), f64::zero())
    }

    fn min_positive_value() -> Self {
        Dual::new(f64::min_positive_value(), f64::zero())
    }

    fn max_value() -> Self {
        Dual::new(f64::max_value(), f64::zero())
    }

    fn is_nan(self) -> bool {
        self.real.is_nan() || self.dual.is_nan()
    }

    fn is_infinite(self) -> bool {
        self.real.is_infinite()
    }

    fn is_finite(self) -> bool {
        self.real.is_finite() && self.dual.is_finite()
    }

    fn is_normal(self) -> bool {
        self.real.is_normal()
    }

    fn classify(self) -> FpCategory {
        self.real.classify()
    }

    fn floor(self) -> Self {
        Dual::new(self.real.floor(), 0.0)
    }

    fn ceil(self) -> Self {
        Dual::new(self.real.ceil(), 0.0)
    }

    fn round(self) -> Self {
        Dual::new(self.real.round(), 0.0)
    }

    fn trunc(self) -> Self {
        Dual::new(self.real.trunc(), 0.0)
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
                (n as f64) * self.dual * self.real.powi(n - 1),
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
        Dual::new(exp2_re, exp2_re * std::f64::consts::LN_2 * self.dual)
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
        Dual::new(
            self.real.ln() / std::f64::consts::LN_2,
            self.dual / (self.real * std::f64::consts::LN_2),
        )
    }

    fn log10(self) -> Self {
        Dual::new(
            self.real.ln() / std::f64::consts::LN_10,
            self.dual / (self.real * std::f64::consts::LN_10),
        )
    }

    fn max(self, other: Self) -> Self {
        let max_re = self.real.max(other.real);
        let max_im = if self.real > other.real {
            self.dual
        } else {
            other.dual
        };
        Dual::new(max_re, max_im)
    }

    fn min(self, other: Self) -> Self {
        let min_re = self.real.min(other.real);
        let min_im = if self.real < other.real {
            self.dual
        } else {
            other.dual
        };
        Dual::new(min_re, min_im)
    }

    #[allow(deprecated)]
    fn abs_sub(self, other: Self) -> Self {
        Dual::new(
            self.real.abs_sub(other.real),
            self.dual - other.dual, // Adjusted to dual semantics TODO
        )
    }

    fn cbrt(self) -> Self {
        let cbrt_re = self.real.cbrt();
        Dual::new(cbrt_re, self.dual / (3.0 * cbrt_re.powi(2)))
    }

    fn hypot(self, other: Self) -> Self {
        let hypot_re = (self.real.powi(2) + other.real.powi(2)).sqrt();
        Dual::new(
            hypot_re,
            (self.real * self.dual + other.real * other.dual) / hypot_re,
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
        Dual::new(self.real.sinh(), self.real.cosh())
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

    fn integer_decode(self) -> (u64, i16, i8) {
        self.real.integer_decode()
    }
}

// -----------------------------------
// Implementing linalg_traits::Scalar.
// -----------------------------------
// https://docs.rs/linalg-traits/latest/linalg_traits/trait.Scalar.html
//
// pub trait Scalar:
//     Float
//     + AddAssign<Self>
//     + SubAssign<Self>
//     + MulAssign<Self>
//     + DivAssign<Self>
//     + RemAssign<Self>
//     + Add<f64, Output = Self>
//     + Sub<f64, Output = Self>
//     + Mul<f64, Output = Self>
//     + Div<f64, Output = Self>
//     + Rem<f64, Output = Self>
//     + AddAssign<f64>
//     + SubAssign<f64>
//     + MulAssign<f64>
//     + DivAssign<f64>
//     + RemAssign<f64>
//     + Debug
//     + 'static { }

impl AddAssign for Dual {
    fn add_assign(&mut self, other: Dual) {
        self.real += other.real;
        self.dual += other.dual;
    }
}

impl SubAssign for Dual {
    fn sub_assign(&mut self, other: Dual) {
        self.real -= other.real;
        self.dual -= other.dual;
    }
}

impl MulAssign for Dual {
    fn mul_assign(&mut self, other: Dual) {
        self.dual = self.dual * other.real + self.real * other.dual;
        self.real *= other.real;
    }
}

impl DivAssign for Dual {
    fn div_assign(&mut self, other: Dual) {
        self.dual = (self.dual * other.real - self.real * other.dual) / other.real.powi(2);
        self.real /= other.real;
    }
}

// TODO unit test
impl RemAssign for Dual {
    fn rem_assign(&mut self, rhs: Self) {
        self.dual -= (self.real / rhs.real).floor() * rhs.dual;
        self.real %= rhs.real;
    }
}

impl Add<f64> for Dual {
    type Output = Dual;
    fn add(self, scalar: f64) -> Dual {
        Dual::new(self.real + scalar, self.dual)
    }
}

impl Sub<f64> for Dual {
    type Output = Dual;
    fn sub(self, scalar: f64) -> Dual {
        Dual::new(self.real - scalar, self.dual)
    }
}

impl Mul<f64> for Dual {
    type Output = Dual;
    fn mul(self, scalar: f64) -> Dual {
        Dual::new(self.real * scalar, self.dual * scalar)
    }
}

impl Div<f64> for Dual {
    type Output = Dual;
    fn div(self, scalar: f64) -> Dual {
        Dual::new(self.real / scalar, self.dual / scalar)
    }
}

impl Rem<f64> for Dual {
    type Output = Dual;
    fn rem(self, rhs: f64) -> Self::Output {
        let re_rem = self.real % rhs;
        let im_rem = if self.real % rhs == 0.0 {
            0.0
        } else {
            self.dual
        };
        Dual::new(re_rem, im_rem)
    }
}

impl AddAssign<f64> for Dual {
    fn add_assign(&mut self, scalar: f64) {
        self.real += scalar;
    }
}

impl SubAssign<f64> for Dual {
    fn sub_assign(&mut self, scalar: f64) {
        self.real -= scalar;
    }
}

impl MulAssign<f64> for Dual {
    fn mul_assign(&mut self, scalar: f64) {
        self.real *= scalar;
        self.dual *= scalar;
    }
}

impl DivAssign<f64> for Dual {
    fn div_assign(&mut self, scalar: f64) {
        self.real /= scalar;
        self.dual /= scalar;
    }
}

impl RemAssign<f64> for Dual {
    fn rem_assign(&mut self, rhs: f64) {
        self.real = self.real % rhs;
        if self.real == 0.0 {
            self.dual = 0.0;
        }
    }
}

// ---------------------------
// Interoperability with f64s. TODO should this be in another section?
// ---------------------------

impl Add<Dual> for f64 {
    type Output = Dual;
    fn add(self, dual: Dual) -> Dual {
        Dual::new(self + dual.real, dual.dual)
    }
}

impl Sub<Dual> for f64 {
    type Output = Dual;
    fn sub(self, dual: Dual) -> Dual {
        Dual::new(self - dual.real, dual.dual)
    }
}

impl Mul<Dual> for f64 {
    type Output = Dual;
    fn mul(self, dual: Dual) -> Dual {
        Dual::new(self * dual.real, self * dual.dual)
    }
}

impl Div<Dual> for f64 {
    type Output = Dual;
    fn div(self, dual: Dual) -> Dual {
        Dual::new(self / dual.real, -self * dual.dual / dual.real.powi(2))
    }
}

// ------------------------
// Implementing trig::Trig.
// ------------------------

#[cfg(feature = "trig")]
impl Trig for Dual {
    fn sin(&self) -> Dual {
        <Dual as Float>::sin(*self)
    }
    fn cos(&self) -> Dual {
        <Dual as Float>::cos(*self)
    }
    fn tan(&self) -> Dual {
        <Dual as Float>::tan(*self)
    }
    fn csc(&self) -> Dual {
        1.0 / self.sin()
    }
    fn sec(&self) -> Dual {
        1.0 / self.cos()
    }
    fn cot(&self) -> Dual {
        1.0 / self.tan()
    }
    fn asin(&self) -> Dual {
        <Dual as Float>::asin(*self)
    }
    fn acos(&self) -> Dual {
        <Dual as Float>::acos(*self)
    }
    fn atan(&self) -> Dual {
        <Dual as Float>::atan(*self)
    }
    fn atan2(&self, other: &Dual) -> Dual {
        <Dual as Float>::atan2(*self, *other)
    }
    fn acsc(&self) -> Dual {
        (Dual::new(1.0, 0.0) / *self).asin()
    }
    fn asec(&self) -> Dual {
        (Dual::new(1.0, 0.0) / *self).acos()
    }
    fn acot(&self) -> Dual {
        (Dual::new(1.0, 0.0) / *self).atan()
    }
    fn deg2rad(&self) -> Dual {
        *self * Dual::new(std::f64::consts::PI / 180.0, 0.0)
    }
    fn rad2deg(&self) -> Dual {
        *self * Dual::new(180.0 / std::f64::consts::PI, 0.0)
    }
    fn sind(&self) -> Dual {
        self.deg2rad().sin()
    }
    fn cosd(&self) -> Dual {
        self.deg2rad().cos()
    }
    fn tand(&self) -> Dual {
        self.deg2rad().tan()
    }
    fn cscd(&self) -> Dual {
        self.deg2rad().csc()
    }
    fn secd(&self) -> Dual {
        self.deg2rad().sec()
    }
    fn cotd(&self) -> Dual {
        self.deg2rad().cot()
    }
    fn asind(&self) -> Dual {
        self.asin().rad2deg()
    }
    fn acosd(&self) -> Dual {
        self.acos().rad2deg()
    }
    fn atand(&self) -> Dual {
        self.atan().rad2deg()
    }
    fn atan2d(&self, other: &Dual) -> Dual {
        self.atan2(other).rad2deg()
    }
    fn acscd(&self) -> Dual {
        self.acsc().rad2deg()
    }
    fn asecd(&self) -> Dual {
        self.asec().rad2deg()
    }
    fn acotd(&self) -> Dual {
        self.acot().rad2deg()
    }
    fn sinh(&self) -> Dual {
        <Dual as Float>::sinh(*self)
    }
    fn cosh(&self) -> Dual {
        <Dual as Float>::cosh(*self)
    }
    fn tanh(&self) -> Dual {
        <Dual as Float>::tanh(*self)
    }
    fn csch(&self) -> Dual {
        1.0 / self.sinh()
    }
    fn sech(&self) -> Dual {
        1.0 / self.cosh()
    }
    fn coth(&self) -> Dual {
        1.0 / self.tanh()
    }
    fn asinh(&self) -> Dual {
        <Dual as Float>::asinh(*self)
    }
    fn acosh(&self) -> Dual {
        <Dual as Float>::acosh(*self)
    }
    fn atanh(&self) -> Dual {
        <Dual as Float>::atanh(*self)
    }
    fn acsch(&self) -> Dual {
        (Dual::new(1.0, 0.0) / *self).asinh()
    }
    fn asech(&self) -> Dual {
        (Dual::new(1.0, 0.0) / *self).acosh()
    }
    fn acoth(&self) -> Dual {
        (Dual::new(1.0, 0.0) / *self).atanh()
    }
}

// --------
// TESTING.
// --------

#[cfg(test)]
mod tests {
    use linalg_traits::Scalar;

    use super::*;

    #[test]
    fn test_add() {
        // Dual + Dual.
        assert_eq!(
            Dual::new(1.0, 2.0) + Dual::new(3.0, 4.0),
            Dual::new(4.0, 6.0)
        );

        // Dual + f64.
        assert_eq!(Dual::new(1.0, 2.0) + 3.0, Dual::new(4.0, 2.0));

        // f64 + Dual.
        assert_eq!(1.0 + Dual::new(2.0, 3.0), Dual::new(3.0, 3.0));
    }

    #[test]
    fn test_add_assign() {
        // Dual += Dual.
        let mut a = Dual::new(1.0, 2.0);
        a += Dual::new(3.0, 4.0);
        assert_eq!(a, Dual::new(4.0, 6.0));

        // Dual += f64.
        let mut b = Dual::new(1.0, 2.0);
        b += 3.0;
        assert_eq!(b, Dual::new(4.0, 2.0));
    }

    #[test]
    fn test_sub() {
        // Dual - Dual.
        assert_eq!(
            Dual::new(1.0, 2.0) - Dual::new(4.0, 3.0),
            Dual::new(-3.0, -1.0)
        );

        // Dual - f64.
        assert_eq!(Dual::new(1.0, 2.0) - 3.0, Dual::new(-2.0, 2.0));

        // f64 - Dual.
        assert_eq!(1.0 - Dual::new(2.0, 3.0), Dual::new(-1.0, 3.0));
    }

    #[test]
    fn test_sub_assign() {
        // Dual -= Dual.
        let mut a = Dual::new(1.0, 2.0);
        a -= Dual::new(4.0, 3.0);
        assert_eq!(a, Dual::new(-3.0, -1.0));

        // Dual -= f64.
        let mut b = Dual::new(1.0, 2.0);
        b -= 3.0;
        assert_eq!(b, Dual::new(-2.0, 2.0));
    }

    #[test]
    fn test_mul() {
        // Dual * Dual.
        assert_eq!(
            Dual::new(1.0, 2.0) * Dual::new(3.0, -4.0),
            Dual::new(3.0, 2.0)
        );

        // Dual * f64.
        assert_eq!(Dual::new(1.0, -2.0) * 3.0, Dual::new(3.0, -6.0));

        // f64 * Dual.
        assert_eq!(5.0 * Dual::new(2.0, -3.0), Dual::new(10.0, -15.0));
    }

    #[test]
    fn test_mul_assign() {
        // Dual *= Dual.
        let mut a = Dual::new(1.0, 2.0);
        a *= Dual::new(3.0, -4.0);
        assert_eq!(a, Dual::new(3.0, 2.0));

        // Dual *= f64.
        let mut b = Dual::new(2.0, -3.0);
        b *= 5.0;
        assert_eq!(b, Dual::new(10.0, -15.0));
    }

    #[test]
    fn test_div() {
        // Dual / Dual.
        assert_eq!(
            Dual::new(1.0, 2.0) / Dual::new(3.0, 4.0),
            Dual::new(1.0 / 3.0, 2.0 / 9.0)
        );

        // Dual / f64.
        assert_eq!(Dual::new(1.0, 2.0) / 4.0, Dual::new(0.25, 0.5));

        // f64 / Dual.
        assert_eq!(5.0 / Dual::new(2.0, -3.0), Dual::new(2.5, 3.75));
    }

    #[test]
    fn test_div_assign() {
        // Dual /= Dual.
        let mut a = Dual::new(1.0, 2.0);
        a /= Dual::new(3.0, 4.0);
        assert_eq!(a, Dual::new(1.0 / 3.0, 2.0 / 9.0));

        // Dual /= f64.
        let mut b = Dual::new(1.0, 2.0);
        b /= 4.0;
        assert_eq!(b, Dual::new(0.25, 0.5));
    }

    #[test]
    fn test_unary_negation() {
        let a = Dual::new(1.0, -2.0);
        let b = -a;
        assert_eq!(b, Dual::new(-1.0, 2.0));
    }

    #[test]
    fn test_zero() {
        // Construction
        assert_eq!(Dual::zero(), Dual::new(0.0, 0.0));

        // Zero-check.
        assert!(Dual::zero().is_zero());
        assert!(Dual::new(0.0, 0.0).is_zero());

        // Dual::zero() * Dual = Dual::zero().
        assert_eq!(Dual::zero() * Dual::new(1.0, 2.0), Dual::zero());
    }

    #[test]
    fn test_one() {
        // Construction.
        assert_eq!(Dual::one(), Dual::new(1.0, 0.0));

        // Dual::one() * Dual = Dual.
        assert_eq!(Dual::one() * Dual::new(1.0, 2.0), Dual::new(1.0, 2.0));

        // Dual::one() * scalar = Dual(scalar, scalar).
        assert_eq!(Dual::one() * 5.0, Dual::new(5.0, 0.0));
    }

    fn add_scalar<S: Scalar>(x: S, y: S) -> S {
        x + y
    }

    #[test]
    fn scalar() {
        assert_eq!(
            add_scalar(Dual::new(5.0, 4.0), Dual::new(3.0, 2.0)),
            Dual::new(8.0, 6.0)
        );
    }
}
