use num_traits::{Float, Num, NumCast, One, ToPrimitive, Zero};
use std::cmp::Ordering;
use std::num::FpCategory;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};

#[cfg(feature = "trig")]
use trig::Trig;

/// First-order dual number.
#[derive(Debug, Clone, Copy)]
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
    ///
    /// # Example
    ///
    /// ```
    /// use numdiff::Dual;
    ///
    /// let num = Dual::new(1.0, 2.0);
    /// ```
    pub fn new(real: f64, dual: f64) -> Dual {
        Dual { real, dual }
    }

    /// Get the real part of the dual number.
    ///
    /// # Returns
    ///
    /// Real part of the dual number.
    ///
    /// # Example
    ///
    /// ```
    /// use numdiff::Dual;
    ///
    /// let num = Dual::new(1.0, 2.0);
    /// assert_eq!(num.get_real(), 1.0);
    /// ```
    pub fn get_real(self) -> f64 {
        self.real
    }

    /// Get the dual part of the dual number.
    ///
    /// # Returns
    ///
    /// Dual part of the dual number.
    ///
    /// # Example
    ///
    /// ```
    /// use numdiff::Dual;
    ///
    /// let num = Dual::new(1.0, 2.0);
    /// assert_eq!(num.get_dual(), 2.0);
    /// ```
    pub fn get_dual(self) -> f64 {
        self.dual
    }
}

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
    fn add(self, rhs: Dual) -> Dual {
        Dual::new(self.real + rhs.real, self.dual + rhs.dual)
    }
}

// Dual - Dual.
impl Sub for Dual {
    type Output = Dual;
    fn sub(self, rhs: Dual) -> Dual {
        Dual::new(self.real - rhs.real, self.dual - rhs.dual)
    }
}

// Dual * Dual.
impl Mul for Dual {
    type Output = Dual;
    fn mul(self, rhs: Dual) -> Dual {
        Dual::new(
            self.real * rhs.real,
            self.dual * rhs.real + self.real * rhs.dual,
        )
    }
}

// Dual / Dual.
impl Div for Dual {
    type Output = Dual;
    fn div(self, rhs: Dual) -> Dual {
        Dual::new(
            self.real / rhs.real,
            (self.dual * rhs.real - self.real * rhs.dual) / rhs.real.powi(2),
        )
    }
}

// Remainder of Dual / Dual.
impl Rem for Dual {
    type Output = Self;
    fn rem(self, rhs: Self) -> Self::Output {
        Dual::new(
            self.real % rhs.real,
            self.dual - (self.real / rhs.real).trunc() * rhs.dual,
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

// https://iel.ucdavis.edu/publication/journal/j_EC1.pdf (p. 11)
impl PartialEq for Dual {
    fn eq(&self, other: &Self) -> bool {
        self.real == other.real && self.dual == other.dual
    }
}

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
        if self.real > other.real {
            self
        } else {
            other
        }
    }

    fn min(self, other: Self) -> Self {
        if self.real < other.real {
            self
        } else {
            other
        }
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

// Dual += Dual.
impl AddAssign for Dual {
    fn add_assign(&mut self, other: Dual) {
        self.real += other.real;
        self.dual += other.dual;
    }
}

// Dual -= Dual.
impl SubAssign for Dual {
    fn sub_assign(&mut self, other: Dual) {
        self.real -= other.real;
        self.dual -= other.dual;
    }
}

// Dual *= Dual.
impl MulAssign for Dual {
    fn mul_assign(&mut self, other: Dual) {
        self.dual = self.dual * other.real + self.real * other.dual;
        self.real *= other.real;
    }
}

// Dual /= Dual.
impl DivAssign for Dual {
    fn div_assign(&mut self, other: Dual) {
        self.dual = (self.dual * other.real - self.real * other.dual) / other.real.powi(2);
        self.real /= other.real;
    }
}

// Dual %= Dual.
impl RemAssign for Dual {
    fn rem_assign(&mut self, rhs: Self) {
        self.dual -= (self.real / rhs.real).floor() * rhs.dual;
        self.real %= rhs.real;
    }
}

// Dual + f64.
impl Add<f64> for Dual {
    type Output = Dual;
    fn add(self, rhs: f64) -> Dual {
        Dual::new(self.real + rhs, self.dual)
    }
}

// Dual - f64.
impl Sub<f64> for Dual {
    type Output = Dual;
    fn sub(self, rhs: f64) -> Dual {
        Dual::new(self.real - rhs, self.dual)
    }
}

// Dual * f64.
impl Mul<f64> for Dual {
    type Output = Dual;
    fn mul(self, rhs: f64) -> Dual {
        Dual::new(self.real * rhs, self.dual * rhs)
    }
}

// Dual / f64.
impl Div<f64> for Dual {
    type Output = Dual;
    fn div(self, rhs: f64) -> Dual {
        Dual::new(self.real / rhs, self.dual / rhs)
    }
}

// Dual % f64.
impl Rem<f64> for Dual {
    type Output = Dual;
    fn rem(self, rhs: f64) -> Self::Output {
        let rem_real = self.real % rhs;
        let rem_dual = if self.real % rhs == 0.0 {
            0.0
        } else {
            self.dual
        };
        Dual::new(rem_real, rem_dual)
    }
}

// Dual += f64.
impl AddAssign<f64> for Dual {
    fn add_assign(&mut self, rhs: f64) {
        self.real += rhs;
    }
}

// Dual -= f64.
impl SubAssign<f64> for Dual {
    fn sub_assign(&mut self, rhs: f64) {
        self.real -= rhs;
    }
}

// Dual *= f64.
impl MulAssign<f64> for Dual {
    fn mul_assign(&mut self, rhs: f64) {
        self.real *= rhs;
        self.dual *= rhs;
    }
}

// Dual /= f64.
impl DivAssign<f64> for Dual {
    fn div_assign(&mut self, rhs: f64) {
        self.real /= rhs;
        self.dual /= rhs;
    }
}

// Dual %= f64.
impl RemAssign<f64> for Dual {
    fn rem_assign(&mut self, rhs: f64) {
        self.real = self.real % rhs;
        if self.real == 0.0 {
            self.dual = 0.0;
        }
    }
}

// ---------------------------
// Interoperability with f64s.
// ---------------------------

// f64 + Dual.
impl Add<Dual> for f64 {
    type Output = Dual;
    fn add(self, rhs: Dual) -> Dual {
        Dual::new(self + rhs.real, rhs.dual)
    }
}

// f64 - Dual.
impl Sub<Dual> for f64 {
    type Output = Dual;
    fn sub(self, rhs: Dual) -> Dual {
        Dual::new(self - rhs.real, rhs.dual)
    }
}

// f64 * Dual.
impl Mul<Dual> for f64 {
    type Output = Dual;
    fn mul(self, rhs: Dual) -> Dual {
        Dual::new(self * rhs.real, self * rhs.dual)
    }
}

// f64 / Dual.
impl Div<Dual> for f64 {
    type Output = Dual;
    fn div(self, rhs: Dual) -> Dual {
        Dual::new(self / rhs.real, -self * rhs.dual / rhs.real.powi(2))
    }
}

// f64 % Dual.
impl Rem<Dual> for f64 {
    type Output = Dual;
    fn rem(self, rhs: Dual) -> Dual {
        Dual::new(self % rhs.real, -(self / rhs.real).floor() * rhs.dual)
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
    use super::*;
    use linalg_traits::Scalar;
    use numtest::*;
    use std::f64::consts::{E, FRAC_PI_4, FRAC_PI_6};

    // Implementing the Compare trait exclusively for testing purposes.
    impl Compare for Dual {
        fn is_equal(&self, other: Self) -> bool {
            let real_equal = self.get_real().is_equal(other.get_real());
            let dual_equal = self.get_dual().is_equal(other.get_dual());
            real_equal & dual_equal
        }
        fn is_equal_to_decimal(&self, other: Self, decimal: i32) -> (bool, i32) {
            let (real_equal, real_decimal) = self
                .get_real()
                .is_equal_to_decimal(other.get_real(), decimal);
            let (dual_equal, dual_decimal) = self
                .get_dual()
                .is_equal_to_decimal(other.get_dual(), decimal);
            (real_equal & dual_equal, real_decimal.min(dual_decimal))
        }
        fn is_equal_to_atol(&self, other: Self, atol: Self) -> (bool, Self) {
            let (real_equal, real_abs_diff) = self
                .get_real()
                .is_equal_to_atol(other.get_real(), atol.get_real());
            let (dual_equal, dual_abs_diff) = self
                .get_dual()
                .is_equal_to_atol(other.get_dual(), atol.get_dual());
            (
                real_equal & dual_equal,
                Dual::new(real_abs_diff, dual_abs_diff),
            )
        }
        fn is_equal_to_rtol(&self, other: Self, rtol: Self) -> (bool, Self) {
            let (real_equal, real_rel_diff) = self
                .get_real()
                .is_equal_to_rtol(other.get_real(), rtol.get_real());
            let (dual_equal, dual_rel_diff) = self
                .get_dual()
                .is_equal_to_rtol(other.get_dual(), rtol.get_dual());
            (
                real_equal & dual_equal,
                Dual::new(real_rel_diff, dual_rel_diff),
            )
        }
    }

    #[test]
    fn test_new() {
        let num1 = Dual::new(1.0, 2.0);
        let num2 = Dual {
            real: 1.0,
            dual: 2.0,
        };
        assert_eq!(num1.real, num2.real);
        assert_eq!(num1.dual, num2.dual);
    }

    #[test]
    fn test_get_real() {
        let num = Dual::new(1.0, 2.0);
        assert_eq!(num.get_real(), 1.0);
    }

    #[test]
    fn test_get_dual() {
        let num = Dual::new(1.0, 2.0);
        assert_eq!(num.get_dual(), 2.0);
    }

    #[test]
    fn test_add_dual_dual() {
        assert_eq!(
            Dual::new(1.0, 2.0) + Dual::new(3.0, 4.0),
            Dual::new(4.0, 6.0)
        );
    }

    #[test]
    fn test_sub_dual_dual() {
        assert_eq!(
            Dual::new(1.0, 2.0) - Dual::new(4.0, 3.0),
            Dual::new(-3.0, -1.0)
        );
    }

    #[test]
    fn test_mul_dual_dual() {
        assert_eq!(
            Dual::new(1.0, 2.0) * Dual::new(3.0, -4.0),
            Dual::new(3.0, 2.0)
        );
    }

    #[test]
    fn test_div_dual_dual() {
        assert_eq!(
            Dual::new(1.0, 2.0) / Dual::new(3.0, 4.0),
            Dual::new(1.0 / 3.0, 2.0 / 9.0)
        );
    }

    #[test]
    fn test_rem_dual_dual() {
        // Spot check.
        assert_eq!(
            Dual::new(5.0, 2.0) % Dual::new(3.0, 4.0),
            Dual::new(2.0, -2.0)
        );

        // Check parity with the truncated definition of the remainder.
        //  --> Reference: https://en.wikipedia.org/wiki/Modulo#In_programming_languages
        let a = Dual::new(5.0, 2.0);
        let n = Dual::new(3.0, 5.0);
        assert_eq!(a % n, a - n * (a / n).trunc());
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

    #[test]
    fn test_from_str_radix() {
        assert_eq!(
            Dual::from_str_radix("2.125", 10).unwrap(),
            Dual::new(2.125, 0.0)
        );
    }

    #[test]
    fn test_to_i64() {
        assert_eq!(Dual::new(1.0, 2.0).to_i64().unwrap(), 1_i64);
        assert_eq!(Dual::new(-1.0, 2.0).to_i64().unwrap(), -1_i64);
    }

    #[test]
    fn test_to_u64() {
        assert_eq!(Dual::new(1.0, 2.0).to_u64().unwrap(), 1_u64);
        assert!(Dual::new(-1.0, 2.0).to_u64().is_none());
    }

    #[test]
    fn test_to_f64() {
        assert_eq!(Dual::new(1.0, 2.0).to_f64().unwrap(), 1.0_f64);
        assert_eq!(Dual::new(-1.0, 2.0).to_f64().unwrap(), -1.0_f64);
    }

    #[test]
    fn test_from_i64() {
        assert_eq!(<Dual as NumCast>::from(1_i64).unwrap(), Dual::new(1.0, 0.0));
        assert_eq!(
            <Dual as NumCast>::from(-1_i64).unwrap(),
            Dual::new(-1.0, 0.0)
        );
    }

    #[test]
    fn test_from_u64() {
        assert_eq!(<Dual as NumCast>::from(1_u64).unwrap(), Dual::new(1.0, 0.0));
    }

    #[test]
    fn test_from_f64() {
        assert_eq!(<Dual as NumCast>::from(1_f64).unwrap(), Dual::new(1.0, 0.0));
    }

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
        assert!(Dual::new(f64::NAN, 0.0).is_nan());
        assert!(!Dual::new(0.0, f64::NAN).is_nan());
        assert!(!Dual::new(0.0, 0.0).is_nan());
    }

    #[test]
    fn test_is_infinite() {
        assert!(Dual::infinity().is_infinite());
        assert!(Dual::neg_infinity().is_infinite());
        assert!(Dual::new(f64::INFINITY, 0.0).is_infinite());
        assert!(Dual::new(f64::NEG_INFINITY, 0.0).is_infinite());
        assert!(!Dual::new(0.0, f64::INFINITY).is_infinite());
        assert!(!Dual::new(0.0, f64::NEG_INFINITY).is_infinite());
        assert!(!Dual::new(0.0, 0.0).is_infinite());
    }

    #[test]
    fn test_is_finite() {
        assert!(!Dual::infinity().is_finite());
        assert!(!Dual::neg_infinity().is_finite());
        assert!(!Dual::new(f64::INFINITY, 0.0).is_finite());
        assert!(!Dual::new(f64::NEG_INFINITY, 0.0).is_finite());
        assert!(Dual::new(0.0, f64::INFINITY).is_finite());
        assert!(Dual::new(0.0, f64::NEG_INFINITY).is_finite());
        assert!(Dual::new(0.0, 0.0).is_finite());
    }

    /// # References
    ///
    /// * https://docs.rs/num-traits/latest/num_traits/float/trait.Float.html#tymethod.is_normal
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
        assert_eq!(Dual::new(2.7, 2.7).floor(), Dual::new(2.0, 0.0));
        assert_eq!(Dual::new(-2.7, -2.7).floor(), Dual::new(-3.0, 0.0));
    }

    #[test]
    fn test_ceil() {
        assert_eq!(Dual::new(2.7, 2.7).ceil(), Dual::new(3.0, 0.0));
        assert_eq!(Dual::new(-2.7, -2.7).ceil(), Dual::new(-2.0, 0.0));
    }

    #[test]
    fn test_round() {
        assert_eq!(Dual::new(2.3, 2.3).round(), Dual::new(2.0, 0.0));
        assert_eq!(Dual::new(2.7, 2.7).round(), Dual::new(3.0, 0.0));
        assert_eq!(Dual::new(-2.7, -2.7).round(), Dual::new(-3.0, 0.0));
        assert_eq!(Dual::new(-2.3, -2.3).round(), Dual::new(-2.0, 0.0));
    }

    #[test]
    fn test_trunc() {
        assert_eq!(Dual::new(2.3, 2.3).trunc(), Dual::new(2.0, 0.0));
        assert_eq!(Dual::new(2.7, 2.7).trunc(), Dual::new(2.0, 0.0));
        assert_eq!(Dual::new(-2.7, -2.7).trunc(), Dual::new(-2.0, 0.0));
        assert_eq!(Dual::new(-2.3, -2.3).trunc(), Dual::new(-2.0, 0.0));
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
        assert_eq!(Dual::new(2.0, 4.0).signum(), Dual::new(1.0, 0.0));
        assert_eq!(Dual::new(-2.0, -4.0).signum(), Dual::new(-1.0, 0.0));
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
        assert_eq!(Dual::new(2.0, -5.0).powi(0), Dual::new(1.0, 0.0));
        assert_eq!(Dual::new(2.0, -5.0).powi(1), Dual::new(2.0, -5.0));
        assert_eq!(Dual::new(2.0, -5.0).powi(2), Dual::new(4.0, -20.0));
        assert_eq!(Dual::new(2.0, -5.0).powi(3), Dual::new(8.0, -60.0));
    }

    #[test]
    fn test_powf() {
        // Test against powi for integer powers.
        assert_eq!(
            Dual::new(2.0, -5.0).powf(Dual::new(0.0, 0.0)),
            Dual::new(2.0, -5.0).powi(0)
        );
        assert_eq!(
            Dual::new(2.0, -5.0).powf(Dual::new(1.0, 0.0)),
            Dual::new(2.0, -5.0).powi(1)
        );
        assert_eq!(
            Dual::new(2.0, -5.0).powf(Dual::new(2.0, 0.0)),
            Dual::new(2.0, -5.0).powi(2)
        );
        assert_equal_to_decimal!(
            Dual::new(2.0, -5.0).powf(Dual::new(3.0, 0.0)),
            Dual::new(2.0, -5.0).powi(3),
            14
        );

        // Test against sqrt.
        assert_equal_to_decimal!(
            Dual::new(2.0, -5.0).powf(Dual::new(0.5, 0.0)),
            Dual::new(2.0, -5.0).sqrt(),
            15
        );

        // Test against cbrt.
        assert_equal_to_decimal!(
            Dual::new(2.0, -5.0).powf(Dual::new(1.0 / 3.0, 0.0)),
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
            Dual::new(5.0, 8.0).log(Dual::new(4.5, 0.0)),
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
    fn test_acos() {
        assert_equal_to_decimal!(
            Dual::new(3.0.sqrt() / 2.0, 3.0).acos(),
            Dual::new(FRAC_PI_6, -6.0),
            15
        );
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
    fn test_atanh() {
        assert_equal_to_decimal!(Dual::new(1.0, 2.0).tanh().atanh(), Dual::new(1.0, 2.0), 16);
    }

    #[test]
    fn test_integer_decode() {
        assert_eq!(
            Dual::new(1.2345e-5, 6.789e-7).integer_decode(),
            (1.2345e-5).integer_decode()
        );
    }

    #[test]
    fn test_add_assign_dual_dual() {
        let mut a = Dual::new(1.0, 2.0);
        a += Dual::new(3.0, 4.0);
        assert_eq!(a, Dual::new(4.0, 6.0));
    }

    #[test]
    fn test_sub_assign_dual_dual() {
        let mut a = Dual::new(1.0, 2.0);
        a -= Dual::new(4.0, 3.0);
        assert_eq!(a, Dual::new(-3.0, -1.0));
    }

    #[test]
    fn test_mul_assign_dual_dual() {
        let mut a = Dual::new(1.0, 2.0);
        a *= Dual::new(3.0, -4.0);
        assert_eq!(a, Dual::new(3.0, 2.0));
    }

    #[test]
    fn test_div_assign_dual_dual() {
        let mut a = Dual::new(1.0, 2.0);
        a /= Dual::new(3.0, 4.0);
        assert_eq!(a, Dual::new(1.0 / 3.0, 2.0 / 9.0));
    }

    #[test]
    fn test_rem_assign_dual_dual() {
        let mut a = Dual::new(5.0, 2.0);
        let b = Dual::new(3.0, 4.0);
        a %= b;
        assert_eq!(a, Dual::new(2.0, -2.0));
    }

    #[test]
    fn test_add_dual_f64() {
        assert_eq!(Dual::new(1.0, 2.0) + 3.0, Dual::new(4.0, 2.0));
    }

    #[test]
    fn test_sub_dual_f64() {
        assert_eq!(Dual::new(1.0, 2.0) - 3.0, Dual::new(-2.0, 2.0));
    }

    #[test]
    fn test_mul_dual_f64() {
        assert_eq!(Dual::new(1.0, -2.0) * 3.0, Dual::new(3.0, -6.0));
    }

    #[test]
    fn test_div_dual_f64() {
        assert_eq!(Dual::new(1.0, 2.0) / 4.0, Dual::new(0.25, 0.5));
    }

    #[test]
    fn test_rem_dual_f64() {
        // Spot check.
        assert_eq!(Dual::new(5.0, 1.0) % 3.0, Dual::new(2.0, 1.0));

        // Check parity with the truncated definition of the remainder.
        //  --> Reference: https://en.wikipedia.org/wiki/Modulo#In_programming_languages
        let a = Dual::new(5.0, 1.0);
        let n = 3.0;
        assert_eq!(a % n, a - n * (a / n).trunc());
    }

    #[test]
    fn test_add_assign_dual_f64() {
        let mut a = Dual::new(1.0, 2.0);
        a += 3.0;
        assert_eq!(a, Dual::new(4.0, 2.0));
    }

    #[test]
    fn test_sub_assign_dual_f64() {
        let mut a = Dual::new(1.0, 2.0);
        a -= 3.0;
        assert_eq!(a, Dual::new(-2.0, 2.0));
    }

    #[test]
    fn test_mul_assign_dual_f64() {
        let mut a = Dual::new(2.0, -3.0);
        a *= 5.0;
        assert_eq!(a, Dual::new(10.0, -15.0));
    }

    #[test]
    fn test_div_assign_dual_f64() {
        let mut a = Dual::new(1.0, 2.0);
        a /= 4.0;
        assert_eq!(a, Dual::new(0.25, 0.5));
    }

    #[test]
    fn test_rem_assign_dual_f64() {
        let mut a = Dual::new(5.0, 1.0);
        let n = 3.0;
        a %= n;
        assert_eq!(a, Dual::new(2.0, 1.0));
    }

    #[test]
    fn test_add_f64_dual() {
        assert_eq!(1.0 + Dual::new(2.0, 3.0), Dual::new(3.0, 3.0));
    }

    #[test]
    fn test_sub_f64_dual() {
        assert_eq!(1.0 - Dual::new(2.0, 3.0), Dual::new(-1.0, 3.0));
    }

    #[test]
    fn test_mul_f64_dual() {
        assert_eq!(5.0 * Dual::new(2.0, -3.0), Dual::new(10.0, -15.0));
    }

    #[test]
    fn test_div_f64_dual() {
        assert_eq!(5.0 / Dual::new(2.0, -3.0), Dual::new(2.5, 3.75));
    }

    #[test]
    fn test_rem_f64_dual() {
        // Spot check.
        assert_eq!(5.0 % Dual::new(2.0, -3.0), Dual::new(1.0, 6.0));

        // Check parity with "Dual % Dual" implementation.
        assert_eq!(
            5.0 % Dual::new(2.0, -3.0),
            Dual::new(5.0, 0.0) % Dual::new(2.0, -3.0)
        );
    }

    // This just verifies that the scalar trait was fully implemented.
    #[test]
    fn test_scalar() {
        fn add_scalar<S: Scalar>(x: S, y: S) -> S {
            x + y
        }
        assert_eq!(
            add_scalar(Dual::new(5.0, 4.0), Dual::new(3.0, 2.0)),
            Dual::new(8.0, 6.0)
        );
    }
}
