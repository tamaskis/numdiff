#[cfg(feature = "ndarray")]
use ndarray::ScalarOperand;
use num_traits::ToPrimitive;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Rem, RemAssign, Sub, SubAssign};

/// First-order dual number.
///
/// A dual number is represented as
///
/// `real + dual * ε`
///
/// where `ε² = 0`.
#[derive(Clone, Copy, Debug)]
pub struct Dual {
    /// Real part of the dual number.
    pub(crate) real: f64,

    /// Dual part of the dual number.
    pub(crate) dual: f64,
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
    #[must_use]
    pub fn new(real: f64, dual: f64) -> Self {
        Self { real, dual }
    }

    /// Construct a purely real dual number.
    ///
    /// # Arguments
    ///
    /// * `real` - Real part.
    ///
    /// # Returns
    ///
    /// Dual number, `real + 0ε`.
    ///
    /// # Example
    ///
    /// ```
    /// use numdiff::Dual;
    ///
    /// let num = Dual::from_real(3.0);
    /// assert_eq!(num.get_real(), 3.0);
    /// assert_eq!(num.get_dual(), 0.0);
    /// ```
    #[must_use]
    pub fn from_real(real: f64) -> Self {
        Self { real, dual: 0.0 }
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
    #[must_use]
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
    #[must_use]
    pub fn get_dual(self) -> f64 {
        self.dual
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

// ---------------------------------------
// Implementing linalg_traits::ScalarBase.
// ---------------------------------------
// https://docs.rs/linalg-traits/latest/linalg_traits/trait.ScalarBase.html
//
// pub trait ScalarBase:
//  Float
//  + AddAssign<Self>
//  + SubAssign<Self>
//  + MulAssign<Self>
//  + DivAssign<Self>
//  + RemAssign<Self>
//  + Add<f64, Output = Self>
//  + Sub<f64, Output = Self>
//  + Mul<f64, Output = Self>
//  + Div<f64, Output = Self>
//  + Rem<f64, Output = Self>
//  + AddAssign<f64>
//  + SubAssign<f64>
//  + MulAssign<f64>
//  + DivAssign<f64>
//  + RemAssign<f64>
//  + From<f64>
//  + Into<f64>
//  + Debug
//  + 'static { }

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

// ---------------------------------------------
// Implementing ndarray::ScalarOperand for Dual.
// ---------------------------------------------
// NOTE: This is required for implementing linalg_traits::Scalar.

#[cfg(feature = "ndarray")]
impl ScalarOperand for Dual {}

// // ---------------------------------------------
// // Implementing ndarray::LinalgScalar for Dual.
// // ---------------------------------------------
// // NOTE: This is required for implementing linalg_traits::Scalar.

// #[cfg(feature = "ndarray")]
// impl ndarray::LinalgScalar for Dual {}

// ------------------------
// Conversions to/from f64.
// ------------------------

// f64 -> Dual.
impl From<f64> for Dual {
    fn from(value: f64) -> Self {
        Dual::from_real(value)
    }
}

// Dual -> f64.
impl From<Dual> for f64 {
    fn from(value: Dual) -> Self {
        value.real
    }
}

// --------
// TESTING.
// --------

#[cfg(test)]
mod tests {
    use super::*;
    use linalg_traits::{Scalar, ScalarBase};
    use num_traits::Float;
    use numtest::*;

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
    fn test_from_real() {
        assert_eq!(Dual::from_real(1.0), Dual::new(1.0, 0.0));
        assert_eq!(Dual::from_real(-2.5), Dual::new(-2.5, 0.0));
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
            Dual::from_real(5.0) % Dual::new(2.0, -3.0)
        );
    }

    #[test]
    fn test_from_f64_std() {
        let dual: Dual = 3.0.into();
        assert_eq!(dual, Dual::new(3.0, 0.0));
    }

    #[test]
    fn test_into_f64_std() {
        let dual_f64: f64 = Dual::new(3.0, 2.0).into();
        assert_eq!(dual_f64, 3.0);
    }

    // Verify that the ScalarBase trait was fully implemented.
    #[test]
    fn test_scalar_base() {
        fn add_scalar<SB: ScalarBase>(x: SB, y: SB) -> SB {
            x + y
        }
        assert_eq!(
            add_scalar(Dual::new(5.0, 4.0), Dual::new(3.0, 2.0)),
            Dual::new(8.0, 6.0)
        );
    }

    // Verify that the `Scalar` trait is fully implemented when `faer` and `ndarray` features are
    // not enabled.
    #[test]
    fn test_scalar_no_faer_no_ndarray() {
        fn add_scalar<S: Scalar>(x: S, y: S) -> S {
            x + y
        }
        assert_eq!(
            add_scalar(Dual::new(5.0, 4.0), Dual::new(3.0, 2.0)),
            Dual::new(8.0, 6.0)
        );
    }

    // Verify that the `Scalar` trait is fully implemented when the `faer` and `ndarray` features
    // are enabled.
    #[test]
    #[cfg(all(feature = "faer", feature = "ndarray"))]
    fn test_scalar_faer_ndarray() {
        fn add_scalar<S: Scalar>(x: S, y: S) -> S {
            x + y
        }
        assert_eq!(
            add_scalar(Dual::new(5.0, 4.0), Dual::new(3.0, 2.0)),
            Dual::new(8.0, 6.0)
        );
    }
}
