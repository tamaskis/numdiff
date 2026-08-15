/// First-order dual number.
///
/// A dual number is represented as
///
/// `real + dual * ε`
///
/// where `ε² = 0`.
#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
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

// --------
// TESTING.
// --------

#[cfg(test)]
mod tests {
    use super::*;
    use numtest::*;

    // Implementing the Compare trait exclusively for testing purposes.
    impl Compare for Dual {
        const NAN: Self = Self {
            real: f64::NAN,
            dual: f64::NAN,
        };
        const MAX_DECIMAL_NUMTEST: u32 = f64::MAX_DECIMAL_NUMTEST;

        fn max_decimal_numtest(&self) -> u32 {
            f64::MAX_DECIMAL_NUMTEST
        }

        fn max_10_exp_numtest(&self) -> i32 {
            f64::MAX_10_EXP
        }

        fn min_10_exp_numtest(&self) -> i32 {
            f64::MIN_10_EXP
        }

        fn epsilon_numtest(&self) -> Self {
            Self::from_real(f64::EPSILON)
        }

        fn is_nan_numtest(&self) -> bool {
            self.real.is_nan_numtest() || self.dual.is_nan_numtest()
        }

        fn is_infinite_numtest(&self) -> bool {
            self.real.is_infinite_numtest() || self.dual.is_infinite_numtest()
        }

        fn abs_numtest(self) -> Self {
            Self::new(self.real.abs_numtest(), self.dual.abs_numtest())
        }

        fn powi_numtest(self, exponent: i32) -> Self {
            Self::new(
                self.real.powi_numtest(exponent),
                self.dual.powi_numtest(exponent),
            )
        }

        fn signum_numtest(self) -> Self {
            Self::new(self.real.signum_numtest(), self.dual.signum_numtest())
        }

        fn max_numtest(self, other: Self) -> Self {
            Self::new(
                self.real.max_numtest(other.real),
                self.dual.max_numtest(other.dual),
            )
        }

        fn is_equal_numtest(&self, other: Self) -> bool {
            let real_equal = self.real.is_equal_numtest(other.real);
            let dual_equal = self.dual.is_equal_numtest(other.dual);
            real_equal & dual_equal
        }

        fn is_equal_to_decimal_numtest(&self, other: Self, decimal: i32) -> (bool, i32) {
            let (real_equal, real_decimal) =
                self.real.is_equal_to_decimal_numtest(other.real, decimal);
            let (dual_equal, dual_decimal) =
                self.dual.is_equal_to_decimal_numtest(other.dual, decimal);
            (real_equal & dual_equal, real_decimal.min(dual_decimal))
        }

        fn is_equal_to_atol_numtest(&self, other: Self, atol: Self) -> (bool, Self) {
            let (real_equal, real_abs_diff) =
                self.real.is_equal_to_atol_numtest(other.real, atol.real);
            let (dual_equal, dual_abs_diff) =
                self.dual.is_equal_to_atol_numtest(other.dual, atol.dual);
            (
                real_equal & dual_equal,
                Dual::new(real_abs_diff, dual_abs_diff),
            )
        }

        fn is_equal_to_rtol_numtest(&self, other: Self, rtol: Self) -> (bool, Self) {
            let (real_equal, real_rel_diff) =
                self.real.is_equal_to_rtol_numtest(other.real, rtol.real);
            let (dual_equal, dual_rel_diff) =
                self.dual.is_equal_to_rtol_numtest(other.dual, rtol.dual);
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
}
