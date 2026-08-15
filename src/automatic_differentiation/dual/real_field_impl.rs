use crate::automatic_differentiation::dual::dual::Dual;
use faer_traits::RealField;

// ---------------------------------------------
// Implementing faer_traits::RealField for Dual.
// ---------------------------------------------
// NOTE: This is required for implementing linalg_traits::Scalar.

#[cfg(feature = "faer")]
impl RealField for Dual {
    #[inline(always)]
    fn epsilon_impl() -> Self {
        Self {
            real: f64::EPSILON,
            dual: 0.0,
        }
    }

    #[inline(always)]
    fn nbits_impl() -> usize {
        f64::MANTISSA_DIGITS as usize
    }

    #[inline(always)]
    fn min_positive_impl() -> Self {
        Self {
            real: f64::MIN_POSITIVE,
            dual: 0.0,
        }
    }

    #[inline(always)]
    fn max_positive_impl() -> Self {
        Self {
            real: f64::MAX,
            dual: 0.0,
        }
    }

    #[inline(always)]
    fn sqrt_min_positive_impl() -> Self {
        Self {
            real: f64::MIN_POSITIVE.sqrt(),
            dual: 0.0,
        }
    }

    #[inline(always)]
    fn sqrt_max_positive_impl() -> Self {
        Self {
            real: f64::MAX.sqrt(),
            dual: 0.0,
        }
    }
}
