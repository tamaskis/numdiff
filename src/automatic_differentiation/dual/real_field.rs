//! Implements [`linalg_traits::RealField`] for [`crate::Dual`].

use crate::automatic_differentiation::dual::dual::Dual;
use linalg_traits::{RealField, verify_trait_implemented};

const _: bool = verify_trait_implemented!(Dual: RealField);

linalg_traits::impl_real_field!(Dual);

#[cfg(test)]
mod tests {
    use super::*;
    use linalg_traits::real_field::{
        assert_f64_conversions, assert_f64_lhs_ops, assert_f64_rhs_ops,
        assert_no_trait_disambiguation_needed, assert_real_field_operations,
    };

    #[test]
    fn test_f64_conversions() {
        assert_f64_conversions(Dual::new(3.0, 2.0), 4.0, 3.0, Dual::new(4.0, 0.0));
    }

    #[test]
    fn test_f64_lhs_ops() {
        assert_f64_lhs_ops(
            3.0,
            Dual::new(2.0, 5.0),
            false,
            false,
            false,
            true,
            true,
            Dual::new(5.0, 5.0),
            Dual::new(1.0, -5.0),
            Dual::new(6.0, 15.0),
            Dual::new(1.5, -3.75),
            Dual::new(1.0, -5.0),
        );
    }

    #[test]
    fn test_f64_rhs_ops() {
        assert_f64_rhs_ops(
            Dual::new(2.0, 5.0),
            3.0,
            false,
            true,
            true,
            false,
            false,
            Dual::new(5.0, 5.0),
            Dual::new(-1.0, 5.0),
            Dual::new(6.0, 15.0),
            Dual::new(2.0 / 3.0, 5.0 / 3.0),
            Dual::new(2.0, 5.0),
        );
    }

    #[test]
    fn test_no_trait_disambiguation_needed() {
        assert_no_trait_disambiguation_needed(
            Dual::new(7.0, 2.0),
            Dual::new(2.0, 5.0),
            Dual::new(-7.0, -2.0),
        );
    }

    #[test]
    fn test_real_field_operations() {
        assert_real_field_operations(
            Dual::new(7.0, 2.0),
            Dual::new(2.0, 5.0),
            Dual::new(-7.0, -2.0),
            Dual::new(9.0, 7.0),
            Dual::new(5.0, -3.0),
            Dual::new(14.0, 39.0),
            Dual::new(3.5, -7.75),
            Dual::new(1.0, -13.0),
            false,
            false,
            false,
            true,
            true,
        );
    }

    #[test]
    fn test_f64_ops() {
        assert_eq!(Dual::new(1.0, 2.0) + 3.0, Dual::new(4.0, 2.0));
        assert_eq!(Dual::new(1.0, 2.0) - 3.0, Dual::new(-2.0, 2.0));
        assert_eq!(Dual::new(1.0, -2.0) * 3.0, Dual::new(3.0, -6.0));
        assert_eq!(Dual::new(1.0, 2.0) / 4.0, Dual::new(0.25, 0.5));
        assert_eq!(1.0 + Dual::new(2.0, 3.0), Dual::new(3.0, 3.0));
        assert_eq!(5.0 * Dual::new(2.0, -3.0), Dual::new(10.0, -15.0));

        let dual: Dual = 3.0.into();
        assert_eq!(dual, Dual::new(3.0, 0.0));

        let dual_f64: f64 = Dual::new(3.0, 2.0).into();
        assert_eq!(dual_f64, 3.0);
    }
}
