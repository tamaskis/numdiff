//! Implements [`linalg_traits::RealField`] for [`crate::HyperDual`].

use crate::automatic_differentiation::hyper_dual::hyper_dual::HyperDual;
use linalg_traits::{RealField, verify_trait_implemented};

const _: bool = verify_trait_implemented!(HyperDual: RealField);

linalg_traits::impl_real_field!(HyperDual);

#[cfg(test)]
mod tests {
    use super::*;
    use linalg_traits::real_field::{
        assert_f64_conversions, assert_f64_lhs_ops, assert_f64_rhs_ops,
        assert_no_trait_disambiguation_needed, assert_real_field_operations,
    };

    #[test]
    fn test_f64_conversions() {
        assert_f64_conversions(
            HyperDual::new(3.0, 2.0, 1.0, -0.5),
            4.0,
            3.0,
            HyperDual::new(4.0, 0.0, 0.0, 0.0),
        );
    }

    #[test]
    fn test_f64_lhs_ops() {
        assert_f64_lhs_ops(
            3.0,
            HyperDual::new(2.0, 5.0, 4.0, -6.0),
            false,
            false,
            false,
            true,
            true,
            HyperDual::new(5.0, 5.0, 4.0, -6.0),
            HyperDual::new(1.0, -5.0, -4.0, 6.0),
            HyperDual::new(6.0, 15.0, 12.0, -18.0),
            HyperDual::new(1.5, -3.75, -3.0, 19.5),
            HyperDual::new(1.0, -5.0, -4.0, 6.0),
        );
    }

    #[test]
    fn test_f64_rhs_ops() {
        assert_f64_rhs_ops(
            HyperDual::new(2.0, 5.0, 4.0, -6.0),
            3.0,
            false,
            true,
            true,
            false,
            false,
            HyperDual::new(5.0, 5.0, 4.0, -6.0),
            HyperDual::new(-1.0, 5.0, 4.0, -6.0),
            HyperDual::new(6.0, 15.0, 12.0, -18.0),
            HyperDual::new(2.0 / 3.0, 1.6666666666666665, 4.0 / 3.0, -2.0),
            HyperDual::new(2.0, 5.0, 4.0, -6.0),
        );
    }

    #[test]
    fn test_no_trait_disambiguation_needed() {
        assert_no_trait_disambiguation_needed(
            HyperDual::new(7.0, 2.0, -1.0, 3.0),
            HyperDual::new(2.0, 5.0, 4.0, -6.0),
            HyperDual::new(-7.0, -2.0, 1.0, -3.0),
        );
    }

    #[test]
    fn test_real_field_operations() {
        assert_real_field_operations(
            HyperDual::new(7.0, 2.0, -1.0, 3.0),
            HyperDual::new(2.0, 5.0, 4.0, -6.0),
            HyperDual::new(-7.0, -2.0, 1.0, -3.0),
            HyperDual::new(9.0, 7.0, 3.0, -3.0),
            HyperDual::new(5.0, -3.0, -5.0, 9.0),
            HyperDual::new(14.0, 39.0, 26.0, -33.0),
            HyperDual::new(3.5, -7.75, -7.5, 46.25),
            HyperDual::new(1.0, -13.0, -13.0, 21.0),
            false,
            false,
            false,
            true,
            true,
        );
    }

    #[test]
    fn test_f64_ops() {
        assert_eq!(
            HyperDual::new(1.0, 2.0, 3.0, 4.0) + 3.0,
            HyperDual::new(4.0, 2.0, 3.0, 4.0)
        );
        assert_eq!(
            HyperDual::new(1.0, 2.0, 3.0, 4.0) / 4.0,
            HyperDual::new(0.25, 0.5, 0.75, 1.0)
        );
        assert_eq!(
            1.0 + HyperDual::new(2.0, 3.0, 4.0, 5.0),
            HyperDual::new(3.0, 3.0, 4.0, 5.0)
        );

        let hyper_dual: HyperDual = 3.0.into();
        assert_eq!(hyper_dual, HyperDual::new(3.0, 0.0, 0.0, 0.0));

        let hyper_dual_f64: f64 = HyperDual::new(3.0, 2.0, 1.5, -0.75).into();
        assert_eq!(hyper_dual_f64, 3.0);
    }
}
