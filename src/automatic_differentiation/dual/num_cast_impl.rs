use crate::automatic_differentiation::dual::dual::Dual;
use num_traits::{NumCast, ToPrimitive};

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
        n.to_f64().map(Dual::from_real)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(
            <Dual as NumCast>::from(1_i64).unwrap(),
            Dual::from_real(1.0)
        );
        assert_eq!(
            <Dual as NumCast>::from(-1_i64).unwrap(),
            Dual::from_real(-1.0)
        );
    }

    #[test]
    fn test_from_u64() {
        assert_eq!(
            <Dual as NumCast>::from(1_u64).unwrap(),
            Dual::from_real(1.0)
        );
    }

    #[test]
    fn test_from_f64() {
        assert_eq!(
            <Dual as NumCast>::from(1_f64).unwrap(),
            Dual::from_real(1.0)
        );
    }
}
