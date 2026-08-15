use crate::automatic_differentiation::dual::dual::Dual;
use num_traits::{Num, One, Zero};

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
        Dual::from_real(0.0)
    }
    fn is_zero(&self) -> bool {
        self.real.is_zero() && self.dual.is_zero()
    }
}

impl One for Dual {
    fn one() -> Self {
        Dual::from_real(1.0)
    }
}

impl Num for Dual {
    type FromStrRadixErr = <f64 as Num>::FromStrRadixErr;
    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        f64::from_str_radix(str, radix).map(Dual::from_real)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_zero() {
        // Construction
        assert_eq!(Dual::zero(), Dual::from_real(0.0));

        // Zero-check.
        assert!(Dual::zero().is_zero());
        assert!(Dual::from_real(0.0).is_zero());

        // Dual::zero() * Dual = Dual::zero().
        assert_eq!(Dual::zero() * Dual::new(1.0, 2.0), Dual::zero());
    }

    #[test]
    fn test_one() {
        // Construction.
        assert_eq!(Dual::one(), Dual::from_real(1.0));

        // Dual::one() * Dual = Dual.
        assert_eq!(Dual::one() * Dual::new(1.0, 2.0), Dual::new(1.0, 2.0));

        // Dual::one() * scalar = Dual(scalar, scalar).
        assert_eq!(Dual::one() * 5.0, Dual::from_real(5.0));
    }

    #[test]
    fn test_from_str_radix() {
        assert_eq!(
            Dual::from_str_radix("2.125", 10).unwrap(),
            Dual::from_real(2.125)
        );
    }
}
