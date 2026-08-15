use crate::automatic_differentiation::dual::dual::Dual;
use std::ops::{Add, Div, Mul, Rem, Sub};

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

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::Float;

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
}
