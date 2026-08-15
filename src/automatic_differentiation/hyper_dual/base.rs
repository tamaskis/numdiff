//! Implements [`linalg_traits::real_field::Base`] for [`crate::HyperDual`].

use crate::automatic_differentiation::hyper_dual::hyper_dual::HyperDual;
use std::fmt::Display;

impl Display for HyperDual {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "{} + {}ε₁ + {}ε₂ + {}ε₁ε₂",
            self.a, self.b, self.c, self.d
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use linalg_traits::real_field::assert_base;

    #[test]
    fn test_base() {
        assert_base::<HyperDual>(
            HyperDual::new(1.5, 2.5, 3.5, 4.5),
            HyperDual::new(0.0, 0.0, 0.0, 0.0),
            "HyperDual { a: 1.5, b: 2.5, c: 3.5, d: 4.5 }",
            "1.5 + 2.5ε₁ + 3.5ε₂ + 4.5ε₁ε₂",
        );
    }
}
