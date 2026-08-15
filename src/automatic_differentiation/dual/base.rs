//! Implements [`linalg_traits::real_field::Base`] for [`crate::Dual`].

use crate::automatic_differentiation::dual::dual::Dual;
use std::fmt::Display;

impl Display for Dual {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("{} + {}ε", self.real, self.dual))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use linalg_traits::real_field::assert_base;

    #[test]
    fn test_base() {
        assert_base::<Dual>(
            Dual::new(1.5, 2.5),
            Dual::new(0.0, 0.0),
            "Dual { real: 1.5, dual: 2.5 }",
            "1.5 + 2.5ε",
        );
    }
}
