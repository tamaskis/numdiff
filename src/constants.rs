//! Constants used for numerical differentiation.

use std::sync::LazyLock;

/// Square root of the machine epsilon ($\sqrt{\varepsilon}$).
pub static SQRT_EPS: LazyLock<f64> = LazyLock::new(|| f64::EPSILON.sqrt());

/// Cube root of the machine epsilon ($\varepsilon^{1/3}$).
pub static CBRT_EPS: LazyLock<f64> = LazyLock::new(|| f64::EPSILON.cbrt());
