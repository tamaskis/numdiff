//! Constants used for numerical differentiation.

use once_cell::sync::Lazy;

/// Square root of the machine epsilon ($\sqrt{\varepsilon}$).
pub static SQRT_EPS: Lazy<f64> = Lazy::new(|| f64::EPSILON.sqrt());

/// Cube root of the machine epsilon ($\varepsilon^{1/3}$).
pub static CBRT_EPS: Lazy<f64> = Lazy::new(|| f64::EPSILON.cbrt());
