use once_cell::sync::Lazy;

/// Square root of the machine epsilon ($\sqrt{\varepsilon}$).
pub(crate) static SQRT_EPS: Lazy<f64> = Lazy::new(|| f64::EPSILON.sqrt());
