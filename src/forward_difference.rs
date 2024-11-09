//! Forward difference approximations.

// Linking project modules.
pub(crate) mod gradient;
pub(crate) mod sderivative;
pub(crate) mod vderivative;

// Re-exports.
pub use gradient::gradient;
pub use sderivative::sderivative;
pub use vderivative::vderivative;
