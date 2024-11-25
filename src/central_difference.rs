//! Forward difference approximations.

// Module declarations.
pub(crate) mod derivative;
pub(crate) mod directional_derivative;
pub(crate) mod gradient;
pub(crate) mod hessian;
pub(crate) mod jacobian;
pub(crate) mod partial_derivative;

// Re-exports.
pub use derivative::scalar_valued::sderivative;
pub use derivative::vector_valued::vderivative;
pub use directional_derivative::directional_derivative;
pub use gradient::gradient;
pub use hessian::scalar_valued::shessian;
pub use hessian::vector_valued::vhessian;
pub use jacobian::jacobian;
pub use partial_derivative::scalar_valued::spartial_derivative;
pub use partial_derivative::vector_valued::vpartial_derivative;
