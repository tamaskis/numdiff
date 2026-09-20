//! Central difference approximations.

// Module declarations.
mod derivative;
mod derivative2;
mod directional_derivative;
mod gradient;
mod hessian;
mod jacobian;
mod mixed_partial_derivative2;
mod partial_derivative;
mod partial_derivative2;

// Re-exports.
pub use derivative::scalar_valued::sderivative;
pub use derivative::vector_valued::vderivative;
pub use derivative2::scalar_valued::sderivative2;
pub use derivative2::vector_valued::vderivative2;
pub use directional_derivative::directional_derivative;
pub use gradient::gradient;
pub use hessian::scalar_valued::shessian;
pub use hessian::vector_valued::vhessian;
pub use jacobian::jacobian;
pub use mixed_partial_derivative2::scalar_valued::mixed_spartial_derivative2;
pub use mixed_partial_derivative2::vector_valued::mixed_vpartial_derivative2;
pub use partial_derivative::scalar_valued::spartial_derivative;
pub use partial_derivative::vector_valued::vpartial_derivative;
pub use partial_derivative2::scalar_valued::spartial_derivative2;
pub use partial_derivative2::vector_valued::vpartial_derivative2;
