// Module declarations.
#[cfg(feature = "faer")]
pub(crate) mod complex_field_impl;
#[allow(clippy::module_inception)]
pub(crate) mod dual;
pub(crate) mod dual_vector;
pub(crate) mod float_impl;
pub(crate) mod num_cast_impl;
pub(crate) mod num_impl;
pub(crate) mod num_ops_impl;
#[cfg(feature = "faer")]
pub(crate) mod real_field_impl;
#[cfg(feature = "faer")]
pub(crate) mod ref_ops_impl;
#[cfg(feature = "ndarray")]
pub(crate) mod scalar_operand_impl;
#[cfg(feature = "trig")]
pub(crate) mod trig_impl;
