use crate::automatic_differentiation::dual::dual::Dual;

use ndarray::ScalarOperand;

// ---------------------------------------------
// Implementing ndarray::ScalarOperand for Dual.
// ---------------------------------------------
// NOTE: This is required for implementing linalg_traits::Scalar.

impl ScalarOperand for Dual {}
