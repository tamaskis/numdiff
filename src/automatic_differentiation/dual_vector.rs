use linalg_traits::{Scalar, Vector};

use crate::automatic_differentiation::dual::Dual;
// TODO: need a method to create a vector of dual numbers
// TODO: need a method to take the dual portion of a vector of dual numbers
/// Trait to create a vector of dual numbers.
pub trait DualVector<S, V>
where
    S: Scalar,
    V: Vector<S>,
{
    /// Convert this vector of scalars to a vector of dual numbers.
    ///
    /// # Returns
    ///
    /// A copy of this vector with each element converted to a dual number (with dual part `0.0`).
    fn to_dual_vector(self) -> V::GenericVector<Dual>;
}

impl<S, V> DualVector<S, V> for V
where
    S: Scalar,
    V: Vector<S>,
{
    fn to_dual_vector(self) -> V::GenericVector<Dual> {
        let mut vec_dual = V::GenericVector::new_with_length(self.len());
        for i in 0..self.len() {
            vec_dual[i] = Dual::new(self[i].to_f64().unwrap(), 0.0);
        }
        vec_dual
    }
}
