use linalg_traits::{Scalar, Vector};

use crate::automatic_differentiation::dual::Dual;

pub trait DualVector<S: Scalar, V:Vector<S>> {
    fn new_dual_vector(self) -> 
}

impl<S, V> DualVector for V
where
    S: Scalar,
    V: Vector<S>,
{
    type DualVector = V<Dual>;
}
