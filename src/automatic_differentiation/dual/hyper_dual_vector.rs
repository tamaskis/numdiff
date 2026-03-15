use crate::HyperDual;
use linalg_traits::{Scalar, Vector};

/// Trait to create a vector of hyper-dual numbers.
pub trait HyperDualVector<S, V>
where
    S: Scalar,
    V: Vector<S>,
{
    /// Convert this vector of scalars to a vector of hyper-dual numbers.
    ///
    /// # Returns
    ///
    /// A copy of this vector with each element converted to a hyper-dual number (with `ε1`, `ε2`,
    /// and `ε1ε2` coefficients equal to `0.0`).
    fn to_hyper_dual_vector(self) -> V::VectorT<HyperDual>;
}

impl<S, V> HyperDualVector<S, V> for V
where
    S: Scalar,
    V: Vector<S>,
{
    fn to_hyper_dual_vector(self) -> V::VectorT<HyperDual> {
        let mut vec_hyper_dual = V::VectorT::new_with_length(self.len());
        for i in 0..self.len() {
            vec_hyper_dual.vset(
                i,
                HyperDual::new(self.vget(i).to_f64().unwrap(), 0.0, 0.0, 0.0),
            );
        }
        vec_hyper_dual
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{SVector, dvector};
    use ndarray::array;

    #[test]
    fn test_vec() {
        let vec = vec![1.0, 2.0, 3.0];
        assert_eq!(
            vec.to_hyper_dual_vector(),
            vec![
                HyperDual::new(1.0, 0.0, 0.0, 0.0),
                HyperDual::new(2.0, 0.0, 0.0, 0.0),
                HyperDual::new(3.0, 0.0, 0.0, 0.0)
            ]
        );
    }

    #[test]
    fn test_nalgebra_dvector() {
        let vec = dvector![1.0, 2.0, 3.0];
        assert_eq!(
            vec.to_hyper_dual_vector(),
            dvector![
                HyperDual::new(1.0, 0.0, 0.0, 0.0),
                HyperDual::new(2.0, 0.0, 0.0, 0.0),
                HyperDual::new(3.0, 0.0, 0.0, 0.0)
            ]
        );
    }

    #[test]
    fn test_nalgebra_svector() {
        let vec = SVector::<f64, 3>::from_row_slice(&[1.0, 2.0, 3.0]);
        assert_eq!(
            vec.to_hyper_dual_vector(),
            SVector::<HyperDual, 3>::from_row_slice(&[
                HyperDual::new(1.0, 0.0, 0.0, 0.0),
                HyperDual::new(2.0, 0.0, 0.0, 0.0),
                HyperDual::new(3.0, 0.0, 0.0, 0.0)
            ])
        );
    }

    #[test]
    fn test_ndarray_array1() {
        let vec = array![1.0, 2.0, 3.0];
        assert_eq!(
            vec.to_hyper_dual_vector(),
            vec![
                HyperDual::new(1.0, 0.0, 0.0, 0.0),
                HyperDual::new(2.0, 0.0, 0.0, 0.0),
                HyperDual::new(3.0, 0.0, 0.0, 0.0)
            ]
        );
    }
}
