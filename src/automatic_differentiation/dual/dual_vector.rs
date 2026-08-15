use crate::Dual;
use linalg_traits::{RealField, Vector};

/// Trait to create a vector of dual numbers.
pub trait DualVector<R, V>
where
    R: RealField,
    V: Vector<R>,
{
    /// Convert this vector of scalars to a vector of dual numbers.
    ///
    /// # Returns
    ///
    /// A copy of this vector with each element converted to a dual number (with dual part `0.0`).
    fn to_dual_vector(self) -> V::VectorT<Dual>;
}

impl<R, V> DualVector<R, V> for V
where
    R: RealField,
    V: Vector<R>,
{
    fn to_dual_vector(self) -> V::VectorT<Dual> {
        let mut vec_dual = V::VectorT::new_with_length(self.len());
        for i in 0..self.len() {
            vec_dual[i] = Dual::from_real(self[i].into());
        }
        vec_dual
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "nalgebra")]
    use nalgebra::{SVector, dvector};
    #[cfg(feature = "ndarray")]
    use ndarray::array;

    #[test]
    fn test_vec() {
        let vec = vec![1.0, 2.0, 3.0];
        assert_eq!(
            vec.to_dual_vector(),
            vec![
                Dual::from_real(1.0),
                Dual::from_real(2.0),
                Dual::from_real(3.0)
            ]
        );
    }

    #[test]
    #[cfg(feature = "nalgebra")]
    fn test_nalgebra_dvector() {
        let vec = dvector![1.0, 2.0, 3.0];
        assert_eq!(
            vec.to_dual_vector(),
            dvector![
                Dual::from_real(1.0),
                Dual::from_real(2.0),
                Dual::from_real(3.0)
            ]
        );
    }

    #[test]
    #[cfg(feature = "nalgebra")]
    fn test_nalgebra_svector() {
        let vec = SVector::<f64, 3>::from_row_slice(&[1.0, 2.0, 3.0]);
        assert_eq!(
            vec.to_dual_vector(),
            SVector::<Dual, 3>::from_row_slice(&[
                Dual::from_real(1.0),
                Dual::from_real(2.0),
                Dual::from_real(3.0)
            ])
        );
    }

    #[test]
    #[cfg(feature = "ndarray")]
    fn test_ndarray_array1() {
        let vec = array![1.0, 2.0, 3.0];
        assert_eq!(
            vec.to_dual_vector(),
            array![
                Dual::from_real(1.0),
                Dual::from_real(2.0),
                Dual::from_real(3.0)
            ]
        );
    }
}
