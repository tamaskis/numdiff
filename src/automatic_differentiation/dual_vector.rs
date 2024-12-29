use crate::automatic_differentiation::dual::Dual;
use linalg_traits::{Scalar, Vector};

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

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{dvector, SVector};
    use ndarray::array;

    #[test]
    fn test_vec() {
        let vec = vec![1.0, 2.0, 3.0];
        assert_eq!(
            vec.to_dual_vector(),
            vec![
                Dual::new(1.0, 0.0),
                Dual::new(2.0, 0.0),
                Dual::new(3.0, 0.0)
            ]
        );
    }

    #[test]
    fn test_nalgebra_dvector() {
        let vec = dvector![1.0, 2.0, 3.0];
        assert_eq!(
            vec.to_dual_vector(),
            dvector![
                Dual::new(1.0, 0.0),
                Dual::new(2.0, 0.0),
                Dual::new(3.0, 0.0)
            ]
        );
    }

    #[test]
    fn test_nalgebra_svector() {
        let vec = SVector::<f64, 3>::from_row_slice(&[1.0, 2.0, 3.0]);
        assert_eq!(
            vec.to_dual_vector(),
            SVector::<Dual, 3>::from_row_slice(&[
                Dual::new(1.0, 0.0),
                Dual::new(2.0, 0.0),
                Dual::new(3.0, 0.0)
            ])
        );
    }

    #[test]
    fn test_ndarray_array1() {
        let vec = array![1.0, 2.0, 3.0];
        assert_eq!(
            vec.to_dual_vector(),
            vec![
                Dual::new(1.0, 0.0),
                Dual::new(2.0, 0.0),
                Dual::new(3.0, 0.0)
            ]
        );
    }
}
