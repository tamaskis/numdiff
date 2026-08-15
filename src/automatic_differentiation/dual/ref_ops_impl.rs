use crate::automatic_differentiation::dual::dual::Dual;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

// ------------------------------------------
// Implementing faer_traits::RefOps for Dual.
// ------------------------------------------
// NOTE: This is required for implementing faer_traits::ComplexField, which in turn is required for
// implementing faer_traits::RealField, which in turn is required for implementing
// linalg_traits::Scalar.

impl<'a> Add<&'a Dual> for Dual {
    type Output = Dual;
    fn add(self, rhs: &'a Dual) -> Dual {
        self + *rhs
    }
}

impl<'a> Sub<&'a Dual> for Dual {
    type Output = Dual;
    fn sub(self, rhs: &'a Dual) -> Dual {
        self - *rhs
    }
}

impl<'a> Mul<&'a Dual> for Dual {
    type Output = Dual;
    fn mul(self, rhs: &'a Dual) -> Dual {
        self * *rhs
    }
}

impl<'a> Div<&'a Dual> for Dual {
    type Output = Dual;
    fn div(self, rhs: &'a Dual) -> Dual {
        self / *rhs
    }
}

impl<'a> Add<Dual> for &'a Dual {
    type Output = Dual;
    fn add(self, rhs: Dual) -> Dual {
        *self + rhs
    }
}

impl<'a> Sub<Dual> for &'a Dual {
    type Output = Dual;
    fn sub(self, rhs: Dual) -> Dual {
        *self - rhs
    }
}

impl<'a> Mul<Dual> for &'a Dual {
    type Output = Dual;
    fn mul(self, rhs: Dual) -> Dual {
        *self * rhs
    }
}

impl<'a> Div<Dual> for &'a Dual {
    type Output = Dual;
    fn div(self, rhs: Dual) -> Dual {
        *self / rhs
    }
}

impl<'a, 'b> Add<&'b Dual> for &'a Dual {
    type Output = Dual;
    fn add(self, rhs: &'b Dual) -> Dual {
        *self + *rhs
    }
}

impl<'a, 'b> Sub<&'b Dual> for &'a Dual {
    type Output = Dual;
    fn sub(self, rhs: &'b Dual) -> Dual {
        *self - *rhs
    }
}

impl<'a, 'b> Mul<&'b Dual> for &'a Dual {
    type Output = Dual;
    fn mul(self, rhs: &'b Dual) -> Dual {
        *self * *rhs
    }
}

impl<'a, 'b> Div<&'b Dual> for &'a Dual {
    type Output = Dual;
    fn div(self, rhs: &'b Dual) -> Dual {
        *self / *rhs
    }
}

impl<'a> AddAssign<&'a Dual> for Dual {
    fn add_assign(&mut self, rhs: &'a Dual) {
        *self += *rhs;
    }
}

impl<'a> SubAssign<&'a Dual> for Dual {
    fn sub_assign(&mut self, rhs: &'a Dual) {
        *self -= *rhs;
    }
}

impl<'a> MulAssign<&'a Dual> for Dual {
    fn mul_assign(&mut self, rhs: &'a Dual) {
        *self *= *rhs;
    }
}

impl<'a> DivAssign<&'a Dual> for Dual {
    fn div_assign(&mut self, rhs: &'a Dual) {
        *self /= *rhs;
    }
}

impl Neg for &Dual {
    type Output = Dual;
    fn neg(self) -> Dual {
        -*self
    }
}
