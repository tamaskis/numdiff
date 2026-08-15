use crate::automatic_differentiation::dual::dual::Dual;
use num_traits::Float;
use std::f64::consts::PI;
use trig::Trig;

impl Trig for Dual {
    fn sin(&self) -> Dual {
        <Dual as Float>::sin(*self)
    }
    fn cos(&self) -> Dual {
        <Dual as Float>::cos(*self)
    }
    fn tan(&self) -> Dual {
        <Dual as Float>::tan(*self)
    }
    fn csc(&self) -> Dual {
        1.0 / self.sin()
    }
    fn sec(&self) -> Dual {
        1.0 / self.cos()
    }
    fn cot(&self) -> Dual {
        1.0 / self.tan()
    }
    fn asin(&self) -> Dual {
        <Dual as Float>::asin(*self)
    }
    fn acos(&self) -> Dual {
        <Dual as Float>::acos(*self)
    }
    fn atan(&self) -> Dual {
        <Dual as Float>::atan(*self)
    }
    fn atan2(&self, other: &Dual) -> Dual {
        <Dual as Float>::atan2(*self, *other)
    }
    fn acsc(&self) -> Dual {
        (Dual::from_real(1.0) / *self).asin()
    }
    fn asec(&self) -> Dual {
        (Dual::from_real(1.0) / *self).acos()
    }
    fn acot(&self) -> Dual {
        (Dual::from_real(1.0) / *self).atan()
    }
    fn deg2rad(&self) -> Dual {
        *self * Dual::from_real(PI / 180.0)
    }
    fn rad2deg(&self) -> Dual {
        *self * Dual::from_real(180.0 / PI)
    }
    fn sind(&self) -> Dual {
        self.deg2rad().sin()
    }
    fn cosd(&self) -> Dual {
        self.deg2rad().cos()
    }
    fn tand(&self) -> Dual {
        self.deg2rad().tan()
    }
    fn cscd(&self) -> Dual {
        self.deg2rad().csc()
    }
    fn secd(&self) -> Dual {
        self.deg2rad().sec()
    }
    fn cotd(&self) -> Dual {
        self.deg2rad().cot()
    }
    fn asind(&self) -> Dual {
        self.asin().rad2deg()
    }
    fn acosd(&self) -> Dual {
        self.acos().rad2deg()
    }
    fn atand(&self) -> Dual {
        self.atan().rad2deg()
    }
    fn atan2d(&self, other: &Dual) -> Dual {
        self.atan2(other).rad2deg()
    }
    fn acscd(&self) -> Dual {
        self.acsc().rad2deg()
    }
    fn asecd(&self) -> Dual {
        self.asec().rad2deg()
    }
    fn acotd(&self) -> Dual {
        self.acot().rad2deg()
    }
    fn sinh(&self) -> Dual {
        <Dual as Float>::sinh(*self)
    }
    fn cosh(&self) -> Dual {
        <Dual as Float>::cosh(*self)
    }
    fn tanh(&self) -> Dual {
        <Dual as Float>::tanh(*self)
    }
    fn csch(&self) -> Dual {
        1.0 / self.sinh()
    }
    fn sech(&self) -> Dual {
        1.0 / self.cosh()
    }
    fn coth(&self) -> Dual {
        1.0 / self.tanh()
    }
    fn asinh(&self) -> Dual {
        <Dual as Float>::asinh(*self)
    }
    fn acosh(&self) -> Dual {
        <Dual as Float>::acosh(*self)
    }
    fn atanh(&self) -> Dual {
        <Dual as Float>::atanh(*self)
    }
    fn acsch(&self) -> Dual {
        (Dual::from_real(1.0) / *self).asinh()
    }
    fn asech(&self) -> Dual {
        (Dual::from_real(1.0) / *self).acosh()
    }
    fn acoth(&self) -> Dual {
        (Dual::from_real(1.0) / *self).atanh()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numtest::*;
    use std::f64::consts::FRAC_PI_6;

    #[test]
    fn test_csc() {
        assert_equal_to_decimal!(
            Dual::new(FRAC_PI_6, 2.0).csc(),
            Dual::new(2.0, -4.0 * 3.0_f64.sqrt()),
            15
        );
    }

    #[test]
    fn test_sec() {
        assert_equal_to_decimal!(
            Dual::new(FRAC_PI_6, 2.0).sec(),
            Dual::new(2.0 / 3.0_f64.sqrt(), 4.0 / 3.0),
            15
        );
    }

    #[test]
    fn test_cot() {
        assert_equal_to_decimal!(
            Dual::new(FRAC_PI_6, 2.0).cot(),
            Dual::new(3.0_f64.sqrt(), -8.0),
            15
        );
    }

    #[test]
    fn test_acsc() {
        assert_equal_to_decimal!(
            Dual::new(FRAC_PI_6, 2.0).csc().acsc(),
            Dual::new(FRAC_PI_6, 2.0),
            14
        );
    }

    #[test]
    fn test_asec() {
        assert_equal_to_decimal!(
            Dual::new(FRAC_PI_6, 2.0).sec().asec(),
            Dual::new(FRAC_PI_6, 2.0),
            14
        );
    }

    #[test]
    fn test_acot() {
        assert_equal_to_decimal!(
            Dual::new(FRAC_PI_6, 2.0).cot().acot(),
            Dual::new(FRAC_PI_6, 2.0),
            14
        );
    }

    #[test]
    fn test_deg2rad() {
        assert_equal_to_decimal!(
            Dual::new(180.0, 2.0).deg2rad(),
            Dual::new(PI, PI / 90.0),
            15
        );
    }

    #[test]
    fn test_rad2deg() {
        assert_equal_to_decimal!(
            Dual::new(PI, 2.0).rad2deg(),
            Dual::new(180.0, 360.0 / PI),
            14
        );
    }

    #[test]
    fn test_sind() {
        assert_equal_to_decimal!(
            Dual::new(30.0, 2.0).sind(),
            Dual::new(0.5, PI * 3.0_f64.sqrt() / 180.0),
            15
        );
    }

    #[test]
    fn test_cosd() {
        assert_equal_to_decimal!(
            Dual::new(30.0, 2.0).cosd(),
            Dual::new(3.0_f64.sqrt() / 2.0, -PI / 180.0),
            15
        );
    }

    #[test]
    fn test_tand() {
        assert_equal_to_decimal!(
            Dual::new(30.0, 2.0).tand(),
            Dual::new(3.0_f64.sqrt() / 3.0, 4.0 * PI / 270.0),
            15
        );
    }

    #[test]
    fn test_cscd() {
        assert_equal_to_decimal!(
            Dual::new(30.0, 2.0).cscd(),
            Dual::new(2.0, -PI * 3.0_f64.sqrt() / 45.0),
            15
        );
    }

    #[test]
    fn test_secd() {
        assert_equal_to_decimal!(
            Dual::new(30.0, 2.0).secd(),
            Dual::new(2.0 / 3.0_f64.sqrt(), PI / 135.0),
            15
        );
    }

    #[test]
    fn test_cotd() {
        assert_equal_to_decimal!(
            Dual::new(30.0, 2.0).cotd(),
            Dual::new(3.0_f64.sqrt(), -2.0 * PI / 45.0),
            15
        );
    }

    #[test]
    fn test_asind() {
        assert_equal_to_decimal!(
            Dual::new(30.0, 2.0).sind().asind(),
            Dual::new(30.0, 2.0),
            12
        );
    }

    #[test]
    fn test_acosd() {
        assert_equal_to_decimal!(
            Dual::new(30.0, 2.0).cosd().acosd(),
            Dual::new(30.0, 2.0),
            12
        );
    }

    #[test]
    fn test_atand() {
        assert_equal_to_decimal!(
            Dual::new(30.0, 2.0).tand().atand(),
            Dual::new(30.0, 2.0),
            12
        );
    }

    #[test]
    fn test_atan2d() {
        assert_equal_to_decimal!(
            Dual::new(-3.0, 5.0).atan2d(&Dual::new(3.0, 2.0)),
            Dual::new(-45.0, 210.0 / PI),
            14
        );
    }

    #[test]
    fn test_acscd() {
        assert_equal_to_decimal!(
            Dual::new(30.0, 2.0).cscd().acscd(),
            Dual::new(30.0, 2.0),
            11
        );
    }

    #[test]
    fn test_asecd() {
        assert_equal_to_decimal!(
            Dual::new(30.0, 2.0).secd().asecd(),
            Dual::new(30.0, 2.0),
            11
        );
    }

    #[test]
    fn test_acotd() {
        assert_equal_to_decimal!(
            Dual::new(30.0, 2.0).cotd().acotd(),
            Dual::new(30.0, 2.0),
            11
        );
    }

    #[test]
    fn test_csch() {
        assert_equal_to_decimal!(
            Dual::new(1.0, 2.0).csch(),
            Dual::new(
                1.0_f64.sinh().recip(),
                -2.0 * 1.0_f64.cosh() / 1.0_f64.sinh().powi(2)
            ),
            15
        );
    }

    #[test]
    fn test_sech() {
        assert_equal_to_decimal!(
            Dual::new(1.0, 2.0).sech(),
            Dual::new(
                1.0_f64.cosh().recip(),
                -2.0 * 1.0_f64.sinh() / 1.0_f64.cosh().powi(2)
            ),
            15
        );
    }

    #[test]
    fn test_coth() {
        assert_equal_to_decimal!(
            Dual::new(1.0, 2.0).coth(),
            Dual::new(
                1.0_f64.cosh() / 1.0_f64.sinh(),
                -2.0 / 1.0_f64.sinh().powi(2)
            ),
            15
        );
    }

    #[test]
    fn test_acsch() {
        assert_equal_to_decimal!(Dual::new(1.0, 2.0).csch().acsch(), Dual::new(1.0, 2.0), 12);
    }

    #[test]
    fn test_asech() {
        assert_equal_to_decimal!(Dual::new(1.0, 2.0).sech().asech(), Dual::new(1.0, 2.0), 12);
    }

    #[test]
    fn test_acoth() {
        assert_equal_to_decimal!(Dual::new(2.0, 2.0).coth().acoth(), Dual::new(2.0, 2.0), 12);
    }
}
