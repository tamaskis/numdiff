use crate::automatic_differentiation::hyper_dual::hyper_dual::HyperDual;
use num_traits::Float;
use std::f64::consts::PI;
use trig::Trig;

impl Trig for HyperDual {
    fn sin(&self) -> HyperDual {
        <HyperDual as Float>::sin(*self)
    }
    fn cos(&self) -> HyperDual {
        <HyperDual as Float>::cos(*self)
    }
    fn tan(&self) -> HyperDual {
        <HyperDual as Float>::tan(*self)
    }
    fn csc(&self) -> HyperDual {
        1.0 / self.sin()
    }
    fn sec(&self) -> HyperDual {
        1.0 / self.cos()
    }
    fn cot(&self) -> HyperDual {
        (*self).univariate_map(
            |x| x.cos() / x.sin(),                 // cot(x) = cos(x) / sin(x)
            |x| -1.0 / (x.sin().powi(2)),          // cot'(x) = -csc²(x) = -1 / sin²(x)
            |x| 2.0 * x.cos() / (x.sin().powi(3)), // cot"(x) = 2cot(x)csc²(x) = 2cos(x) / sin³(x)
        )
    }
    fn asin(&self) -> HyperDual {
        <HyperDual as Float>::asin(*self)
    }
    fn acos(&self) -> HyperDual {
        <HyperDual as Float>::acos(*self)
    }
    fn atan(&self) -> HyperDual {
        <HyperDual as Float>::atan(*self)
    }
    fn atan2(&self, other: &HyperDual) -> HyperDual {
        <HyperDual as Float>::atan2(*self, *other)
    }
    fn acsc(&self) -> HyperDual {
        (*self).univariate_map(
            |x| (1.0 / x).asin(),                            // f(x) = acsc(x) = asin(1/x)
            |x| -1.0 / (x.abs() * (x.powi(2) - 1.0).sqrt()), // f'(x) = -1 / (|x|√(x² - 1))
            |x| {
                let x2 = x.powi(2);
                let denom = x2 * x.abs() * (x2 - 1.0).powf(1.5);
                (2.0 * x2 - 1.0) / denom
            }, // f"(x) = (2x² - 1) / (x²|x|(x² - 1)³ᐟ²)
        )
    }
    fn asec(&self) -> HyperDual {
        (*self).univariate_map(
            |x| (1.0 / x).acos(),                           // f(x) = asec(x) = acos(1/x)
            |x| 1.0 / (x.abs() * (x.powi(2) - 1.0).sqrt()), // f'(x) = 1 / (|x|√(x² - 1))
            |x| {
                let x2 = x.powi(2);
                let denom = x2 * x.abs() * (x2 - 1.0).powf(1.5);
                -(2.0 * x2 - 1.0) / denom
            }, // f"(x) = -(2x² - 1) / (x²|x|(x² - 1)³ᐟ²)
        )
    }
    fn acot(&self) -> HyperDual {
        HyperDual::from_real(PI / 2.0) - self.atan() // acot(x) = π/2 - atan(x)
    }
    fn deg2rad(&self) -> HyperDual {
        *self * HyperDual::from_real(PI / 180.0)
    }
    fn rad2deg(&self) -> HyperDual {
        *self * HyperDual::from_real(180.0 / PI)
    }
    fn sind(&self) -> HyperDual {
        self.deg2rad().sin()
    }
    fn cosd(&self) -> HyperDual {
        self.deg2rad().cos()
    }
    fn tand(&self) -> HyperDual {
        self.deg2rad().tan()
    }
    fn cscd(&self) -> HyperDual {
        self.deg2rad().csc()
    }
    fn secd(&self) -> HyperDual {
        self.deg2rad().sec()
    }
    fn cotd(&self) -> HyperDual {
        self.deg2rad().cot()
    }
    fn asind(&self) -> HyperDual {
        self.asin().rad2deg()
    }
    fn acosd(&self) -> HyperDual {
        self.acos().rad2deg()
    }
    fn atand(&self) -> HyperDual {
        self.atan().rad2deg()
    }
    fn atan2d(&self, other: &HyperDual) -> HyperDual {
        self.atan2(other).rad2deg()
    }
    fn acscd(&self) -> HyperDual {
        self.acsc().rad2deg()
    }
    fn asecd(&self) -> HyperDual {
        self.asec().rad2deg()
    }
    fn acotd(&self) -> HyperDual {
        self.acot().rad2deg()
    }
    fn sinh(&self) -> HyperDual {
        <HyperDual as Float>::sinh(*self)
    }
    fn cosh(&self) -> HyperDual {
        <HyperDual as Float>::cosh(*self)
    }
    fn tanh(&self) -> HyperDual {
        <HyperDual as Float>::tanh(*self)
    }
    fn csch(&self) -> HyperDual {
        1.0 / self.sinh()
    }
    fn sech(&self) -> HyperDual {
        (*self).univariate_map(
            |x| 1.0 / x.cosh(),         // f(x) = sech(x) = 1 / cosh(x)
            |x| -(x.tanh() / x.cosh()), // f'(x) = -sech(x)tanh(x) = -tanh(x) / cosh(x)
            |x| {
                let sech_x = 1.0 / x.cosh();
                let tanh_x = x.tanh();
                sech_x * (sech_x.powi(2) - tanh_x.powi(2))
            }, // sech(x)(sech²(x) - tanh²(x))
        )
    }
    fn coth(&self) -> HyperDual {
        1.0 / self.tanh()
    }
    fn asinh(&self) -> HyperDual {
        <HyperDual as Float>::asinh(*self)
    }
    fn acosh(&self) -> HyperDual {
        <HyperDual as Float>::acosh(*self)
    }
    fn atanh(&self) -> HyperDual {
        <HyperDual as Float>::atanh(*self)
    }
    fn acsch(&self) -> HyperDual {
        (*self).univariate_map(
            |x| (1.0 / x).asinh(),                           // f(x) = acsch(x) = asinh(1/x)
            |x| -1.0 / (x.abs() * (x.powi(2) + 1.0).sqrt()), // f'(x) = -1 / (|x|√(x² + 1))
            |x| {
                let x2 = x.powi(2);
                let denom = x2 * x.abs() * (x2 + 1.0).powf(1.5);
                (2.0 * x2 + 1.0) / denom
            }, // f"(x) = (2x² + 1) / (x²|x|(x² + 1)³ᐟ²)
        )
    }
    fn asech(&self) -> HyperDual {
        (*self).univariate_map(
            |x| (1.0 / x).acosh(),                     // f(x) = asech(x) = acosh(1/x)
            |x| -1.0 / (x * (1.0 - x.powi(2)).sqrt()), // f'(x) = -1 / (x√(1 - x²))
            |x| {
                let x2 = x.powi(2);
                let denom = x2 * (1.0 - x2).powf(1.5);
                -(2.0 * x2 - 1.0) / denom
            }, // f"(x) = -(2x² - 1) / (x²(1 - x²)³ᐟ²)
        )
    }
    fn acoth(&self) -> HyperDual {
        // acoth(x) = (1/2)ln(|x+1|/|x-1|) = (1/2)ln|(x+1)/(x-1)|
        let one = HyperDual::from_real(1.0);
        let half = HyperDual::from_real(0.5);
        half * ((*self + one) / (*self - one)).ln()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::assert_hyper_dual_close;
    use std::f64::consts::FRAC_PI_6;
    use std::f64::consts::PI;

    #[test]
    fn test_csc() {
        assert_hyper_dual_close(
            HyperDual::new(FRAC_PI_6, 2.0, 1.5, -0.75).csc(),
            HyperDual::new(
                2.0,
                -4.0 * 3.0_f64.sqrt(),
                -5.196152422706632,
                44.598076211353316,
            ),
            15,
        );
    }

    #[test]
    fn test_sec() {
        assert_hyper_dual_close(
            HyperDual::new(FRAC_PI_6, 2.0, 1.5, -0.75).sec(),
            HyperDual::new(2.0 / 3.0_f64.sqrt(), 4.0 / 3.0, 1.0, 5.273502691896258),
            15,
        );
    }

    #[test]
    fn test_cot() {
        assert_hyper_dual_close(
            HyperDual::new(FRAC_PI_6, 2.0, 1.5, -0.75).cot(),
            HyperDual::new(3.0_f64.sqrt(), -8.0, -6.0, 44.569219381653035),
            12, // Reduced precision due to rounding from exact values
        );
    }

    #[test]
    fn test_acsc() {
        assert_hyper_dual_close(
            HyperDual::new(FRAC_PI_6, 2.0, 1.5, -0.75).csc().acsc(),
            HyperDual::new(FRAC_PI_6, 2.0, 1.5, -6.812177826491073),
            14,
        );
    }

    #[test]
    fn test_asec() {
        assert_hyper_dual_close(
            HyperDual::new(FRAC_PI_6, 2.0, 1.5, -0.75).sec().asec(),
            HyperDual::new(0.5235987755982991, 2.0, 1.5, 0.4102540378443935),
            13,
        );
    }

    #[test]
    fn test_acot() {
        assert_hyper_dual_close(
            HyperDual::new(FRAC_PI_6, 2.0, 1.5, -0.75).cot().acot(),
            HyperDual::new(FRAC_PI_6, 2.0, 1.5, -0.75),
            13,
        );
    }

    #[test]
    fn test_deg2rad() {
        assert_hyper_dual_close(
            HyperDual::new(180.0, 2.0, 1.5, -0.75).deg2rad(),
            HyperDual::new(PI, PI / 90.0, 0.026179938779914945, -0.013089969389957472),
            15,
        );
    }

    #[test]
    fn test_rad2deg() {
        assert_hyper_dual_close(
            HyperDual::new(PI, 2.0, 1.5, -0.75).rad2deg(),
            HyperDual::new(180.0, 360.0 / PI, 85.94366926962348, -42.97183463481174),
            14,
        );
    }

    #[test]
    fn test_sind() {
        assert_hyper_dual_close(
            HyperDual::new(30.0, 2.0, 1.5, -0.75).sind(),
            HyperDual::new(
                0.5,
                PI * 3.0_f64.sqrt() / 180.0,
                0.022672492052927727,
                -0.011793172156143927,
            ),
            14,
        );
    }

    #[test]
    fn test_cosd() {
        assert_hyper_dual_close(
            HyperDual::new(30.0, 2.0, 1.5, -0.75).cosd(),
            HyperDual::new(
                3.0_f64.sqrt() / 2.0,
                -PI / 180.0,
                -0.01308996938995747,
                0.005753565423067061,
            ),
            15,
        );
    }

    #[test]
    fn test_tand() {
        assert_hyper_dual_close(
            HyperDual::new(30.0, 2.0, 1.5, -0.75).tand(),
            HyperDual::new(
                3.0_f64.sqrt() / 3.0,
                4.0 * PI / 270.0,
                0.034906585039886584,
                -0.01604632492543365,
            ),
            15,
        );
    }

    #[test]
    fn test_cscd() {
        assert_hyper_dual_close(
            HyperDual::new(30.0, 2.0, 1.5, -0.75).cscd(),
            HyperDual::new(
                2.0,
                -PI * 3.0_f64.sqrt() / 45.0,
                -0.09068996821171092,
                0.05813891573689724,
            ),
            14,
        );
    }

    #[test]
    fn test_secd() {
        assert_hyper_dual_close(
            HyperDual::new(30.0, 2.0, 1.5, -0.75).secd(),
            HyperDual::new(
                1.1547005383792515,
                PI / 135.0,
                0.017453292519943292,
                -0.006967936766834592,
            ),
            15,
        );
    }

    #[test]
    fn test_cotd() {
        assert_hyper_dual_close(
            HyperDual::new(30.0, 2.0, 1.5, -0.75).cotd(),
            HyperDual::new(
                1.7320508075688776,
                -2.0 * PI / 45.0,
                -0.10471975511965981,
                0.06502258591041671,
            ),
            15,
        );
    }

    #[test]
    fn test_asind() {
        assert_hyper_dual_close(
            HyperDual::new(30.0, 2.0, 1.5, -0.75).sind().asind(),
            HyperDual::new(30.0, 2.0, 1.5, -0.75),
            11,
        );
    }

    #[test]
    fn test_acosd() {
        assert_hyper_dual_close(
            HyperDual::new(30.0, 2.0, 1.5, -0.75).cosd().acosd(),
            HyperDual::new(30.0, 2.0, 1.5, -0.75),
            11,
        );
    }

    #[test]
    fn test_atand() {
        assert_hyper_dual_close(
            HyperDual::new(30.0, 2.0, 1.5, -0.75).tand().atand(),
            HyperDual::new(30.0, 2.0, 1.5, -0.75),
            11,
        );
    }

    #[test]
    fn test_atan2d() {
        assert_hyper_dual_close(
            HyperDual::new(-3.0, 5.0, 1.5, -0.75).atan2d(&HyperDual::new(3.0, 2.0, 1.5, -0.75)),
            HyperDual::new(-45.0, 210.0 / PI, 28.64788975654116, 1.5902773407317584e-15),
            14,
        );
    }

    #[test]
    fn test_acscd() {
        assert_hyper_dual_close(
            HyperDual::new(30.0, 2.0, 1.5, -0.75).cscd().acscd(),
            HyperDual::new(30.0, 2.0, 1.5, -0.8558049629136627),
            10,
        );
    }

    #[test]
    fn test_asecd() {
        assert_hyper_dual_close(
            HyperDual::new(30.0, 2.0, 1.5, -0.75).secd().asecd(),
            HyperDual::new(30.0, 2.0, 1.5, -0.7297497468800564),
            10,
        );
    }

    #[test]
    fn test_acotd() {
        assert_hyper_dual_close(
            HyperDual::new(30.0, 2.0, 1.5, -0.75).cotd().acotd(),
            HyperDual::new(30.0, 2.0, 1.5, -0.75),
            10,
        );
    }

    #[test]
    fn test_csch() {
        assert_hyper_dual_close(
            HyperDual::new(1.0, 2.0, 1.5, -0.75).csch(),
            HyperDual::new(
                1.0_f64.sinh().recip(),
                -2.0 * 1.0_f64.cosh() / 1.0_f64.sinh().powi(2),
                -1.6759282911739115,
                7.087421689980765,
            ),
            15,
        );
    }

    #[test]
    fn test_sech() {
        assert_hyper_dual_close(
            HyperDual::new(1.0, 2.0, 1.5, -0.75).sech(),
            HyperDual::new(
                1.0_f64.cosh().recip(),
                -2.0 * 1.0_f64.sinh() / 1.0_f64.cosh().powi(2),
                -0.7403315213468595,
                0.05899994115465074,
            ),
            15,
        );
    }

    #[test]
    fn test_coth() {
        assert_hyper_dual_close(
            HyperDual::new(1.0, 2.0, 1.5, -0.75).coth(),
            HyperDual::new(
                1.0_f64.cosh() / 1.0_f64.sinh(),
                -2.0 / 1.0_f64.sinh().powi(2),
                -1.0860924914494658,
                6.247357304080852,
            ),
            15,
        );
    }

    #[test]
    fn test_acsch() {
        assert_hyper_dual_close(
            HyperDual::new(1.0, 2.0, 1.5, -0.75).csch().acsch(),
            HyperDual::new(1.0, 2.0, 1.5, 0.2299754803086938),
            11,
        );
    }

    #[test]
    fn test_asech() {
        assert_hyper_dual_close(
            HyperDual::new(1.0, 2.0, 1.5, -0.75).sech().asech(),
            HyperDual::new(1.0, 2.0, 1.5, 0.5109181584731899),
            11,
        );
    }

    #[test]
    fn test_acoth() {
        assert_hyper_dual_close(
            HyperDual::new(2.0, 2.0, 1.5, -0.75).coth().acoth(),
            HyperDual::new(2.0, 2.0, 1.5, -0.75),
            11,
        );
    }
}
