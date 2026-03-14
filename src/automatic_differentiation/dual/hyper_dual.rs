use num_traits::{Float, Num, NumCast, One, ToPrimitive, Zero};
use std::cmp::Ordering;
use std::f64::consts::{LN_2, LN_10};
use std::num::FpCategory;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};

#[cfg(feature = "trig")]
use std::f64::consts::PI;

#[cfg(feature = "trig")]
use trig::Trig;

/// Second-order hyper-dual number.
///
/// A hyper-dual number is represented as
///
/// `a + (b)ε₁ + (c)ε₂ + (d)ε₁ε₂`
///
/// where `ε₁² = ε₂² = 0` and `ε₁ε₂ = ε₂ε₁`.
#[derive(Debug, Clone, Copy)]
pub struct HyperDual {
    /// Real part.
    a: f64,

    /// Coefficient of `ε₁`.
    b: f64,

    /// Coefficient of `ε₂`.
    c: f64,

    /// Coefficient of `ε₁ε₂`.
    d: f64,
}

impl HyperDual {
    /// Constructor.
    ///
    /// # Arguments
    ///
    /// * `a` - Real part.
    /// * `b` - Coefficient of `ε₁`.
    /// * `c` - Coefficient of `ε₂`.
    /// * `d` - Coefficient of `ε₁ε₂`.
    ///
    /// # Returns
    ///
    /// Hyper-dual number, `a + (b)ε₁ + (c)ε₂ + (d)ε₁ε₂`.
    #[must_use]
    pub fn new(a: f64, b: f64, c: f64, d: f64) -> Self {
        Self { a, b, c, d }
    }

    /// Get the real part.
    #[must_use]
    pub fn get_a(self) -> f64 {
        self.a
    }

    /// Get the `ε₁` coefficient.
    #[must_use]
    pub fn get_b(self) -> f64 {
        self.b
    }

    /// Get the `ε₂` coefficient.
    #[must_use]
    pub fn get_c(self) -> f64 {
        self.c
    }

    /// Get the `ε₁ε₂` coefficient.
    #[must_use]
    pub fn get_d(self) -> f64 {
        self.d
    }

    /// Construct a purely real hyper-dual number.
    ///
    /// # Arguments
    ///
    /// * `real` - Real part.
    ///
    /// # Returns
    ///
    /// Hyper-dual number, `real + 0ε₁ + 0ε₂ + 0ε₁ε₂`.
    ///
    /// # Example
    ///
    /// ```
    /// use numdiff::HyperDual;
    ///
    /// let num = HyperDual::from_real(2.5);
    /// assert_eq!(num.get_a(), 2.5);
    /// assert_eq!(num.get_b(), 0.0);
    /// assert_eq!(num.get_c(), 0.0);
    /// assert_eq!(num.get_d(), 0.0);
    /// ```
    #[must_use]
    pub fn from_real(real: f64) -> Self {
        Self::new(real, 0.0, 0.0, 0.0)
    }

    /// Apply a univariate, scalar-valued function to this hyper-dual number.
    ///
    /// This helper centralizes the second-order hyper-dual chain rule for univariate, scalar-valued
    /// functions of the form `f(a)`, where `f`, `f'`, and `f"` are provided separately.
    ///
    /// For ordinary arithmetic expressions, the chain rule is already induced by the overloaded
    /// implementations of `Add`, `Sub`, `Mul`, `Div`, and related operators. That is enough for
    /// expressions built entirely from those primitive operations. However, intrinsic univariate
    /// functions such as `sin`, `cos`, `exp`, `ln`, `sqrt`, and `atan` are not expressed in this
    /// file in terms of those operators alone, so each such method still needs an explicit
    /// hyper-dual propagation rule.
    ///
    /// Without this helper, every univariate `Float` method would need to manually repeat the same
    /// second-order pattern: the real part uses `f(a)`, both first-order components scale by
    ///  `f'(a)`, and the mixed `ε₁ε₂` component combines both `f"(a)` and `f'(a)`. This method
    /// factors that pattern into one place, which keeps the implementations of methods such as
    /// `sin`, `cos`, `exp`, and `ln` short and makes the second-order logic less error-prone.
    ///
    /// # Type Parameters
    ///
    /// * `F` - Type of the univariate, scalar-valued function `f(x)`.
    /// * `DF` - Type of the first derivative `f'(x) = df/dx`.
    /// * `D2F` - Type of the second derivative `f"(x) = d²f/dx²`.
    ///
    /// # Arguments
    ///
    /// * `f` - Univariate, scalar-valued function, `f(x)`.
    /// * `df` - First derivative of `f` with respect to `x`, `df/dx`.
    /// * `d2f` - Second derivative of `f` with respect to `x`, `d²f/dx²`.
    ///
    /// # Returns
    ///
    /// Hyper-dual number obtained by applying `f` to `self` using the second-order univariate
    /// hyper-dual chain rule.
    ///
    /// If this dual number is
    ///
    /// ```text
    /// a + bε₁ + cε₂ + dε₁ε₂
    /// ```
    ///
    /// then this function (`univariate_map`) returns:
    ///
    /// ```text
    /// f(a) + f'(a)bε₁ + f'(a)cε₂ + [f"(a)bc + f'(a)d]ε₁ε₂
    /// ```
    ///
    /// where `f'` and `f"` are evaluated at `a`.
    fn univariate_map<F, DF, D2F>(self, f: F, df: DF, d2f: D2F) -> HyperDual
    where
        F: Fn(f64) -> f64,
        DF: Fn(f64) -> f64,
        D2F: Fn(f64) -> f64,
    {
        let df_a = df(self.a);
        HyperDual::new(
            f(self.a),
            df_a * self.b,
            df_a * self.c,
            d2f(self.a) * self.b * self.c + df_a * self.d,
        )
    }

    /// Apply a bivariate, scalar-valued function to two hyper-dual numbers.
    ///
    /// This helper centralizes the second-order hyper-dual chain rule for bivariate, scalar-valued
    /// functions of the form `g(x, y)`, where the function, its first partial derivatives, and its
    /// second partial derivatives are supplied separately.
    ///
    /// As with [`Self::univariate_map`], overloaded arithmetic operators already propagate
    /// derivatives for expressions assembled directly from primitive operations. The purpose of
    /// this helper is to avoid re-deriving the full second-order propagation formula for genuinely
    /// bivariate intrinsic functions such as `atan2` and `hypot`, whose implementations naturally
    /// depend on `g(x, y)`, `∂g/∂x`, `∂g/∂y`, `∂²g/∂x²`, `∂²g/∂x∂y`, and `∂²g/∂y²`.
    ///
    /// In particular, the first-order `ε₁` and `ε₂` components are determined by the ordinary
    /// multivariate chain rule, while the mixed `ε₁ε₂` component additionally depends on the second
    /// partial derivatives `∂²g/∂x²`, `∂²g/∂x∂y`, and `∂²g/∂y²`. This helper keeps that formula in
    /// one place so the individual `Float` methods can focus on just providing the corresponding
    /// real derivatives.
    ///
    /// # Type Parameters
    ///
    /// * `G` - Type of the bivariate, scalar-valued function `g(x, y)`.
    /// * `GX` - Type of the partial derivative `gₓ(x, y) = ∂g/∂x`.
    /// * `GY` - Type of the partial derivative `gᵧ(x, y) = ∂g/∂y`.
    /// * `GXX` - Type of the second partial derivative `gₓₓ(x, y) = ∂²g/∂x²`.
    /// * `GXY` - Type of the mixed second partial derivative `gₓᵧ(x, y) = ∂²g/∂x∂y`.
    /// * `GYY` - Type of the second partial derivative `gᵧᵧ(x, y) = ∂²g/∂y²`.
    ///
    /// # Arguments
    ///
    /// * `other` - Second hyper-dual input.
    /// * `g` - Bivariate, scalar-valued function, `g(x, y)`.
    /// * `gx` - Partial derivative of `g` with respect to the first argument, `gₓ(x, y) = ∂g/∂x`.
    /// * `gy` - Partial derivative of `g` with respect to the second argument, `gᵧ(x, y) = ∂g/∂y`.
    /// * `gxx` - Second partial derivative of `g` with respect to the first argument twice,
    ///   `gₓₓ(x, y) = ∂²g/∂x²`.
    /// * `gxy` - Mixed second partial derivative of `g`, `gₓᵧ(x, y) = ∂²g/∂x∂y`.
    /// * `gyy` - Second partial derivative of `g` with respect to the second argument twice,
    ///   `gᵧᵧ(x, y) = ∂²g/∂y²`.
    ///
    /// # Returns
    ///
    /// Hyper-dual number obtained by applying `g` to `self` and `other` using the second-order
    /// multivariate hyper-dual chain rule.
    ///
    /// If `self` is the hyper-dual number
    ///
    /// ```text
    /// a + bε₁ + cε₂ + dε₁ε₂
    /// ```
    ///
    /// and `other` is the hyper-dual number
    ///
    /// ```text
    /// p + qε₁ + rε₂ + sε₁ε₂
    /// ```
    ///
    /// then this function (`bivariate_map`) returns:
    ///
    /// ```text
    /// g(a, p) + (gₓb + gᵧq)ε₁ + (gₓc + gᵧr)ε₂ + [gₓd + gᵧs + gₓₓbc + gₓᵧ(br + cq) + gᵧᵧqr]ε₁ε₂
    /// ```
    ///
    /// where all derivatives of `g` are evaluated at the point `(a, p)`.
    #[allow(
        clippy::too_many_arguments,
        clippy::many_single_char_names,
        clippy::similar_names
    )]
    fn bivariate_map<G, GX, GY, GXX, GXY, GYY>(
        self,
        other: HyperDual,
        g: G,
        gx: GX,
        gy: GY,
        gxx: GXX,
        gxy: GXY,
        gyy: GYY,
    ) -> HyperDual
    where
        G: Fn(f64, f64) -> f64,
        GX: Fn(f64, f64) -> f64,
        GY: Fn(f64, f64) -> f64,
        GXX: Fn(f64, f64) -> f64,
        GXY: Fn(f64, f64) -> f64,
        GYY: Fn(f64, f64) -> f64,
    {
        // Extract the components of this hyper-dual number.
        let a = self.a;
        let b = self.b;
        let c = self.c;
        let d = self.d;

        // Extract the components of the other hyper-dual number.
        let p = other.a;
        let q = other.b;
        let r = other.c;
        let s = other.d;

        // Compute the partial derivatives of `g` at the point `(a, p)`.
        let gx_ap = gx(a, p);
        let gy_ap = gy(a, p);

        // Apply the second-order multivariate hyper-dual chain rule.
        HyperDual::new(
            g(a, p),
            gx_ap * b + gy_ap * q,
            gx_ap * c + gy_ap * r,
            gx_ap * d
                + gy_ap * s
                + gxx(a, p) * b * c
                + gxy(a, p) * (b * r + c * q)
                + gyy(a, p) * q * r,
        )
    }
}

// --------------------------------
// Implementing num_traits::NumOps.
// --------------------------------

// HyperDual + HyperDual.
impl Add for HyperDual {
    type Output = HyperDual;
    fn add(self, rhs: HyperDual) -> HyperDual {
        HyperDual::new(
            self.a + rhs.a,
            self.b + rhs.b,
            self.c + rhs.c,
            self.d + rhs.d,
        )
    }
}

// HyperDual - HyperDual.
impl Sub for HyperDual {
    type Output = HyperDual;
    fn sub(self, rhs: HyperDual) -> HyperDual {
        HyperDual::new(
            self.a - rhs.a,
            self.b - rhs.b,
            self.c - rhs.c,
            self.d - rhs.d,
        )
    }
}

// HyperDual * HyperDual.
impl Mul for HyperDual {
    type Output = HyperDual;
    fn mul(self, rhs: HyperDual) -> HyperDual {
        HyperDual::new(
            self.a * rhs.a,
            self.b * rhs.a + self.a * rhs.b,
            self.c * rhs.a + self.a * rhs.c,
            self.d * rhs.a + self.b * rhs.c + self.c * rhs.b + self.a * rhs.d,
        )
    }
}

// HyperDual / HyperDual.
impl Div for HyperDual {
    type Output = HyperDual;
    fn div(self, rhs: HyperDual) -> HyperDual {
        self.bivariate_map(
            rhs,
            |x, y| x / y,
            |_, y| 1.0 / y,
            |x, y| -x / y.powi(2),
            |_, _| 0.0,
            |_, y| -1.0 / y.powi(2),
            |x, y| 2.0 * x / y.powi(3),
        )
    }
}

// Remainder of HyperDual / HyperDual.
impl Rem for HyperDual {
    type Output = HyperDual;
    fn rem(self, rhs: HyperDual) -> HyperDual {
        let q = (self.a / rhs.a).trunc();
        HyperDual::new(
            self.a % rhs.a,
            self.b - q * rhs.b,
            self.c - q * rhs.c,
            self.d - q * rhs.d,
        )
    }
}

// -----------------------------
// Implementing num_traits::Num.
// -----------------------------

impl PartialEq for HyperDual {
    fn eq(&self, other: &Self) -> bool {
        self.a == other.a && self.b == other.b && self.c == other.c && self.d == other.d
    }
}

impl Zero for HyperDual {
    fn zero() -> Self {
        HyperDual::new(0.0, 0.0, 0.0, 0.0)
    }
    fn is_zero(&self) -> bool {
        self.a.is_zero() && self.b.is_zero() && self.c.is_zero() && self.d.is_zero()
    }
}

impl One for HyperDual {
    fn one() -> Self {
        HyperDual::new(1.0, 0.0, 0.0, 0.0)
    }
}

impl Num for HyperDual {
    type FromStrRadixErr = <f64 as Num>::FromStrRadixErr;
    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        f64::from_str_radix(str, radix).map(HyperDual::from_real)
    }
}

// -------------------------------------
// Implementing num_traits::ToPrimitive.
// -------------------------------------

impl ToPrimitive for HyperDual {
    fn to_i64(&self) -> Option<i64> {
        self.a.to_i64()
    }
    fn to_u64(&self) -> Option<u64> {
        self.a.to_u64()
    }
    fn to_f64(&self) -> Option<f64> {
        Some(self.a)
    }
}

// ---------------------------------
// Implementing num_traits::NumCast.
// ---------------------------------

impl NumCast for HyperDual {
    fn from<T: ToPrimitive>(n: T) -> Option<Self> {
        n.to_f64().map(HyperDual::from_real)
    }
}

// -------------------------------
// Implementing num_traits::Float.
// -------------------------------

// Only perform comparisons on the real part.
impl PartialOrd for HyperDual {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.a.partial_cmp(&other.a)
    }
}

impl Neg for HyperDual {
    type Output = HyperDual;
    fn neg(self) -> HyperDual {
        HyperDual::new(-self.a, -self.b, -self.c, -self.d)
    }
}

impl Float for HyperDual {
    fn nan() -> Self {
        HyperDual::new(f64::nan(), f64::nan(), f64::nan(), f64::nan())
    }

    fn infinity() -> Self {
        HyperDual::new(
            f64::infinity(),
            f64::infinity(),
            f64::infinity(),
            f64::infinity(),
        )
    }

    fn neg_infinity() -> Self {
        HyperDual::new(
            f64::neg_infinity(),
            f64::neg_infinity(),
            f64::neg_infinity(),
            f64::neg_infinity(),
        )
    }

    fn neg_zero() -> Self {
        HyperDual::new(
            f64::neg_zero(),
            f64::neg_zero(),
            f64::neg_zero(),
            f64::neg_zero(),
        )
    }

    fn min_value() -> Self {
        HyperDual::new(
            f64::min_value(),
            f64::min_value(),
            f64::min_value(),
            f64::min_value(),
        )
    }

    fn min_positive_value() -> Self {
        HyperDual::new(
            f64::min_positive_value(),
            f64::min_positive_value(),
            f64::min_positive_value(),
            f64::min_positive_value(),
        )
    }

    fn max_value() -> Self {
        HyperDual::new(
            f64::max_value(),
            f64::max_value(),
            f64::max_value(),
            f64::max_value(),
        )
    }

    fn is_nan(self) -> bool {
        self.a.is_nan()
    }

    fn is_infinite(self) -> bool {
        self.a.is_infinite()
    }

    fn is_finite(self) -> bool {
        self.a.is_finite()
    }

    fn is_normal(self) -> bool {
        self.a.is_normal()
    }

    fn classify(self) -> FpCategory {
        self.a.classify()
    }

    fn floor(self) -> Self {
        HyperDual::new(self.a.floor(), 0.0, 0.0, 0.0)
    }

    fn ceil(self) -> Self {
        HyperDual::new(self.a.ceil(), 0.0, 0.0, 0.0)
    }

    fn round(self) -> Self {
        HyperDual::new(self.a.round(), 0.0, 0.0, 0.0)
    }

    fn trunc(self) -> Self {
        HyperDual::new(self.a.trunc(), 0.0, 0.0, 0.0)
    }

    fn fract(self) -> Self {
        HyperDual::new(self.a.fract(), self.b, self.c, self.d)
    }

    fn abs(self) -> Self {
        let sign = self.a.signum();
        HyperDual::new(self.a.abs(), self.b * sign, self.c * sign, self.d * sign)
    }

    fn signum(self) -> Self {
        HyperDual::from_real(self.a.signum())
    }

    fn is_sign_positive(self) -> bool {
        self.a.is_sign_positive()
    }

    fn is_sign_negative(self) -> bool {
        self.a.is_sign_negative()
    }

    fn mul_add(self, a: Self, b: Self) -> Self {
        self * a + b
    }

    fn recip(self) -> Self {
        self.univariate_map(f64::recip, |x| -1.0 / x.powi(2), |x| 2.0 / x.powi(3))
    }

    fn powi(self, n: i32) -> Self {
        if n == 0 {
            HyperDual::one()
        } else {
            self.univariate_map(
                |x| x.powi(n),
                |x| <f64 as From<i32>>::from(n) * x.powi(n - 1),
                |x| <f64 as From<i32>>::from(n) * <f64 as From<i32>>::from(n - 1) * x.powi(n - 2),
            )
        }
    }

    // Numerically-stable version.
    fn powf(self, n: Self) -> Self {
        (self.ln() * n).exp()
    }

    fn sqrt(self) -> Self {
        self.univariate_map(
            f64::sqrt,
            |x| 1.0 / (2.0 * x.sqrt()),
            |x| -1.0 / (4.0 * x.powf(1.5)),
        )
    }

    fn exp(self) -> Self {
        self.univariate_map(f64::exp, f64::exp, f64::exp)
    }

    fn exp2(self) -> Self {
        self.univariate_map(f64::exp2, |x| LN_2 * x.exp2(), |x| LN_2.powi(2) * x.exp2())
    }

    fn ln(self) -> Self {
        self.univariate_map(f64::ln, |x| 1.0 / x, |x| -1.0 / x.powi(2))
    }

    fn log(self, base: Self) -> Self {
        self.ln() / base.ln()
    }

    fn log2(self) -> Self {
        self.univariate_map(
            f64::log2,
            |x| 1.0 / (x * LN_2),
            |x| -1.0 / (x.powi(2) * LN_2),
        )
    }

    fn log10(self) -> Self {
        self.univariate_map(
            f64::log10,
            |x| 1.0 / (x * LN_10),
            |x| -1.0 / (x.powi(2) * LN_10),
        )
    }

    fn max(self, other: Self) -> Self {
        if self.a > other.a { self } else { other }
    }

    fn min(self, other: Self) -> Self {
        if self.a < other.a { self } else { other }
    }

    #[allow(deprecated)]
    fn abs_sub(self, other: Self) -> Self {
        if self.a > other.a {
            self - other
        } else {
            Self::zero()
        }
    }

    fn cbrt(self) -> Self {
        self.univariate_map(
            f64::cbrt,
            |x| 1.0 / (3.0 * x.cbrt().powi(2)),
            |x| -2.0 / (9.0 * x.cbrt().powi(5)),
        )
    }

    fn hypot(self, other: Self) -> Self {
        self.bivariate_map(
            other,
            |x, y| (x.powi(2) + y.powi(2)).sqrt(),
            |x, y| {
                let r = (x.powi(2) + y.powi(2)).sqrt();
                x / r
            },
            |x, y| {
                let r = (x.powi(2) + y.powi(2)).sqrt();
                y / r
            },
            |x, y| {
                let r3 = (x.powi(2) + y.powi(2)).powf(1.5);
                y.powi(2) / r3
            },
            |x, y| {
                let r3 = (x.powi(2) + y.powi(2)).powf(1.5);
                -x * y / r3
            },
            |x, y| {
                let r3 = (x.powi(2) + y.powi(2)).powf(1.5);
                x.powi(2) / r3
            },
        )
    }

    fn sin(self) -> Self {
        self.univariate_map(f64::sin, f64::cos, |x| -x.sin())
    }

    fn cos(self) -> Self {
        self.univariate_map(f64::cos, |x| -x.sin(), |x| -x.cos())
    }

    fn tan(self) -> Self {
        self.univariate_map(
            f64::tan,
            |x| 1.0 / x.cos().powi(2),
            |x| 2.0 * x.tan() / x.cos().powi(2),
        )
    }

    fn asin(self) -> Self {
        self.univariate_map(
            f64::asin,
            |x| 1.0 / (1.0 - x.powi(2)).sqrt(),
            |x| x / (1.0 - x.powi(2)).powf(1.5),
        )
    }

    fn acos(self) -> Self {
        self.univariate_map(
            f64::acos,
            |x| -1.0 / (1.0 - x.powi(2)).sqrt(),
            |x| -x / (1.0 - x.powi(2)).powf(1.5),
        )
    }

    fn atan(self) -> Self {
        self.univariate_map(
            f64::atan,
            |x| 1.0 / (1.0 + x.powi(2)),
            |x| -2.0 * x / (1.0 + x.powi(2)).powi(2),
        )
    }

    fn atan2(self, other: Self) -> Self {
        self.bivariate_map(
            other,
            f64::atan2,
            |x, y| y / (x.powi(2) + y.powi(2)),
            |x, y| -x / (x.powi(2) + y.powi(2)),
            |x, y| -2.0 * x * y / (x.powi(2) + y.powi(2)).powi(2),
            |x, y| (x.powi(2) - y.powi(2)) / (x.powi(2) + y.powi(2)).powi(2),
            |x, y| 2.0 * x * y / (x.powi(2) + y.powi(2)).powi(2),
        )
    }

    fn sin_cos(self) -> (Self, Self) {
        (self.sin(), self.cos())
    }

    fn exp_m1(self) -> Self {
        self.univariate_map(f64::exp_m1, f64::exp, f64::exp)
    }

    fn ln_1p(self) -> Self {
        self.univariate_map(
            f64::ln_1p,
            |x| 1.0 / (1.0 + x),
            |x| -1.0 / (1.0 + x).powi(2),
        )
    }

    fn sinh(self) -> Self {
        self.univariate_map(f64::sinh, f64::cosh, f64::sinh)
    }

    fn cosh(self) -> Self {
        self.univariate_map(f64::cosh, f64::sinh, f64::cosh)
    }

    fn tanh(self) -> Self {
        self.univariate_map(
            f64::tanh,
            |x| 1.0 - x.tanh().powi(2),
            |x| -2.0 * x.tanh() * (1.0 - x.tanh().powi(2)),
        )
    }

    fn asinh(self) -> Self {
        self.univariate_map(
            f64::asinh,
            |x| 1.0 / (x.powi(2) + 1.0).sqrt(),
            |x| -x / (x.powi(2) + 1.0).powf(1.5),
        )
    }

    fn acosh(self) -> Self {
        self.univariate_map(
            f64::acosh,
            |x| 1.0 / (x.powi(2) - 1.0).sqrt(),
            |x| -x / (x.powi(2) - 1.0).powf(1.5),
        )
    }

    fn atanh(self) -> Self {
        self.univariate_map(
            f64::atanh,
            |x| 1.0 / (1.0 - x.powi(2)),
            |x| 2.0 * x / (1.0 - x.powi(2)).powi(2),
        )
    }

    // This method is really irrelevant, but we need to implement it anyway to satisfy the Float
    // trait.
    fn integer_decode(self) -> (u64, i16, i8) {
        self.a.integer_decode()
    }
}

// -----------------------------------
// Implementing linalg_traits::Scalar.
// -----------------------------------

// HyperDual += HyperDual.
impl AddAssign for HyperDual {
    fn add_assign(&mut self, other: HyperDual) {
        self.a += other.a;
        self.b += other.b;
        self.c += other.c;
        self.d += other.d;
    }
}

// HyperDual -= HyperDual.
impl SubAssign for HyperDual {
    fn sub_assign(&mut self, other: HyperDual) {
        self.a -= other.a;
        self.b -= other.b;
        self.c -= other.c;
        self.d -= other.d;
    }
}

// HyperDual *= HyperDual.
impl MulAssign for HyperDual {
    fn mul_assign(&mut self, other: HyperDual) {
        let result = *self * other;
        *self = result;
    }
}

// HyperDual /= HyperDual.
impl DivAssign for HyperDual {
    fn div_assign(&mut self, other: HyperDual) {
        let result = *self / other;
        *self = result;
    }
}

// HyperDual %= HyperDual.
impl RemAssign for HyperDual {
    fn rem_assign(&mut self, rhs: HyperDual) {
        let result = *self % rhs;
        *self = result;
    }
}

// HyperDual + f64.
impl Add<f64> for HyperDual {
    type Output = HyperDual;
    fn add(self, rhs: f64) -> HyperDual {
        HyperDual::new(self.a + rhs, self.b, self.c, self.d)
    }
}

// HyperDual - f64.
impl Sub<f64> for HyperDual {
    type Output = HyperDual;
    fn sub(self, rhs: f64) -> HyperDual {
        HyperDual::new(self.a - rhs, self.b, self.c, self.d)
    }
}

// HyperDual * f64.
impl Mul<f64> for HyperDual {
    type Output = HyperDual;
    fn mul(self, rhs: f64) -> HyperDual {
        HyperDual::new(self.a * rhs, self.b * rhs, self.c * rhs, self.d * rhs)
    }
}

// HyperDual / f64.
impl Div<f64> for HyperDual {
    type Output = HyperDual;
    fn div(self, rhs: f64) -> HyperDual {
        HyperDual::new(self.a / rhs, self.b / rhs, self.c / rhs, self.d / rhs)
    }
}

// HyperDual % f64.
impl Rem<f64> for HyperDual {
    type Output = HyperDual;
    fn rem(self, rhs: f64) -> HyperDual {
        if self.a % rhs == 0.0 {
            HyperDual::new(self.a % rhs, 0.0, 0.0, 0.0)
        } else {
            HyperDual::new(self.a % rhs, self.b, self.c, self.d)
        }
    }
}

// HyperDual += f64.
impl AddAssign<f64> for HyperDual {
    fn add_assign(&mut self, rhs: f64) {
        self.a += rhs;
    }
}

// HyperDual -= f64.
impl SubAssign<f64> for HyperDual {
    fn sub_assign(&mut self, rhs: f64) {
        self.a -= rhs;
    }
}

// HyperDual *= f64.
impl MulAssign<f64> for HyperDual {
    fn mul_assign(&mut self, rhs: f64) {
        self.a *= rhs;
        self.b *= rhs;
        self.c *= rhs;
        self.d *= rhs;
    }
}

// HyperDual /= f64.
impl DivAssign<f64> for HyperDual {
    fn div_assign(&mut self, rhs: f64) {
        self.a /= rhs;
        self.b /= rhs;
        self.c /= rhs;
        self.d /= rhs;
    }
}

// HyperDual %= f64.
impl RemAssign<f64> for HyperDual {
    fn rem_assign(&mut self, rhs: f64) {
        self.a %= rhs;
        if self.a == 0.0 {
            self.b = 0.0;
            self.c = 0.0;
            self.d = 0.0;
        }
    }
}

// ---------------------------
// Interoperability with f64s.
// ---------------------------

// f64 + HyperDual.
impl Add<HyperDual> for f64 {
    type Output = HyperDual;
    fn add(self, rhs: HyperDual) -> HyperDual {
        HyperDual::new(self + rhs.a, rhs.b, rhs.c, rhs.d)
    }
}

// f64 - HyperDual.
impl Sub<HyperDual> for f64 {
    type Output = HyperDual;
    fn sub(self, rhs: HyperDual) -> HyperDual {
        HyperDual::new(self - rhs.a, -rhs.b, -rhs.c, -rhs.d)
    }
}

// f64 * HyperDual.
impl Mul<HyperDual> for f64 {
    type Output = HyperDual;
    fn mul(self, rhs: HyperDual) -> HyperDual {
        HyperDual::new(self * rhs.a, self * rhs.b, self * rhs.c, self * rhs.d)
    }
}

// f64 / HyperDual.
impl Div<HyperDual> for f64 {
    type Output = HyperDual;
    fn div(self, rhs: HyperDual) -> HyperDual {
        HyperDual::from_real(self) / rhs
    }
}

// f64 % HyperDual.
impl Rem<HyperDual> for f64 {
    type Output = HyperDual;
    fn rem(self, rhs: HyperDual) -> HyperDual {
        let q = (self / rhs.a).floor();
        HyperDual::new(self % rhs.a, -q * rhs.b, -q * rhs.c, -q * rhs.d)
    }
}

// ------------------------
// Implementing trig::Trig.
// ------------------------

#[cfg(feature = "trig")]
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
        1.0 / self.tan()
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
        (HyperDual::from_real(1.0) / *self).asin()
    }
    fn asec(&self) -> HyperDual {
        (HyperDual::from_real(1.0) / *self).acos()
    }
    fn acot(&self) -> HyperDual {
        (HyperDual::from_real(1.0) / *self).atan()
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
        1.0 / self.cosh()
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
        (HyperDual::from_real(1.0) / *self).asinh()
    }
    fn asech(&self) -> HyperDual {
        (HyperDual::from_real(1.0) / *self).acosh()
    }
    fn acoth(&self) -> HyperDual {
        (HyperDual::from_real(1.0) / *self).atanh()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use linalg_traits::Scalar;
    use numtest::*;
    use std::f64::consts::{E, FRAC_PI_4, FRAC_PI_6};

    #[cfg(feature = "trig")]
    use trig::Trig;

    #[cfg(feature = "trig")]
    use std::f64::consts::PI;

    fn assert_hyper_dual_close(left: HyperDual, right: HyperDual, decimal: i32) {
        assert_arrays_equal_to_decimal!(
            [left.get_a(), left.get_b(), left.get_c(), left.get_d()],
            [right.get_a(), right.get_b(), right.get_c(), right.get_d()],
            decimal
        );
    }

    #[test]
    fn test_new() {
        let num1 = HyperDual::new(1.0, 2.0, 3.0, 4.0);
        let num2 = HyperDual {
            a: 1.0,
            b: 2.0,
            c: 3.0,
            d: 4.0,
        };
        assert_eq!(num1.a, num2.a);
        assert_eq!(num1.b, num2.b);
        assert_eq!(num1.c, num2.c);
        assert_eq!(num1.d, num2.d);
    }

    #[test]
    fn test_get_a() {
        let num = HyperDual::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(num.get_a(), 1.0);
    }

    #[test]
    fn test_get_b() {
        let num = HyperDual::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(num.get_b(), 2.0);
    }

    #[test]
    fn test_get_c() {
        let num = HyperDual::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(num.get_c(), 3.0);
    }

    #[test]
    fn test_get_d() {
        let num = HyperDual::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(num.get_d(), 4.0);
    }

    #[test]
    fn test_from_real() {
        assert_eq!(
            HyperDual::from_real(1.0),
            HyperDual::new(1.0, 0.0, 0.0, 0.0)
        );
        assert_eq!(
            HyperDual::from_real(-2.5),
            HyperDual::new(-2.5, 0.0, 0.0, 0.0)
        );
    }

    #[test]
    fn test_add_hyper_dual_hyper_dual() {
        assert_eq!(
            HyperDual::new(1.0, 2.0, 3.0, 4.0) + HyperDual::new(5.0, 6.0, 7.0, 8.0),
            HyperDual::new(6.0, 8.0, 10.0, 12.0)
        );
    }

    #[test]
    fn test_sub_hyper_dual_hyper_dual() {
        assert_eq!(
            HyperDual::new(1.0, 2.0, 3.0, 4.0) - HyperDual::new(5.0, 6.0, 7.0, 8.0),
            HyperDual::new(-4.0, -4.0, -4.0, -4.0)
        );
    }

    #[test]
    fn test_mul_hyper_dual_hyper_dual() {
        assert_eq!(
            HyperDual::new(1.0, 2.0, 3.0, 4.0) * HyperDual::new(5.0, 6.0, 7.0, 8.0),
            HyperDual::new(5.0, 16.0, 22.0, 60.0)
        );
    }

    #[test]
    fn test_div_hyper_dual_hyper_dual() {
        assert_eq!(
            HyperDual::new(1.0, 2.0, 3.0, 4.0) / HyperDual::from_real(2.0),
            HyperDual::new(0.5, 1.0, 1.5, 2.0)
        );
    }

    #[test]
    fn test_rem_hyper_dual_hyper_dual() {
        // Spot check.
        assert_eq!(
            HyperDual::new(5.0, 2.0, 3.0, 4.0) % HyperDual::new(3.0, 4.0, 5.0, 6.0),
            HyperDual::new(2.0, -2.0, -2.0, -2.0)
        );

        // Check parity with the truncated definition of the remainder.
        //  --> Reference: https://en.wikipedia.org/wiki/Modulo#In_programming_languages
        let a = HyperDual::new(5.0, 2.0, 3.0, 4.0);
        let n = HyperDual::new(3.0, 5.0, 6.0, 7.0);
        assert_eq!(a % n, a - n * (a / n).trunc());
    }

    #[test]
    fn test_zero() {
        // Construction.
        assert_eq!(HyperDual::zero(), HyperDual::from_real(0.0));

        // Zero-check.
        assert!(HyperDual::zero().is_zero());
        assert!(HyperDual::from_real(0.0).is_zero());

        // HyperDual::zero() * HyperDual = HyperDual::zero().
        assert_eq!(
            HyperDual::zero() * HyperDual::new(1.0, 2.0, 3.0, 4.0),
            HyperDual::zero()
        );
    }

    #[test]
    fn test_one() {
        // Construction.
        assert_eq!(HyperDual::one(), HyperDual::from_real(1.0));

        // HyperDual::one() * HyperDual = HyperDual.
        assert_eq!(
            HyperDual::one() * HyperDual::new(1.0, 2.0, 3.0, 4.0),
            HyperDual::new(1.0, 2.0, 3.0, 4.0)
        );

        // HyperDual::one() * scalar = HyperDual::from_real(scalar).
        assert_eq!(HyperDual::one() * 5.0, HyperDual::from_real(5.0));
    }

    #[test]
    fn test_from_str_radix() {
        assert_eq!(
            HyperDual::from_str_radix("2.125", 10).unwrap(),
            HyperDual::from_real(2.125)
        );
    }

    #[test]
    fn test_to_i64() {
        assert_eq!(HyperDual::new(1.0, 2.0, 3.0, 4.0).to_i64().unwrap(), 1_i64);
        assert_eq!(
            HyperDual::new(-1.0, 2.0, 3.0, 4.0).to_i64().unwrap(),
            -1_i64
        );
    }

    #[test]
    fn test_to_u64() {
        assert_eq!(HyperDual::new(1.0, 2.0, 3.0, 4.0).to_u64().unwrap(), 1_u64);
        assert!(HyperDual::new(-1.0, 2.0, 3.0, 4.0).to_u64().is_none());
    }

    #[test]
    fn test_to_f64() {
        assert_eq!(
            HyperDual::new(1.0, 2.0, 3.0, 4.0).to_f64().unwrap(),
            1.0_f64
        );
        assert_eq!(
            HyperDual::new(-1.0, 2.0, 3.0, 4.0).to_f64().unwrap(),
            -1.0_f64
        );
    }

    #[test]
    fn test_from_i64() {
        assert_eq!(
            <HyperDual as NumCast>::from(1_i64).unwrap(),
            HyperDual::from_real(1.0)
        );
        assert_eq!(
            <HyperDual as NumCast>::from(-1_i64).unwrap(),
            HyperDual::from_real(-1.0)
        );
    }

    #[test]
    fn test_from_u64() {
        assert_eq!(
            <HyperDual as NumCast>::from(1_u64).unwrap(),
            HyperDual::from_real(1.0)
        );
    }

    #[test]
    fn test_from_f64() {
        assert_eq!(
            <HyperDual as NumCast>::from(1_f64).unwrap(),
            HyperDual::from_real(1.0)
        );
    }

    #[test]
    fn test_partial_ord() {
        // Check <.
        assert!(HyperDual::new(1.0, 2.0, 3.0, 4.0) < HyperDual::new(3.0, 4.0, 5.0, 6.0));
        assert!(HyperDual::new(1.0, 9.0, 8.0, 7.0) < HyperDual::new(3.0, -1.0, -2.0, -3.0));
        assert!(HyperDual::new(-3.0, -4.0, -5.0, -6.0) < HyperDual::new(-1.0, -2.0, -3.0, -4.0));
        assert!(HyperDual::new(-3.0, -1.0, -2.0, -3.0) < HyperDual::new(-1.0, -4.0, -5.0, -6.0));

        // Check >.
        assert!(HyperDual::new(3.0, 4.0, 5.0, 6.0) > HyperDual::new(1.0, 2.0, 3.0, 4.0));
        assert!(HyperDual::new(3.0, 2.0, 1.0, 0.0) > HyperDual::new(1.0, 4.0, 5.0, 6.0));
        assert!(HyperDual::new(-1.0, -2.0, -3.0, -4.0) > HyperDual::new(-3.0, -4.0, -5.0, -6.0));
        assert!(HyperDual::new(-1.0, -4.0, -5.0, -6.0) > HyperDual::new(-3.0, -2.0, -1.0, 0.0));

        // Check <=.
        assert!(HyperDual::new(0.0, 2.0, 3.0, 4.0) <= HyperDual::new(1.0, 2.0, 3.0, 4.0));
        assert!(HyperDual::new(1.0, 2.0, 3.0, 4.0) <= HyperDual::new(1.0, 2.0, 3.0, 4.0));

        // Check >=.
        assert!(HyperDual::new(2.0, 2.0, 3.0, 4.0) >= HyperDual::new(1.0, 2.0, 3.0, 4.0));
        assert!(HyperDual::new(1.0, 2.0, 3.0, 4.0) >= HyperDual::new(1.0, 2.0, 3.0, 4.0));
    }

    #[test]
    fn test_neg() {
        assert_eq!(
            -HyperDual::new(1.0, 2.0, 3.0, 4.0),
            HyperDual::new(-1.0, -2.0, -3.0, -4.0)
        );
        assert_eq!(
            -HyperDual::new(1.0, -2.0, 3.0, -4.0),
            HyperDual::new(-1.0, 2.0, -3.0, 4.0)
        );
        assert_eq!(
            -HyperDual::new(-1.0, 2.0, -3.0, 4.0),
            HyperDual::new(1.0, -2.0, 3.0, -4.0)
        );
        assert_eq!(
            -HyperDual::new(-1.0, -2.0, -3.0, -4.0),
            HyperDual::new(1.0, 2.0, 3.0, 4.0)
        );
    }

    #[test]
    fn test_nan() {
        let num = HyperDual::nan();
        assert!(num.get_a().is_nan());
        assert!(num.get_b().is_nan());
        assert!(num.get_c().is_nan());
        assert!(num.get_d().is_nan());
    }

    #[test]
    fn test_infinity() {
        let num = HyperDual::infinity();
        assert!(num.get_a().is_infinite() & (num.get_a() > 0.0));
        assert!(num.get_b().is_infinite() & (num.get_b() > 0.0));
        assert!(num.get_c().is_infinite() & (num.get_c() > 0.0));
        assert!(num.get_d().is_infinite() & (num.get_d() > 0.0));
    }

    #[test]
    fn test_neg_infinity() {
        let num = HyperDual::neg_infinity();
        assert!(num.get_a().is_infinite() & (num.get_a() < 0.0));
        assert!(num.get_b().is_infinite() & (num.get_b() < 0.0));
        assert!(num.get_c().is_infinite() & (num.get_c() < 0.0));
        assert!(num.get_d().is_infinite() & (num.get_d() < 0.0));
    }

    #[test]
    fn test_neg_zero() {
        let num = HyperDual::neg_zero();
        assert!(num.get_a().is_zero());
        assert!(num.get_b().is_zero());
        assert!(num.get_c().is_zero());
        assert!(num.get_d().is_zero());
    }

    #[test]
    fn test_min_value() {
        let num = HyperDual::min_value();
        assert!(num.get_a() == f64::MIN);
        assert!(num.get_b() == f64::MIN);
        assert!(num.get_c() == f64::MIN);
        assert!(num.get_d() == f64::MIN);
    }

    #[test]
    fn test_min_positive_value() {
        let num = HyperDual::min_positive_value();
        assert!(num.get_a() == f64::MIN_POSITIVE);
        assert!(num.get_b() == f64::MIN_POSITIVE);
        assert!(num.get_c() == f64::MIN_POSITIVE);
        assert!(num.get_d() == f64::MIN_POSITIVE);
    }

    #[test]
    fn test_max_value() {
        let num = HyperDual::max_value();
        assert!(num.get_a() == f64::MAX);
        assert!(num.get_b() == f64::MAX);
        assert!(num.get_c() == f64::MAX);
        assert!(num.get_d() == f64::MAX);
    }

    #[test]
    fn test_is_nan() {
        assert!(HyperDual::nan().is_nan());
        assert!(HyperDual::from_real(f64::NAN).is_nan());
        assert!(!HyperDual::new(0.0, f64::NAN, f64::NAN, f64::NAN).is_nan());
        assert!(!HyperDual::from_real(0.0).is_nan());
    }

    #[test]
    fn test_is_infinite() {
        assert!(HyperDual::infinity().is_infinite());
        assert!(HyperDual::neg_infinity().is_infinite());
        assert!(HyperDual::from_real(f64::INFINITY).is_infinite());
        assert!(HyperDual::from_real(f64::NEG_INFINITY).is_infinite());
        assert!(!HyperDual::new(0.0, f64::INFINITY, f64::INFINITY, f64::INFINITY).is_infinite());
        assert!(
            !HyperDual::new(0.0, f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY)
                .is_infinite()
        );
        assert!(!HyperDual::from_real(0.0).is_infinite());
    }

    #[test]
    fn test_is_finite() {
        assert!(!HyperDual::infinity().is_finite());
        assert!(!HyperDual::neg_infinity().is_finite());
        assert!(!HyperDual::from_real(f64::INFINITY).is_finite());
        assert!(!HyperDual::from_real(f64::NEG_INFINITY).is_finite());
        assert!(HyperDual::new(0.0, f64::INFINITY, f64::INFINITY, f64::INFINITY).is_finite());
        assert!(
            HyperDual::new(0.0, f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY)
                .is_finite()
        );
        assert!(HyperDual::from_real(0.0).is_finite());
    }

    /// # References
    ///
    /// * <https://docs.rs/num-traits/latest/num_traits/float/trait.Float.html#tymethod.is_normal>
    ///
    /// # Note
    ///
    /// For each of these tests, we use `f64::NAN` in the non-real components to ensure that
    /// `is_normal` is only checking the real part.
    #[test]
    fn test_is_normal() {
        // Normal (for these checks we use not-normal non-real components to ensure that only the
        // real part is being checked).
        assert!(HyperDual::new(1.0, f64::NAN, f64::NAN, f64::NAN).is_normal());
        assert!(HyperDual::new(f64::MIN_POSITIVE, f64::NAN, f64::NAN, f64::NAN).is_normal());
        assert!(HyperDual::new(f64::MAX, f64::NAN, f64::NAN, f64::NAN).is_normal());

        // Not normal (for these checks we use normal non-real components to ensure that only the
        // real part is being checked).
        assert!(!HyperDual::new(0.0, 1.0, 1.0, 1.0).is_normal()); // Zero.
        assert!(!HyperDual::new(f64::NAN, 1.0, 1.0, 1.0).is_normal()); // NaN.
        assert!(!HyperDual::new(f64::INFINITY, 1.0, 1.0, 1.0).is_normal()); // Infinite.
        assert!(!HyperDual::new(f64::NEG_INFINITY, 1.0, 1.0, 1.0).is_normal()); // Infinite.
        assert!(!HyperDual::new(1.0e-308_f64, 1.0, 1.0, 1.0).is_normal()); // Subnormal.
    }

    #[test]
    fn test_classify() {
        // Normal (for these checks we use not-normal non-real components to ensure that only the
        // real part is being checked).
        assert_eq!(
            HyperDual::new(1.0, f64::NAN, f64::NAN, f64::NAN).classify(),
            FpCategory::Normal
        );
        assert_eq!(
            HyperDual::new(f64::MIN_POSITIVE, f64::NAN, f64::NAN, f64::NAN).classify(),
            FpCategory::Normal
        );
        assert_eq!(
            HyperDual::new(f64::MAX, f64::NAN, f64::NAN, f64::NAN).classify(),
            FpCategory::Normal
        );

        // Not normal (for these checks we use normal non-real components to ensure that only the
        // real part is being checked).
        assert_eq!(
            HyperDual::new(0.0, 1.0, 1.0, 1.0).classify(),
            FpCategory::Zero
        );
        assert_eq!(
            HyperDual::new(f64::NAN, 1.0, 1.0, 1.0).classify(),
            FpCategory::Nan
        );
        assert_eq!(
            HyperDual::new(f64::INFINITY, 1.0, 1.0, 1.0).classify(),
            FpCategory::Infinite
        );
        assert_eq!(
            HyperDual::new(f64::NEG_INFINITY, 1.0, 1.0, 1.0).classify(),
            FpCategory::Infinite
        );
        assert_eq!(
            HyperDual::new(1.0e-308_f64, 1.0, 1.0, 1.0).classify(),
            FpCategory::Subnormal
        );
    }

    #[test]
    fn test_floor() {
        assert_eq!(
            HyperDual::new(2.7, 2.7, -2.7, 5.0).floor(),
            HyperDual::from_real(2.0)
        );
        assert_eq!(
            HyperDual::new(-2.7, 2.7, -2.7, 5.0).floor(),
            HyperDual::from_real(-3.0)
        );
    }

    #[test]
    fn test_ceil() {
        assert_eq!(
            HyperDual::new(2.7, 2.7, -2.7, 5.0).ceil(),
            HyperDual::from_real(3.0)
        );
        assert_eq!(
            HyperDual::new(-2.7, 2.7, -2.7, 5.0).ceil(),
            HyperDual::from_real(-2.0)
        );
    }

    #[test]
    fn test_round() {
        assert_eq!(
            HyperDual::new(2.3, 2.3, -2.3, 5.0).round(),
            HyperDual::from_real(2.0)
        );
        assert_eq!(
            HyperDual::new(2.7, 2.7, -2.7, 5.0).round(),
            HyperDual::from_real(3.0)
        );
        assert_eq!(
            HyperDual::new(-2.7, 2.7, -2.7, 5.0).round(),
            HyperDual::from_real(-3.0)
        );
        assert_eq!(
            HyperDual::new(-2.3, 2.3, -2.3, 5.0).round(),
            HyperDual::from_real(-2.0)
        );
    }

    #[test]
    fn test_trunc() {
        assert_eq!(
            HyperDual::new(2.3, 2.3, -2.3, 5.0).trunc(),
            HyperDual::from_real(2.0)
        );
        assert_eq!(
            HyperDual::new(2.7, 2.7, -2.7, 5.0).trunc(),
            HyperDual::from_real(2.0)
        );
        assert_eq!(
            HyperDual::new(-2.7, 2.7, -2.7, 5.0).trunc(),
            HyperDual::from_real(-2.0)
        );
        assert_eq!(
            HyperDual::new(-2.3, 2.3, -2.3, 5.0).trunc(),
            HyperDual::from_real(-2.0)
        );
    }

    #[test]
    fn test_fract() {
        assert_eq!(
            HyperDual::new(2.5, 2.5, 3.5, 4.5).fract(),
            HyperDual::new(0.5, 2.5, 3.5, 4.5)
        );
        assert_eq!(
            HyperDual::new(-2.5, -2.5, -3.5, -4.5).fract(),
            HyperDual::new(-0.5, -2.5, -3.5, -4.5)
        );
    }

    #[test]
    fn test_abs() {
        assert_eq!(
            HyperDual::new(1.0, 2.0, 3.0, 4.0).abs(),
            HyperDual::new(1.0, 2.0, 3.0, 4.0)
        );
        assert_eq!(
            HyperDual::new(-1.0, -2.0, -3.0, -4.0).abs(),
            HyperDual::new(1.0, 2.0, 3.0, 4.0)
        );
        assert_eq!(
            HyperDual::new(-1.0, 2.0, -3.0, 4.0).abs(),
            HyperDual::new(1.0, -2.0, 3.0, -4.0)
        );
    }

    #[test]
    fn test_signum() {
        assert_eq!(
            HyperDual::new(2.0, 4.0, -3.0, 5.0).signum(),
            HyperDual::from_real(1.0)
        );
        assert_eq!(
            HyperDual::new(-2.0, -4.0, 3.0, -5.0).signum(),
            HyperDual::from_real(-1.0)
        );
    }

    #[test]
    fn test_is_sign_positive() {
        assert!(HyperDual::new(2.0, -4.0, 7.0, -8.0).is_sign_positive());
        assert!(!HyperDual::new(-2.0, 4.0, -7.0, 8.0).is_sign_positive());
    }

    #[test]
    fn test_is_sign_negative() {
        assert!(HyperDual::new(-2.0, 4.0, -7.0, 8.0).is_sign_negative());
        assert!(!HyperDual::new(2.0, -4.0, 7.0, -8.0).is_sign_negative());
    }

    #[test]
    fn test_mul_add() {
        let a = HyperDual::new(1.0, 3.0, -2.0, 4.0);
        let b = HyperDual::new(-2.0, 5.0, 7.0, -3.0);
        let c = HyperDual::new(10.0, -4.0, 6.0, 8.0);
        assert_eq!(c.mul_add(a, b), (c * a) + b);
    }

    #[test]
    fn test_recip() {
        assert_hyper_dual_close(
            HyperDual::new(2.0, -5.0, 7.0, 11.0).recip(),
            HyperDual::new(0.5, 1.25, -1.75, -11.5),
            14,
        );
    }

    #[test]
    fn test_powi() {
        let x = HyperDual::new(2.0, -5.0, 7.0, 11.0);
        assert_eq!(x.powi(0), HyperDual::new(1.0, 0.0, 0.0, 0.0));
        assert_eq!(x.powi(1), x);

        assert_hyper_dual_close(
            HyperDual::new(2.0, -5.0, 7.0, 11.0).powi(2),
            HyperDual::new(4.0, -20.0, 28.0, -26.0),
            14,
        );

        assert_hyper_dual_close(
            HyperDual::new(2.0, -5.0, 7.0, 11.0).powi(3),
            HyperDual::new(8.0, -60.0, 84.0, -288.0),
            14,
        );
    }

    #[test]
    fn test_powf() {
        let assert_close = |out: HyperDual, expected: HyperDual| {
            assert_hyper_dual_close(out, expected, 13);
        };

        let x = HyperDual::new(2.0, -5.0, 7.0, 11.0);

        // Test against powi for integer powers.
        assert_close(x.powf(HyperDual::from_real(0.0)), x.powi(0));
        assert_close(x.powf(HyperDual::from_real(1.0)), x.powi(1));
        assert_close(x.powf(HyperDual::from_real(2.0)), x.powi(2));
        assert_close(x.powf(HyperDual::from_real(3.0)), x.powi(3));

        // Test against sqrt.
        assert_close(x.powf(HyperDual::from_real(0.5)), x.sqrt());

        // Test against cbrt.
        assert_close(x.powf(HyperDual::from_real(1.0 / 3.0)), x.cbrt());
    }

    #[test]
    fn test_sqrt() {
        assert_hyper_dual_close(
            HyperDual::new(4.0, 25.0, -3.0, 7.0).sqrt(),
            HyperDual::new(2.0, 6.25, -0.75, 4.09375),
            14,
        );
    }

    #[test]
    fn test_exp() {
        assert_hyper_dual_close(
            HyperDual::new(2.0, -3.0, 5.0, 7.0).exp(),
            HyperDual::new(
                7.38905609893065,
                -22.16716829679195,
                36.945280494653254,
                -59.11244879144519,
            ),
            14,
        );
    }

    #[test]
    fn test_exp2() {
        assert_hyper_dual_close(
            HyperDual::new(2.0, -3.0, 5.0, 7.0).exp2(),
            HyperDual::new(
                4.0,
                -8.317766166719343,
                13.862943611198906,
                -9.419059779413612,
            ),
            14,
        );
    }

    #[test]
    fn test_ln() {
        assert_hyper_dual_close(
            HyperDual::new(5.0, 8.0, -3.0, 7.0).ln(),
            HyperDual::new(
                1.6094379124341003,
                1.6,
                -0.6000000000000001,
                2.3600000000000003,
            ),
            14,
        );
    }

    #[test]
    fn test_log() {
        let x = HyperDual::new(5.0, 8.0, -3.0, 7.0);
        let base = HyperDual::new(4.5, 2.0, 6.0, -5.0);
        let out = x.log(base);
        let expected = x.ln() / base.ln();
        assert_hyper_dual_close(out, expected, 14);
    }

    #[test]
    fn test_log2() {
        assert_hyper_dual_close(
            HyperDual::new(5.0, 8.0, -3.0, 7.0).log2(),
            HyperDual::new(
                2.321928094887362,
                2.3083120654223412,
                -0.865617024533378,
                3.4047602964979533,
            ),
            14,
        );
    }

    #[test]
    fn test_log10() {
        assert_hyper_dual_close(
            HyperDual::new(5.0, 8.0, -3.0, 7.0).log10(),
            HyperDual::new(
                0.6989700043360189,
                0.6948711710452028,
                -0.26057668914195103,
                1.0249349772916743,
            ),
            14,
        );
    }

    #[test]
    fn test_max() {
        assert_eq!(
            HyperDual::new(1.0, 2.0, 3.0, 4.0).max(HyperDual::new(3.0, 4.0, 5.0, 6.0)),
            HyperDual::new(3.0, 4.0, 5.0, 6.0)
        );
        assert_eq!(
            HyperDual::new(3.0, 2.0, 3.0, 4.0).max(HyperDual::new(1.0, 4.0, 5.0, 6.0)),
            HyperDual::new(3.0, 2.0, 3.0, 4.0)
        );
        assert_eq!(
            HyperDual::new(3.0, 4.0, 5.0, 6.0).max(HyperDual::new(1.0, 2.0, 3.0, 4.0)),
            HyperDual::new(3.0, 4.0, 5.0, 6.0)
        );
        assert_eq!(
            HyperDual::new(-1.0, 2.0, 3.0, 4.0).max(HyperDual::new(-3.0, 4.0, 5.0, 6.0)),
            HyperDual::new(-1.0, 2.0, 3.0, 4.0)
        );
    }

    #[test]
    fn test_min() {
        assert_eq!(
            HyperDual::new(1.0, 2.0, 3.0, 4.0).min(HyperDual::new(3.0, 4.0, 5.0, 6.0)),
            HyperDual::new(1.0, 2.0, 3.0, 4.0)
        );
        assert_eq!(
            HyperDual::new(3.0, 2.0, 3.0, 4.0).min(HyperDual::new(1.0, 4.0, 5.0, 6.0)),
            HyperDual::new(1.0, 4.0, 5.0, 6.0)
        );
        assert_eq!(
            HyperDual::new(3.0, 4.0, 5.0, 6.0).min(HyperDual::new(1.0, 2.0, 3.0, 4.0)),
            HyperDual::new(1.0, 2.0, 3.0, 4.0)
        );
        assert_eq!(
            HyperDual::new(-1.0, 2.0, 3.0, 4.0).min(HyperDual::new(-3.0, 4.0, 5.0, 6.0)),
            HyperDual::new(-3.0, 4.0, 5.0, 6.0)
        );
    }

    #[test]
    fn test_abs_sub() {
        assert_eq!(
            HyperDual::new(4.0, 5.0, 6.0, 7.0).abs_sub(HyperDual::new(2.0, 8.0, 9.0, 10.0)),
            HyperDual::new(2.0, -3.0, -3.0, -3.0)
        );
    }

    #[test]
    fn test_cbrt() {
        assert_hyper_dual_close(
            HyperDual::new(8.0, 27.0, -4.0, 5.0).cbrt(),
            HyperDual::new(2.0, 2.25, -0.3333333333333333, 1.1666666666666665),
            14,
        );
    }

    #[test]
    fn test_hypot() {
        let x = HyperDual::new(1.0, 2.0, -3.0, 5.0);
        let y = HyperDual::new(3.0, 4.0, 7.0, -11.0);

        assert_hyper_dual_close(
            x.hypot(y),
            HyperDual::new(
                3.1622776601683795,
                4.427188724235731,
                5.692099788303082,
                -9.866306299725345,
            ),
            14,
        );

        // Check parity with Euclidian norm.
        let norm = (x.powi(2) + y.powi(2)).sqrt();
        assert_hyper_dual_close(x.hypot(y), norm, 14);
    }

    #[test]
    fn test_sin() {
        assert_hyper_dual_close(
            HyperDual::new(FRAC_PI_6, 2.0, -3.0, 5.0).sin(),
            HyperDual::new(
                0.5,
                1.7320508075688772,
                -2.598076211353316,
                7.330127018922193,
            ),
            14,
        );
    }

    #[test]
    fn test_cos() {
        assert_hyper_dual_close(
            HyperDual::new(FRAC_PI_6, 2.0, -3.0, 5.0).cos(),
            HyperDual::new(0.8660254037844386, -1.0, 1.5, 2.696152422706632),
            14,
        );
    }

    #[test]
    fn test_tan() {
        assert_hyper_dual_close(
            HyperDual::new(FRAC_PI_6, 2.0, -3.0, 5.0).tan(),
            HyperDual::new(
                0.5773502691896258,
                2.666666666666667,
                -4.0,
                -2.5709376403673474,
            ),
            14,
        );
    }

    #[test]
    fn test_asin() {
        assert_hyper_dual_close(
            HyperDual::new(0.5, 3.0, -2.0, 7.0).asin(),
            HyperDual::new(
                0.5235987755982988,
                3.4641016151377553,
                -2.3094010767585034,
                3.4641016151377553,
            ),
            14,
        );
    }

    #[test]
    fn test_asin_near_domain_edges() {
        let (b, c, d) = (1.2, -0.8, 0.5);
        for x in [-0.99_f64, -0.9, 0.9, 0.99] {
            let denom_sqrt = (1.0 - x.powi(2)).sqrt();
            let df = 1.0 / denom_sqrt;
            let d2f = x / (1.0 - x.powi(2)).powf(1.5);
            assert_hyper_dual_close(
                HyperDual::new(x, b, c, d).asin(),
                HyperDual::new(x.asin(), df * b, df * c, d2f * b * c + df * d),
                12,
            );
        }
    }

    #[test]
    fn test_asin_out_of_domain_nan() {
        assert!(
            HyperDual::new(1.0001, 2.0, -3.0, 4.0)
                .asin()
                .get_a()
                .is_nan()
        );
        assert!(
            HyperDual::new(-1.0001, 2.0, -3.0, 4.0)
                .asin()
                .get_a()
                .is_nan()
        );
    }

    #[test]
    fn test_acos() {
        assert_hyper_dual_close(
            HyperDual::new(3.0_f64.sqrt() / 2.0, 3.0, -2.0, 7.0).acos(),
            HyperDual::new(
                FRAC_PI_6,
                -5.999999999999998,
                3.999999999999999,
                27.56921938165303,
            ),
            14,
        );
    }

    #[test]
    fn test_acos_near_domain_edges() {
        let (b, c, d) = (-1.1, 0.9, -0.4);
        for x in [-0.99_f64, -0.9, 0.9, 0.99] {
            let denom_sqrt = (1.0 - x.powi(2)).sqrt();
            let df = -1.0 / denom_sqrt;
            let d2f = -x / (1.0 - x.powi(2)).powf(1.5);
            assert_hyper_dual_close(
                HyperDual::new(x, b, c, d).acos(),
                HyperDual::new(x.acos(), df * b, df * c, d2f * b * c + df * d),
                12,
            );
        }
    }

    #[test]
    fn test_acos_out_of_domain_nan() {
        assert!(
            HyperDual::new(1.0001, 2.0, -3.0, 4.0)
                .acos()
                .get_a()
                .is_nan()
        );
        assert!(
            HyperDual::new(-1.0001, 2.0, -3.0, 4.0)
                .acos()
                .get_a()
                .is_nan()
        );
    }

    #[test]
    fn test_atan() {
        assert_hyper_dual_close(
            HyperDual::new(1.0, 3.0, -2.0, 7.0).atan(),
            HyperDual::new(FRAC_PI_4, 1.5, -1.0, 6.5),
            14,
        );
    }

    #[test]
    fn test_atan2() {
        assert_hyper_dual_close(
            HyperDual::new(-3.0, 5.0, 7.0, -6.0).atan2(HyperDual::new(3.0, 2.0, -1.0, 4.0)),
            HyperDual::new(
                -FRAC_PI_4,
                1.1666666666666665,
                0.9999999999999999,
                1.7222222222222223,
            ),
            14,
        );
    }

    #[test]
    fn test_sin_cos() {
        let x = HyperDual::new(FRAC_PI_6, 2.0, -3.0, 5.0);
        let (sin, cos) = x.sin_cos();
        assert_hyper_dual_close(
            sin,
            HyperDual::new(
                0.5,
                1.7320508075688772,
                -2.598076211353316,
                7.330127018922193,
            ),
            14,
        );
        assert_hyper_dual_close(
            cos,
            HyperDual::new(0.8660254037844386, -1.0, 1.5, 2.696152422706632),
            14,
        );
    }

    #[test]
    fn test_exp_m1() {
        let x = HyperDual::new(3.0, 5.0, -2.0, 4.0);
        assert_hyper_dual_close(x.exp_m1(), x.exp() - HyperDual::one(), 14);
    }

    #[test]
    fn test_ln_1p() {
        let x = HyperDual::new(3.0, 5.0, -2.0, 4.0);
        assert_hyper_dual_close(x.ln_1p(), (x + HyperDual::one()).ln(), 14);
    }

    #[test]
    fn test_sinh() {
        assert_hyper_dual_close(
            HyperDual::new(1.0, 2.0, -3.0, 4.0).sinh(),
            HyperDual::new(
                1.1752011936438014,
                3.0861612696304874,
                -4.629241904445731,
                -0.8788846226018334,
            ),
            14,
        );
        // Keep one exact-value sanity check for the real and first-order term.
        assert_hyper_dual_close(
            HyperDual::new(1.0, 2.0, 1.5, -0.75).sinh(),
            HyperDual::new(
                ((E * E) - 1.0) / (2.0 * E),
                ((E * E) + 1.0) / E,
                2.3146209522228656,
                2.3682931048199714,
            ),
            15,
        );
    }

    #[test]
    fn test_cosh() {
        assert_hyper_dual_close(
            HyperDual::new(1.0, 2.0, -3.0, 4.0).cosh(),
            HyperDual::new(
                1.5430806348152437,
                2.3504023872876028,
                -3.525603580931404,
                -4.557679034316257,
            ),
            14,
        );
    }

    #[test]
    fn test_tanh() {
        assert_hyper_dual_close(
            HyperDual::new(1.0, 2.0, -3.0, 4.0).tanh(),
            HyperDual::new(
                0.7615941559557649,
                0.8399486832280523,
                -1.2599230248420783,
                5.518097417151452,
            ),
            14,
        );
    }

    #[test]
    fn test_asinh() {
        assert_hyper_dual_close(
            HyperDual::new(1.0, 2.0, -3.0, 4.0).sinh().asinh(),
            HyperDual::new(1.0, 2.0, -3.0, 4.0),
            13,
        );
    }

    #[test]
    fn test_acosh() {
        assert_hyper_dual_close(
            HyperDual::new(1.0, 2.0, -3.0, 4.0).cosh().acosh(),
            HyperDual::new(1.0, 2.0, -3.0, 4.0),
            13,
        );
    }

    #[test]
    fn test_acosh_near_domain_boundary() {
        let (b, c, d) = (1.1, -0.7, 0.3);
        for x in [1.0001_f64, 1.01, 2.0] {
            let df = 1.0 / ((x.powi(2) - 1.0).sqrt());
            let d2f = -x / (x.powi(2) - 1.0).powf(1.5);
            assert_hyper_dual_close(
                HyperDual::new(x, b, c, d).acosh(),
                HyperDual::new(x.acosh(), df * b, df * c, d2f * b * c + df * d),
                11,
            );
        }
    }

    #[test]
    fn test_acosh_out_of_domain_nan() {
        assert!(
            HyperDual::new(0.9999, 2.0, -3.0, 4.0)
                .acosh()
                .get_a()
                .is_nan()
        );
    }

    #[test]
    fn test_atanh() {
        assert_hyper_dual_close(
            HyperDual::new(0.5, 2.0, -3.0, 4.0).tanh().atanh(),
            HyperDual::new(0.5, 2.0, -3.0, 4.0),
            12,
        );
    }

    #[test]
    fn test_atanh_near_domain_edges() {
        let (b, c, d) = (-0.9, 0.6, -0.2);
        for x in [-0.99_f64, -0.9, 0.9, 0.99] {
            let df = 1.0 / (1.0 - x.powi(2));
            let d2f = 2.0 * x / (1.0 - x.powi(2)).powi(2);
            assert_hyper_dual_close(
                HyperDual::new(x, b, c, d).atanh(),
                HyperDual::new(x.atanh(), df * b, df * c, d2f * b * c + df * d),
                11,
            );
        }
    }

    #[test]
    fn test_atanh_out_of_domain_nan() {
        assert!(
            HyperDual::new(1.0001, 2.0, -3.0, 4.0)
                .atanh()
                .get_a()
                .is_nan()
        );
        assert!(
            HyperDual::new(-1.0001, 2.0, -3.0, 4.0)
                .atanh()
                .get_a()
                .is_nan()
        );
    }

    #[test]
    fn test_integer_decode() {
        assert_eq!(
            HyperDual::new(1.2345e-5, 6.789e-7, 2.468e-7, -1.357e-7).integer_decode(),
            (1.2345e-5).integer_decode()
        );
    }

    #[test]
    fn test_add_assign_hyper_dual_hyper_dual() {
        let mut a = HyperDual::new(1.0, 2.0, 3.0, 4.0);
        a += HyperDual::new(3.0, 4.0, 5.0, 6.0);
        assert_eq!(a, HyperDual::new(4.0, 6.0, 8.0, 10.0));
    }

    #[test]
    fn test_sub_assign_hyper_dual_hyper_dual() {
        let mut a = HyperDual::new(1.0, 2.0, 3.0, 4.0);
        a -= HyperDual::new(4.0, 3.0, 2.0, 1.0);
        assert_eq!(a, HyperDual::new(-3.0, -1.0, 1.0, 3.0));
    }

    #[test]
    fn test_mul_assign_hyper_dual_hyper_dual() {
        let mut a = HyperDual::new(1.0, 2.0, 3.0, 4.0);
        a *= HyperDual::new(3.0, -4.0, 5.0, -6.0);
        assert_eq!(a, HyperDual::new(3.0, 2.0, 14.0, 4.0));
    }

    #[test]
    fn test_div_assign_hyper_dual_hyper_dual() {
        let mut a = HyperDual::new(1.0, 2.0, 3.0, 4.0);
        a /= HyperDual::from_real(2.0);
        assert_eq!(a, HyperDual::new(0.5, 1.0, 1.5, 2.0));
    }

    #[test]
    fn test_rem_assign_hyper_dual_hyper_dual() {
        let mut a = HyperDual::new(5.0, 2.0, 3.0, 4.0);
        a %= HyperDual::new(3.0, 4.0, 5.0, 6.0);
        assert_eq!(a, HyperDual::new(2.0, -2.0, -2.0, -2.0));
    }

    #[test]
    fn test_add_hyper_dual_f64() {
        assert_eq!(
            HyperDual::new(1.0, 2.0, 3.0, 4.0) + 3.0,
            HyperDual::new(4.0, 2.0, 3.0, 4.0)
        );
    }

    #[test]
    fn test_sub_hyper_dual_f64() {
        assert_eq!(
            HyperDual::new(1.0, 2.0, 3.0, 4.0) - 3.0,
            HyperDual::new(-2.0, 2.0, 3.0, 4.0)
        );
    }

    #[test]
    fn test_mul_hyper_dual_f64() {
        assert_eq!(
            HyperDual::new(1.0, -2.0, 3.0, -4.0) * 3.0,
            HyperDual::new(3.0, -6.0, 9.0, -12.0)
        );
    }

    #[test]
    fn test_div_hyper_dual_f64() {
        assert_eq!(
            HyperDual::new(1.0, 2.0, 3.0, 4.0) / 4.0,
            HyperDual::new(0.25, 0.5, 0.75, 1.0)
        );
    }

    #[test]
    fn test_rem_hyper_dual_f64() {
        // Spot check.
        assert_eq!(
            HyperDual::new(5.0, 1.0, 2.0, 3.0) % 3.0,
            HyperDual::new(2.0, 1.0, 2.0, 3.0)
        );

        // Check parity with the truncated definition of the remainder.
        //  --> Reference: https://en.wikipedia.org/wiki/Modulo#In_programming_languages
        let a = HyperDual::new(5.0, 1.0, 2.0, 3.0);
        let n = 3.0;
        assert_eq!(a % n, a - n * (a / n).trunc());
    }

    #[test]
    fn test_add_assign_hyper_dual_f64() {
        let mut a = HyperDual::new(1.0, 2.0, 3.0, 4.0);
        a += 3.0;
        assert_eq!(a, HyperDual::new(4.0, 2.0, 3.0, 4.0));
    }

    #[test]
    fn test_sub_assign_hyper_dual_f64() {
        let mut a = HyperDual::new(1.0, 2.0, 3.0, 4.0);
        a -= 3.0;
        assert_eq!(a, HyperDual::new(-2.0, 2.0, 3.0, 4.0));
    }

    #[test]
    fn test_mul_assign_hyper_dual_f64() {
        let mut a = HyperDual::new(2.0, -3.0, 4.0, -5.0);
        a *= 5.0;
        assert_eq!(a, HyperDual::new(10.0, -15.0, 20.0, -25.0));
    }

    #[test]
    fn test_div_assign_hyper_dual_f64() {
        let mut a = HyperDual::new(1.0, 2.0, 3.0, 4.0);
        a /= 4.0;
        assert_eq!(a, HyperDual::new(0.25, 0.5, 0.75, 1.0));
    }

    #[test]
    fn test_rem_assign_hyper_dual_f64() {
        let mut a = HyperDual::new(5.0, 1.0, 2.0, 3.0);
        a %= 3.0;
        assert_eq!(a, HyperDual::new(2.0, 1.0, 2.0, 3.0));
    }

    #[test]
    fn test_add_f64_hyper_dual() {
        assert_eq!(
            1.0 + HyperDual::new(2.0, 3.0, 4.0, 5.0),
            HyperDual::new(3.0, 3.0, 4.0, 5.0)
        );
    }

    #[test]
    fn test_sub_f64_hyper_dual() {
        assert_eq!(
            1.0 - HyperDual::new(2.0, 3.0, 4.0, 5.0),
            HyperDual::new(-1.0, -3.0, -4.0, -5.0)
        );
    }

    #[test]
    fn test_mul_f64_hyper_dual() {
        assert_eq!(
            5.0 * HyperDual::new(2.0, -3.0, 4.0, -5.0),
            HyperDual::new(10.0, -15.0, 20.0, -25.0)
        );
    }

    #[test]
    fn test_div_f64_hyper_dual() {
        assert_eq!(
            5.0 / HyperDual::new(2.0, -3.0, 4.0, -5.0),
            HyperDual::new(2.5, 3.75, -5.0, -8.75)
        );
    }

    #[test]
    fn test_rem_f64_hyper_dual() {
        // Spot check.
        assert_eq!(
            5.0 % HyperDual::new(2.0, -3.0, -4.0, -5.0),
            HyperDual::new(1.0, 6.0, 8.0, 10.0)
        );

        // Check parity with "HyperDual % HyperDual" implementation.
        assert_eq!(
            5.0 % HyperDual::new(2.0, -3.0, -4.0, -5.0),
            HyperDual::from_real(5.0) % HyperDual::new(2.0, -3.0, -4.0, -5.0)
        );
    }

    #[cfg(feature = "trig")]
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

    #[cfg(feature = "trig")]
    #[test]
    fn test_sec() {
        assert_hyper_dual_close(
            HyperDual::new(FRAC_PI_6, 2.0, 1.5, -0.75).sec(),
            HyperDual::new(2.0 / 3.0_f64.sqrt(), 4.0 / 3.0, 1.0, 5.273502691896258),
            15,
        );
    }

    #[cfg(feature = "trig")]
    #[test]
    fn test_cot() {
        assert_hyper_dual_close(
            HyperDual::new(FRAC_PI_6, 2.0, 1.5, -0.75).cot(),
            HyperDual::new(
                3.0_f64.sqrt(),
                -7.999999999999998,
                -5.999999999999998,
                44.569219381653035,
            ),
            15,
        );
    }

    #[cfg(feature = "trig")]
    #[test]
    fn test_acsc() {
        assert_hyper_dual_close(
            HyperDual::new(FRAC_PI_6, 2.0, 1.5, -0.75).csc().acsc(),
            HyperDual::new(FRAC_PI_6, 2.0, 1.5000000000000002, -0.7500000000000004),
            14,
        );
    }

    #[cfg(feature = "trig")]
    #[test]
    fn test_asec() {
        assert_hyper_dual_close(
            HyperDual::new(FRAC_PI_6, 2.0, 1.5, -0.75).sec().asec(),
            HyperDual::new(
                0.5235987755982991,
                1.9999999999999982,
                1.4999999999999987,
                -0.7499999999999893,
            ),
            14,
        );
    }

    #[cfg(feature = "trig")]
    #[test]
    fn test_acot() {
        assert_hyper_dual_close(
            HyperDual::new(FRAC_PI_6, 2.0, 1.5, -0.75).cot().acot(),
            HyperDual::new(FRAC_PI_6, 2.0000000000000004, 1.5, -0.7500000000000018),
            14,
        );
    }

    #[cfg(feature = "trig")]
    #[test]
    fn test_deg2rad() {
        assert_hyper_dual_close(
            HyperDual::new(180.0, 2.0, 1.5, -0.75).deg2rad(),
            HyperDual::new(PI, PI / 90.0, 0.026179938779914945, -0.013089969389957472),
            15,
        );
    }

    #[cfg(feature = "trig")]
    #[test]
    fn test_rad2deg() {
        assert_hyper_dual_close(
            HyperDual::new(PI, 2.0, 1.5, -0.75).rad2deg(),
            HyperDual::new(180.0, 360.0 / PI, 85.94366926962348, -42.97183463481174),
            14,
        );
    }

    #[cfg(feature = "trig")]
    #[test]
    fn test_sind() {
        assert_hyper_dual_close(
            HyperDual::new(30.0, 2.0, 1.5, -0.75).sind(),
            HyperDual::new(
                0.49999999999999994,
                PI * 3.0_f64.sqrt() / 180.0,
                0.022672492052927727,
                -0.011793172156143927,
            ),
            15,
        );
    }

    #[cfg(feature = "trig")]
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

    #[cfg(feature = "trig")]
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

    #[cfg(feature = "trig")]
    #[test]
    fn test_cscd() {
        assert_hyper_dual_close(
            HyperDual::new(30.0, 2.0, 1.5, -0.75).cscd(),
            HyperDual::new(
                2.0000000000000004,
                -PI * 3.0_f64.sqrt() / 45.0,
                -0.09068996821171092,
                0.05813891573689724,
            ),
            15,
        );
    }

    #[cfg(feature = "trig")]
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

    #[cfg(feature = "trig")]
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

    #[cfg(feature = "trig")]
    #[test]
    fn test_asind() {
        assert_hyper_dual_close(
            HyperDual::new(30.0, 2.0, 1.5, -0.75).sind().asind(),
            HyperDual::new(
                29.999999999999996,
                2.0000000000000004,
                1.5000000000000004,
                -0.7500000000000002,
            ),
            12,
        );
    }

    #[cfg(feature = "trig")]
    #[test]
    fn test_acosd() {
        assert_hyper_dual_close(
            HyperDual::new(30.0, 2.0, 1.5, -0.75).cosd().acosd(),
            HyperDual::new(
                29.999999999999996,
                2.0,
                1.5000000000000002,
                -0.7500000000000001,
            ),
            12,
        );
    }

    #[cfg(feature = "trig")]
    #[test]
    fn test_atand() {
        assert_hyper_dual_close(
            HyperDual::new(30.0, 2.0, 1.5, -0.75).tand().atand(),
            HyperDual::new(
                29.999999999999996,
                2.0000000000000004,
                1.5000000000000002,
                -0.7500000000000001,
            ),
            12,
        );
    }

    #[cfg(feature = "trig")]
    #[test]
    fn test_atan2d() {
        assert_hyper_dual_close(
            HyperDual::new(-3.0, 5.0, 1.5, -0.75).atan2d(&HyperDual::new(3.0, 2.0, 1.5, -0.75)),
            HyperDual::new(-45.0, 210.0 / PI, 28.64788975654116, 1.5902773407317584e-15),
            14,
        );
    }

    #[cfg(feature = "trig")]
    #[test]
    fn test_acscd() {
        assert_hyper_dual_close(
            HyperDual::new(30.0, 2.0, 1.5, -0.75).cscd().acscd(),
            HyperDual::new(
                29.99999999999993,
                1.9999999999999996,
                1.4999999999999998,
                -0.7500000000000001,
            ),
            11,
        );
    }

    #[cfg(feature = "trig")]
    #[test]
    fn test_asecd() {
        assert_hyper_dual_close(
            HyperDual::new(30.0, 2.0, 1.5, -0.75).secd().asecd(),
            HyperDual::new(
                29.999999999999996,
                1.9999999999999996,
                1.5000000000000002,
                -0.75,
            ),
            11,
        );
    }

    #[cfg(feature = "trig")]
    #[test]
    fn test_acotd() {
        assert_hyper_dual_close(
            HyperDual::new(30.0, 2.0, 1.5, -0.75).cotd().acotd(),
            HyperDual::new(
                29.999999999999996,
                2.0000000000000004,
                1.5000000000000004,
                -0.7500000000000002,
            ),
            11,
        );
    }

    #[cfg(feature = "trig")]
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

    #[cfg(feature = "trig")]
    #[test]
    fn test_sech() {
        assert_hyper_dual_close(
            HyperDual::new(1.0, 2.0, 1.5, -0.75).sech(),
            HyperDual::new(
                1.0_f64.cosh().recip(),
                -2.0 * 1.0_f64.sinh() / 1.0_f64.cosh().powi(2),
                -0.7403315213468595,
                0.6813315801922095,
            ),
            15,
        );
    }

    #[cfg(feature = "trig")]
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

    #[cfg(feature = "trig")]
    #[test]
    fn test_acsch() {
        assert_hyper_dual_close(
            HyperDual::new(1.0, 2.0, 1.5, -0.75).csch().acsch(),
            HyperDual::new(
                1.0,
                2.0000000000000004,
                1.5000000000000004,
                -0.7499999999999996,
            ),
            12,
        );
    }

    #[cfg(feature = "trig")]
    #[test]
    fn test_asech() {
        assert_hyper_dual_close(
            HyperDual::new(1.0, 2.0, 1.5, -0.75).sech().asech(),
            HyperDual::new(
                1.0,
                1.9999999999999996,
                1.4999999999999996,
                -0.7500000000000004,
            ),
            12,
        );
    }

    #[cfg(feature = "trig")]
    #[test]
    fn test_acoth() {
        assert_hyper_dual_close(
            HyperDual::new(2.0, 2.0, 1.5, -0.75).coth().acoth(),
            HyperDual::new(
                1.9999999999999987,
                1.9999999999999933,
                1.499999999999995,
                -0.750000000000016,
            ),
            12,
        );
    }

    // This just verifies that the scalar trait was fully implemented.
    #[test]
    fn test_scalar() {
        fn add_scalar<S: Scalar>(x: S, y: S) -> S {
            x + y
        }
        assert_eq!(
            add_scalar(
                HyperDual::new(5.0, 4.0, 3.0, 2.0),
                HyperDual::new(3.0, 2.0, 1.0, 0.0)
            ),
            HyperDual::new(8.0, 6.0, 4.0, 2.0)
        );
    }
}
