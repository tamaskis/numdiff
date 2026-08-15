/// Second-order hyper-dual number.
///
/// A hyper-dual number is represented as
///
/// `a + (b)ε₁ + (c)ε₂ + (d)ε₁ε₂`
///
/// where `ε₁² = ε₂² = 0` and `ε₁ε₂ = ε₂ε₁`.
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct HyperDual {
    /// Real part.
    pub(crate) a: f64,

    /// Coefficient of `ε₁`.
    pub(crate) b: f64,

    /// Coefficient of `ε₂`.
    pub(crate) c: f64,

    /// Coefficient of `ε₁ε₂`.
    pub(crate) d: f64,
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
    pub(crate) fn univariate_map<F, DF, D2F>(self, f: F, df: DF, d2f: D2F) -> HyperDual
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
    pub(crate) fn bivariate_map<G, GX, GY, GXX, GXY, GYY>(
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
