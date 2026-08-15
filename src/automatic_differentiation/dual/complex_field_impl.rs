use crate::automatic_differentiation::dual::dual::Dual;
use faer_traits::{ComplexField, SimdCapabilities, pulp::Scalar as FaerScalar, pulp::Simd};
use num_traits::Float;
use num_traits::{One, Zero};

// ------------------------------------------------
// Implementing faer_traits::ComplexField for Dual.
// ------------------------------------------------
// NOTE: This is required for implementing faer_traits::RealField, which in turn is required for
// implementing linalg_traits::Scalar.

#[cfg(feature = "faer")]
impl ComplexField for Dual {
    type Arch = FaerScalar;
    type Unit = Self;
    type Index = usize;

    // No SIMD representation yet.
    type SimdCtx<S: Simd> = S;
    type SimdVec<S: Simd> = ();
    type SimdMask<S: Simd> = ();
    type SimdMemMask<S: Simd> = ();
    type SimdIndex<S: Simd> = ();

    type Real = Self;

    const IS_REAL: bool = true;
    const SIMD_CAPABILITIES: SimdCapabilities = SimdCapabilities::Copy;

    #[inline(always)]
    fn zero_impl() -> Self {
        Self::zero()
    }

    #[inline(always)]
    fn one_impl() -> Self {
        Self::one()
    }

    #[inline(always)]
    fn nan_impl() -> Self {
        Self::nan()
    }

    #[inline(always)]
    fn infinity_impl() -> Self {
        Self::infinity()
    }

    #[inline(always)]
    fn from_real_impl(value: &Self::Real) -> Self {
        *value
    }

    #[inline(always)]
    fn from_f64_impl(value: f64) -> Self {
        <Self as From<f64>>::from(value)
    }

    #[inline(always)]
    fn real_part_impl(value: &Self) -> Self::Real {
        *value
    }

    #[inline(always)]
    fn imag_part_impl(_: &Self) -> Self::Real {
        Self::zero_impl()
    }

    #[inline(always)]
    fn copy_impl(value: &Self) -> Self {
        *value
    }

    #[inline(always)]
    fn conj_impl(value: &Self) -> Self {
        *value
    }

    #[inline(always)]
    fn recip_impl(value: &Self) -> Self {
        Self::recip(*value)
    }

    #[inline(always)]
    fn sqrt_impl(value: &Self) -> Self {
        Self::sqrt(*value)
    }

    #[inline(always)]
    fn abs_impl(value: &Self) -> Self::Real {
        Self::abs(*value)
    }

    #[inline(always)]
    fn abs1_impl(value: &Self) -> Self::Real {
        Self::abs_impl(value)
    }

    #[inline(always)]
    fn abs2_impl(value: &Self) -> Self::Real {
        *value * *value
    }

    #[inline(always)]
    fn mul_real_impl(lhs: &Self, rhs: &Self::Real) -> Self {
        *lhs * *rhs
    }

    #[inline(always)]
    fn mul_pow2_impl(lhs: &Self, rhs: &Self::Real) -> Self {
        *lhs * *rhs
    }

    #[inline(always)]
    fn is_finite_impl(value: &Self) -> bool {
        Self::is_finite(*value)
    }

    // ------------------------------------------------------------------
    // SIMD plumbing
    //
    // Dual currently declares SIMD_CAPABILITIES = Copy, so these are
    // never used as a vectorized implementation. They only exist because
    // ComplexField requires the associated machinery.
    // ------------------------------------------------------------------

    #[inline(always)]
    fn simd_ctx<S: Simd>(simd: S) -> Self::SimdCtx<S> {
        simd
    }

    #[inline(always)]
    fn ctx_from_simd<S: Simd>(ctx: &Self::SimdCtx<S>) -> S {
        *ctx
    }

    #[inline(always)]
    fn simd_mask_between<S: Simd>(
        _: &Self::SimdCtx<S>,
        _: Self::Index,
        _: Self::Index,
    ) -> Self::SimdMask<S> {
        ()
    }

    #[inline(always)]
    fn simd_mem_mask_between<S: Simd>(
        _: &Self::SimdCtx<S>,
        _: Self::Index,
        _: Self::Index,
    ) -> Self::SimdMemMask<S> {
        ()
    }

    #[inline(always)]
    unsafe fn simd_mask_load_raw<S: Simd>(
        _: &Self::SimdCtx<S>,
        _: Self::SimdMemMask<S>,
        _: *const Self::SimdVec<S>,
    ) -> Self::SimdVec<S> {
        ()
    }

    #[inline(always)]
    unsafe fn simd_mask_store_raw<S: Simd>(
        _: &Self::SimdCtx<S>,
        _: Self::SimdMemMask<S>,
        _: *mut Self::SimdVec<S>,
        _: Self::SimdVec<S>,
    ) {
    }

    #[inline(always)]
    fn simd_splat<S: Simd>(_: &Self::SimdCtx<S>, _: &Self) -> Self::SimdVec<S> {
        ()
    }

    #[inline(always)]
    fn simd_splat_real<S: Simd>(_: &Self::SimdCtx<S>, _: &Self::Real) -> Self::SimdVec<S> {
        ()
    }

    #[inline(always)]
    fn simd_add<S: Simd>(
        _: &Self::SimdCtx<S>,
        _: Self::SimdVec<S>,
        _: Self::SimdVec<S>,
    ) -> Self::SimdVec<S> {
        ()
    }

    #[inline(always)]
    fn simd_sub<S: Simd>(
        _: &Self::SimdCtx<S>,
        _: Self::SimdVec<S>,
        _: Self::SimdVec<S>,
    ) -> Self::SimdVec<S> {
        ()
    }

    #[inline(always)]
    fn simd_neg<S: Simd>(_: &Self::SimdCtx<S>, _: Self::SimdVec<S>) -> Self::SimdVec<S> {
        ()
    }

    #[inline(always)]
    fn simd_conj<S: Simd>(_: &Self::SimdCtx<S>, _: Self::SimdVec<S>) -> Self::SimdVec<S> {
        ()
    }

    #[inline(always)]
    fn simd_abs1<S: Simd>(_: &Self::SimdCtx<S>, _: Self::SimdVec<S>) -> Self::SimdVec<S> {
        ()
    }

    #[inline(always)]
    fn simd_abs_max<S: Simd>(_: &Self::SimdCtx<S>, _: Self::SimdVec<S>) -> Self::SimdVec<S> {
        ()
    }

    #[inline(always)]
    fn simd_mul_real<S: Simd>(
        _: &Self::SimdCtx<S>,
        _: Self::SimdVec<S>,
        _: Self::SimdVec<S>,
    ) -> Self::SimdVec<S> {
        ()
    }

    #[inline(always)]
    fn simd_mul_pow2<S: Simd>(
        _: &Self::SimdCtx<S>,
        _: Self::SimdVec<S>,
        _: Self::SimdVec<S>,
    ) -> Self::SimdVec<S> {
        ()
    }

    #[inline(always)]
    fn simd_mul<S: Simd>(
        _: &Self::SimdCtx<S>,
        _: Self::SimdVec<S>,
        _: Self::SimdVec<S>,
    ) -> Self::SimdVec<S> {
        ()
    }

    #[inline(always)]
    fn simd_conj_mul<S: Simd>(
        _: &Self::SimdCtx<S>,
        _: Self::SimdVec<S>,
        _: Self::SimdVec<S>,
    ) -> Self::SimdVec<S> {
        ()
    }

    #[inline(always)]
    fn simd_mul_add<S: Simd>(
        _: &Self::SimdCtx<S>,
        _: Self::SimdVec<S>,
        _: Self::SimdVec<S>,
        _: Self::SimdVec<S>,
    ) -> Self::SimdVec<S> {
        ()
    }

    #[inline(always)]
    fn simd_conj_mul_add<S: Simd>(
        _: &Self::SimdCtx<S>,
        _: Self::SimdVec<S>,
        _: Self::SimdVec<S>,
        _: Self::SimdVec<S>,
    ) -> Self::SimdVec<S> {
        ()
    }

    #[inline(always)]
    fn simd_abs2<S: Simd>(_: &Self::SimdCtx<S>, _: Self::SimdVec<S>) -> Self::SimdVec<S> {
        ()
    }

    #[inline(always)]
    fn simd_abs2_add<S: Simd>(
        _: &Self::SimdCtx<S>,
        _: Self::SimdVec<S>,
        _: Self::SimdVec<S>,
    ) -> Self::SimdVec<S> {
        ()
    }

    #[inline(always)]
    fn simd_reduce_sum<S: Simd>(_: &Self::SimdCtx<S>, _: Self::SimdVec<S>) -> Self {
        Self::zero_impl()
    }

    #[inline(always)]
    fn simd_reduce_max<S: Simd>(_: &Self::SimdCtx<S>, _: Self::SimdVec<S>) -> Self {
        Self::zero_impl()
    }

    #[inline(always)]
    fn simd_equal<S: Simd>(
        _: &Self::SimdCtx<S>,
        _: Self::SimdVec<S>,
        _: Self::SimdVec<S>,
    ) -> Self::SimdMask<S> {
        ()
    }

    #[inline(always)]
    fn simd_less_than<S: Simd>(
        _: &Self::SimdCtx<S>,
        _: Self::SimdVec<S>,
        _: Self::SimdVec<S>,
    ) -> Self::SimdMask<S> {
        ()
    }

    #[inline(always)]
    fn simd_less_than_or_equal<S: Simd>(
        _: &Self::SimdCtx<S>,
        _: Self::SimdVec<S>,
        _: Self::SimdVec<S>,
    ) -> Self::SimdMask<S> {
        ()
    }

    #[inline(always)]
    fn simd_greater_than<S: Simd>(
        _: &Self::SimdCtx<S>,
        _: Self::SimdVec<S>,
        _: Self::SimdVec<S>,
    ) -> Self::SimdMask<S> {
        ()
    }

    #[inline(always)]
    fn simd_greater_than_or_equal<S: Simd>(
        _: &Self::SimdCtx<S>,
        _: Self::SimdVec<S>,
        _: Self::SimdVec<S>,
    ) -> Self::SimdMask<S> {
        ()
    }

    #[inline(always)]
    fn simd_select<S: Simd>(
        _: &Self::SimdCtx<S>,
        _: Self::SimdMask<S>,
        _: Self::SimdVec<S>,
        _: Self::SimdVec<S>,
    ) -> Self::SimdVec<S> {
        ()
    }

    #[inline(always)]
    fn simd_index_select<S: Simd>(
        _: &Self::SimdCtx<S>,
        _: Self::SimdMask<S>,
        _: Self::SimdIndex<S>,
        _: Self::SimdIndex<S>,
    ) -> Self::SimdIndex<S> {
        ()
    }

    #[inline(always)]
    fn simd_index_splat<S: Simd>(_: &Self::SimdCtx<S>, _: Self::Index) -> Self::SimdIndex<S> {
        ()
    }

    #[inline(always)]
    fn simd_index_add<S: Simd>(
        _: &Self::SimdCtx<S>,
        _: Self::SimdIndex<S>,
        _: Self::SimdIndex<S>,
    ) -> Self::SimdIndex<S> {
        ()
    }

    #[inline(always)]
    fn simd_index_less_than<S: Simd>(
        _: &Self::SimdCtx<S>,
        _: Self::SimdIndex<S>,
        _: Self::SimdIndex<S>,
    ) -> Self::SimdMask<S> {
        ()
    }

    #[inline(always)]
    fn simd_and_mask<S: Simd>(
        _: &Self::SimdCtx<S>,
        _: Self::SimdMask<S>,
        _: Self::SimdMask<S>,
    ) -> Self::SimdMask<S> {
        ()
    }

    #[inline(always)]
    fn simd_or_mask<S: Simd>(
        _: &Self::SimdCtx<S>,
        _: Self::SimdMask<S>,
        _: Self::SimdMask<S>,
    ) -> Self::SimdMask<S> {
        ()
    }

    #[inline(always)]
    fn simd_not_mask<S: Simd>(_: &Self::SimdCtx<S>, _: Self::SimdMask<S>) -> Self::SimdMask<S> {
        ()
    }

    #[inline(always)]
    fn simd_first_true_mask<S: Simd>(_: &Self::SimdCtx<S>, _: Self::SimdMask<S>) -> usize {
        0
    }
}
