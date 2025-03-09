# Changelog

## 0.1.6

1. Updated `linalg-traits` dependency from `0.9.1` to `0.11.1`.
    * This also adds support for using `faer::Mat` with any of the differentation functions/macros provided by this crate, including automatic differentiation.
1. Updated `once_cell` dependency from `1.1.19` to `1.20.3`.
1. Updated `trig` optional dependency from `0.1.3` to `0.1.4`.
1. Updated `nalgebra` dev dependency from `0.33.0` to `0.33.2`.
1. Updated `ndarray` dev dependency from `0.16.0` to `0.16.1`.
1. Updated `numtest` dev dependency from `0.2.0` to `0.2.2`.

## 0.1.5

1. Implemented the following forward-mode automatic differentiation macros:

    - `numdiff::get_jacobian!`

## 0.1.4

1. Implemented the following forward-mode automatic differentiation macros:

    - `numdiff::get_spartial_derivative!`
    - `numdiff::get_vpartial_derivative!`

## 0.1.3

1. Implemented the following forward-mode automatic differentiation macros:

    - `numdiff::get_directional_derivative!`

1. Swapped the order of the arguments to `numdiff::get_gradient!` to match the other forward-mode automatic differentiation macros.

## 0.1.2

1. Implemented the following forward-mode automatic differentiation macros:

    - `numdiff::get_sderivative!`
    - `numdiff::get_vderivative!`
    - `numdiff::get_gradient!`

## 0.1.1

1. Implemented the following central difference approximation differentiation functions:

    - `numdiff::central_difference::sderivative`
    - `numdiff::central_difference::vderivative`
    - `numdiff::central_difference::spartial`
    - `numdiff::central_difference::vpartial`
    - `numdiff::central_difference::gradient`
    - `numdiff::central_difference::directional_derivative`
    - `numdiff::central_difference::jacobian`
    - `numdiff::central_difference::shessian`
    - `numdiff::central_difference::vhessian`

## 0.1.0

1. Initial release.
1. Implemented the following forward difference approximation differentiation functions:

    - `numdiff::forward_difference::sderivative`
    - `numdiff::forward_difference::vderivative`
    - `numdiff::forward_difference::spartial`
    - `numdiff::forward_difference::vpartial`
    - `numdiff::forward_difference::gradient`
    - `numdiff::forward_difference::directional_derivative`
    - `numdiff::forward_difference::jacobian`
    - `numdiff::forward_difference::shessian`
    - `numdiff::forward_difference::vhessian`