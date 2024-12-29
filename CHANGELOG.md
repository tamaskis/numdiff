# Changelog

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