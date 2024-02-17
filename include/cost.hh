#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <numeric>

#include "model.hh"
#include "reduce.hh"

namespace moptimizer {

/// @brief Cost function class.
/// @tparam In Input data type. Assume this is a vect
/// @tparam Out Measured data type.
/// @tparam Scalar scalar primitive.
template <class In, class Out, class Model, class Scalar = double>
class Cost {
 public:
  Cost() = delete;
  Cost(const Cost& rhs) = delete;

  /// @brief Cost function constructor.
  /// @param model Model
  /// @param input input data iterator
  /// @param measurements measured data iterator
  /// @param num_elements number of elements.
  Cost(const In* input, const Out* measurements, size_t num_elements)
      : input_(input), measurements_(measurements), num_elements_(num_elements) {}

  /// @brief Computes sum of errors
  /// @param x
  /// @return
  Scalar sum(const Scalar* x) const {
    Scalar total = 0.;
    reduce_functor<Scalar> reducer;
    return std::transform_reduce(input_, input_ + num_elements_, measurements_, total,
                                 reducer,    // reduce
                                 Model(x));  // Transform
    // transform
  }

  /// @brief Computes the hessian of a cost function numerically.
  /// @tparam parameter_dim
  /// @param x
  /// @param hessian
  /// @param b
  template <int parameter_dim>
  void linearize(const Scalar* x, Scalar* hessian, Scalar* b) const {
    constexpr size_t dim_parameter = parameter_dim;
    constexpr size_t dim_output = sizeof(Out) / sizeof(Scalar);
    using HessianType = Eigen::Matrix<Scalar, dim_parameter, dim_parameter>;
    using JacobianType = Eigen::Matrix<Scalar, dim_output, dim_parameter>;
    using ParameterType = Eigen::Matrix<Scalar, dim_parameter, 1>;
    // using LinearizePair = std::pair<HessianType, ParameterType>;

    Eigen::Map<const ParameterType> parameter(x);
    HessianType hessian_;
    ParameterType b_;
    hessian_.setZero();
    b_.setZero();

    Model model(x);
    std::vector<Model> models_plus;
    std::vector<ParameterType> parameters_plus(dim_parameter);

    const Scalar min_step_size = std::sqrt(std::numeric_limits<Scalar>::epsilon());

    // Prepare models.
    for (size_t p_dim = 0; p_dim < dim_parameter; ++p_dim) {
      parameters_plus[p_dim] = parameter;
      parameters_plus[p_dim][p_dim] += min_step_size;
      models_plus.push_back(Model(parameters_plus[p_dim].data()));
    }

    auto numeric_jacobian = [&](const In& in, const Out& out) -> JacobianType {
      JacobianType local_jacobian;
      for (size_t p_dim = 0; p_dim < dim_parameter; ++p_dim) {
        const auto&& diff = (models_plus[p_dim](in, out) - model(in, out)) / min_step_size;
        if constexpr (dim_output == 1)
          local_jacobian[p_dim] = diff;
        else
          local_jacobian.col(p_dim) = diff;
      }
      return local_jacobian;
    };
    // Perform reduction.
    hessian_ = std::transform_reduce(input_, input_ + num_elements_, measurements_, hessian_,
                                     // Reduce
                                     reduce_functor_jacobian<HessianType, JacobianType>(),
                                     // Transform (compute jacobian per data input)
                                     numeric_jacobian
    );
    // Assign outputs
    Eigen::Map<HessianType> ret(hessian);
    ret = hessian_;
  }

 private:
  const In* input_;
  const Out* measurements_;
  size_t num_elements_;
};
}  // namespace moptimizer