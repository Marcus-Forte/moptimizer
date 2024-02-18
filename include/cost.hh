#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <execution>
#include <numeric>

#include "Icost.hh"
#include "jacobian.hh"
#include "model.hh"

namespace moptimizer {

/// @brief Cost function class.
/// @tparam parameter_dim parameter dimension
/// @tparam In Input data type
/// @tparam Out Measured data type.
/// @tparam ModelT Fuctor of application model.
/// @tparam JacobianModelT (optional)  Functor of jacobian model
/// @tparam Scalar scalar primitive.
template <int parameter_dim, class In, class Out, class ModelT, class JacobianModelT = void*, class Scalar = double>
class Cost : public ICost<Scalar> {
 public:
  Cost() = delete;
  Cost(const Cost& rhs) = delete;

  /// @brief Cost function constructor.
  /// @param input input data iterator
  /// @param measurements measured data iterator
  /// @param num_elements number of elements.
  Cost(const In* input, const Out* measurements, size_t num_elements)
      : input_(input), measurements_(measurements), num_elements_(num_elements) {
    static_assert(std::is_base_of<Model<In, Out, Scalar>, ModelT>::value,
                  "ModelT must be inherited from moptimizer::Model");
  }

  /// @brief Computes sum of squared errors
  /// @param x
  /// @return
  Scalar sum(const Scalar* x) const {
    constexpr size_t dim_output = sizeof(Out) / sizeof(Scalar);
    using OutputAsVector = Eigen::Matrix<Scalar, dim_output, 1>;
    Scalar total = 0.;
    return std::transform_reduce(
        input_, input_ + num_elements_, measurements_, total,
        // Reduce
        [&](const Out& init, const Out& res) -> Scalar {
          Eigen::Map<const OutputAsVector> init_map(&init);
          Eigen::Map<const OutputAsVector> res_map(&res);

          // Compute squared error
          return init + res_map.dot(res_map);
        },
        // Transform
        ModelT(x));
  }

  /// @brief Computes the hessian and b vector of a cost function.
  /// @tparam parameter_dim
  /// @param x
  /// @param hessian
  /// @param b
  /// @return Sum of squared errors.
  Scalar linearize(const Scalar* x, Scalar* hessian, Scalar* b) const override {
    constexpr size_t dim_parameter = parameter_dim;
    constexpr size_t dim_output = sizeof(Out) / sizeof(Scalar);
    using HessianType = Eigen::Matrix<Scalar, dim_parameter, dim_parameter>;
    using JacobianType = Eigen::Matrix<Scalar, dim_output, dim_parameter>;
    using ParameterType = Eigen::Matrix<Scalar, dim_parameter, 1>;
    using ResultPair = jacobian_result_pair<dim_parameter, Scalar>;

    ResultPair result;
    auto reduce_lambda = [&](ResultPair init, const ResultPair& b) -> ResultPair { return init + b; };
    if constexpr (std::is_same<JacobianModelT, void*>::value) {
      numeric_jacobian<dim_parameter, dim_output, ModelT, In, Out, Scalar> numeric_jacobian(x);
      result = std::transform_reduce(std::execution::par, input_, input_ + num_elements_, measurements_, result,
                                     // Reduce
                                     reduce_lambda,
                                     // Transform
                                     numeric_jacobian);
    } else {
      static_assert(std::is_base_of<JacobianModel<In, JacobianType, Scalar>, JacobianModelT>::value,
                    "ModelT must be inherited from moptimizer::JacobianModel");
      analytic_jacobian<dim_parameter, dim_output, ModelT, JacobianModelT, In, Out, Scalar> analytic_jacobian(x);
      result = std::transform_reduce(std::execution::par, input_, input_ + num_elements_, measurements_, result,
                                     // Reduce
                                     reduce_lambda,
                                     // Transform
                                     analytic_jacobian);
    }

    // Assign outputs
    Eigen::Map<HessianType> ret_h(hessian);
    Eigen::Map<ParameterType> ret_b(b);
    ret_h = result.JTJ;
    ret_b = result.JTr;
    return result.squaredError;
  }

 private:
  const In* input_;
  const Out* measurements_;
  size_t num_elements_;
};
}  // namespace moptimizer