#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <numeric>


namespace moptimizer {

/// @brief Cost function class.
/// @tparam In Input data type
/// @tparam Out Measured data type.
/// @tparam Scalar scalar primitive.
template <class In, class Out, class Model, class Scalar = double>
class Cost {
 public:
  Cost() = delete;
  Cost(const Cost& rhs) = delete;

  /// @brief Cost function constructor.
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
    return std::transform_reduce(input_, input_ + num_elements_, measurements_, total,
                                 std::plus<Out>(),  // reduce. Use operator+() of Out.
                                 Model(x));         // Transform
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
    struct ResultPair {
      HessianType JTJ;    // JTJ -> Hessian
      ParameterType JTr;  // JTr -> b
      ResultPair() {
        JTJ.setZero();
        JTr.setZero();
      }
      ResultPair operator+(const ResultPair& rhs) {
        this->JTJ += rhs.JTJ;
        this->JTr += rhs.JTr;
        return *this;
      }
    };

    Eigen::Map<const ParameterType> parameter(x);

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

    // Returns [JTJ, JTb]
    auto numeric_jacobian = [&](const In& in, const Out& out) -> ResultPair {
      JacobianType local_jacobian;
      const auto&& residual = model(in, out);
      for (size_t p_dim = 0; p_dim < dim_parameter; ++p_dim) {
        const auto&& diff = (models_plus[p_dim](in, out) - residual) / min_step_size;
        if constexpr (dim_output == 1)
          local_jacobian[p_dim] = diff;
        else
          local_jacobian.col(p_dim) = diff;
      }

      HessianType&& local_hessian = local_jacobian.transpose() * local_jacobian;
      ParameterType&& local_b = local_jacobian.transpose() * residual;
      ResultPair res;
      res.JTJ = local_hessian;
      res.JTr = local_b;
      return res;
    };
    ResultPair res;
    // Perform reduction.
    res = std::transform_reduce(
        input_, input_ + num_elements_, measurements_, res,
        // Reduce
        [&](ResultPair init, const ResultPair& b) -> ResultPair { return init + b; },
        // Transform (compute jacobian per data input)
        numeric_jacobian);
    // Assign outputs
    Eigen::Map<HessianType> ret_h(hessian);
    Eigen::Map<ParameterType> ret_b(b);
    ret_h = res.JTJ;
    ret_b = res.JTr;
  }

 private:
  const In* input_;
  const Out* measurements_;
  size_t num_elements_;
};
}  // namespace moptimizer