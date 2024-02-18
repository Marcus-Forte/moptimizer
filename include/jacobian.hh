#pragma once

#include <Eigen/Dense>
namespace moptimizer {

/// @brief Jacobian result pair.
/// Contains:
/// Hessian (JacobianTranspose * Jacobian) and
/// B vector (JacobianTranpose * residuals)
/// @tparam Scalar
/// @tparam dim_parameter
template <int dim_parameter, class Scalar>
struct jacobian_result_pair {
  using HessianType = Eigen::Matrix<Scalar, dim_parameter, dim_parameter>;
  using ParameterType = Eigen::Matrix<Scalar, dim_parameter, 1>;
  jacobian_result_pair() {
    JTJ.setZero();
    JTr.setZero();
  }

  jacobian_result_pair operator+(const jacobian_result_pair& rhs) {
    this->JTJ += rhs.JTJ;
    this->JTr += rhs.JTr;
    return *this;
  }

  HessianType JTJ;    // JTJ -> Hessian
  ParameterType JTr;  // JTr -> b
};

/// @brief Numerical jacobian functor.
/// @tparam Model
/// @tparam In
/// @tparam Out
/// @tparam Scalar
/// @tparam dim_parameter
/// @tparam dim_output
template <int dim_parameter, int dim_output, class ModelT, class In, class Out, class Scalar>
struct numeric_jacobian {
  using HessianType = Eigen::Matrix<Scalar, dim_parameter, dim_parameter>;
  using JacobianType = Eigen::Matrix<Scalar, dim_output, dim_parameter>;
  using ParameterType = Eigen::Matrix<Scalar, dim_parameter, 1>;
  using ResultPair = jacobian_result_pair<dim_parameter, Scalar>;

  numeric_jacobian(const Scalar* x) : model_(x) {
    Eigen::Map<const ParameterType> parameter(x);
    parameters_plus.resize(dim_parameter);
    min_step_size_ = std::sqrt(std::numeric_limits<Scalar>::epsilon());
    for (size_t p_dim = 0; p_dim < dim_parameter; ++p_dim) {
      parameters_plus[p_dim] = parameter;
      parameters_plus[p_dim][p_dim] += min_step_size_;
      models_plus_.push_back(ModelT(parameters_plus[p_dim].data()));
    }
  }

  ResultPair operator()(const In& input, const Out& measurement) const {
    JacobianType local_jacobian;
    const auto&& residual = model_(input, measurement);
    for (size_t p_dim = 0; p_dim < dim_parameter; ++p_dim) {
      const auto&& diff = (models_plus_[p_dim](input, measurement) - residual) / min_step_size_;
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
  }

  Scalar min_step_size_;
  ModelT model_;
  std::vector<ModelT> models_plus_;
  std::vector<ParameterType> parameters_plus;
};

/// @brief Analytic jacobian functor.
/// @tparam JacobianModel
/// @tparam In
/// @tparam Out
/// @tparam Scalar
/// @tparam Model
/// @tparam dim_parameter
/// @tparam dim_output
template <int dim_parameter, int dim_output, class ModelT, class JacobianModelT, class In, class Out, class Scalar>
struct analytic_jacobian {
  using HessianType = Eigen::Matrix<Scalar, dim_parameter, dim_parameter>;
  using JacobianType = Eigen::Matrix<Scalar, dim_output, dim_parameter>;
  using ParameterType = Eigen::Matrix<Scalar, dim_parameter, 1>;
  using ResultPair = jacobian_result_pair<dim_parameter, Scalar>;

  analytic_jacobian(const Scalar* x) : model_(x), jacobian_model_(x) { Eigen::Map<const ParameterType> parameter(x); }

  ResultPair operator()(const In& input, const Out& measurement) const {
    JacobianType&& local_jacobian = jacobian_model_(input);
    const auto&& residual = model_(input, measurement);
    HessianType&& local_hessian = local_jacobian.transpose() * local_jacobian;
    ParameterType&& local_b = local_jacobian.transpose() * residual;
    ResultPair res;
    res.JTJ = local_hessian;
    res.JTr = local_b;
    return res;
  }

  ModelT model_;
  JacobianModelT jacobian_model_;
};
}  // namespace moptimizer