#pragma once

namespace moptimizer {

template <class dim_parameter, class dim_output, class Scalar>
struct numeric_jacobian {
  using HessianType = Eigen::Matrix<Scalar, dim_parameter, dim_parameter>;
  using JacobianType = Eigen::Matrix<Scalar, dim_output, dim_parameter>;
  using ParameterType = Eigen::Matrix<Scalar, dim_parameter, 1>;

  numeric_jacobian() { min_step_size_ = std::sqrt(std::numeric_limits<Scalar>::epsilon()); }
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

  ResultPair operator()(const In& in, const Out& out)->ResultPair {
    JacobianType local_jacobian;
    const auto&& residual = model(in, out);
    for (size_t p_dim = 0; p_dim < dim_parameter_; ++p_dim) {
      const auto&& diff = (models_plus[p_dim](in, out) - residual) / min_step_size_;
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

  Scalar min_step_size_;
  constexpr size_t dim_parameter_;
};
}  // namespace moptimizer