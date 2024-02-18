#pragma once

#include <iostream>
#include <vector>

#include "Icost.hh"

namespace moptimizer {
template <class Scalar>
class ISolver {
 public:
  using CostT = ICost<Scalar>;
  using HessianType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  using ParameterType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
  ISolver(int parameter_dim) : parameter_dim_(parameter_dim), max_iterations_(10) {}
  virtual ~ISolver() = default;
  void addCost(CostT* cost) { costs_.push_back(cost); }

  void setMaxIterations(int max_iterations) { max_iterations_ = max_iterations; }
  virtual void step(Scalar* parameters) const = 0;
  virtual void minimize(Scalar* parameters) const {
    Eigen::Map<ParameterType> parameters_map(parameters, parameter_dim_);
    HessianType hessian(parameter_dim_, parameter_dim_);
    ParameterType b(parameter_dim_);
    ParameterType delta(parameter_dim_);

    Scalar cost_value = std::numeric_limits<Scalar>::max();
    int iterations = 0;

    for (; iterations < max_iterations_; ++iterations) {
      hessian.setZero();
      b.setZero();
      Scalar total_cost_value = 0.;
      for (const auto& cost : costs_) {
        HessianType local_hessian(parameter_dim_, parameter_dim_);
        ParameterType local_b(parameter_dim_);
        local_hessian.setZero();
        local_b.setZero();
        total_cost_value += cost->linearize(parameters, local_hessian.data(), local_b.data());
        hessian += local_hessian;
        b += local_b;
      }

      if (isCostSmall(total_cost_value)) return;
      // Solve
      delta = hessian.ldlt().solve(-b);
      parameters_map += delta;
    }
  }

 protected:
  std::vector<CostT*> costs_;
  int max_iterations_;

  // Problem space
  int parameter_dim_;

  bool isCostSmall(const Scalar cost_value) const {
    if (std::abs(cost_value) < 8 * (std::numeric_limits<Scalar>::epsilon())) return true;
    return false;
  }
};
}  // namespace moptimizer