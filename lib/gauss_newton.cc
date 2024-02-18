#include <Eigen/Dense>
#include "gauss_newton.hh"

namespace moptimizer {
  template<class Scalar>
  void GaussNewton<Scalar>::step(Scalar* parameters ) const {
    using HessianType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using ParameterType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    HessianType hessian(parameter_dim_,parameter_dim_);
    ParameterType b(parameter_dim_);
    Eigen::Map<ParameterType> parameters_eigen(parameters, parameter_dim_);

    
    for (auto &cost : costs_) {
      cost->linearize(parameters, hessian.data(), b.data());
    }

    ParameterType delta = hessian.ldlt().solve(-b);

    parameters_eigen += delta;

  }
}


// Instantiate
template class moptimizer::GaussNewton<double>;
template class moptimizer::GaussNewton<float>;