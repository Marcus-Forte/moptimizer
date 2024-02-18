#pragma once

struct simple : public moptimizer::Model<double, double, double> {
  simple(const double* p) : Model(p) {}
  double operator()(const double& in, const double& measured) const override {
    return measured - (parameters_[0] * in) / (parameters_[1] + in);
  }
};

struct simple_jacobian : public moptimizer::JacobianModel<double, Eigen::Matrix<double, 1, 2>, double> {
  simple_jacobian(const double* p) : JacobianModel(p) {}
  Eigen::Matrix<double, 1, 2> operator()(const double& input) const override {
    Eigen::Matrix<double, 1, 2> ret;
    ret[0] = -input / (parameters_[1] + input);
    ret[1] = (parameters_[0] * input) / ((parameters_[1] + input) * (parameters_[1] + input));
    return ret;
  }
};

struct curve_fitting : public moptimizer::Model<double, double, double> {
  curve_fitting(const double* parameters) : Model(parameters) {}
  double operator()(const double& x, const double& measured) const override {
    return measured - exp(parameters_[0] * x + parameters_[1]);
  }
};

struct curve_fitting_jacobian : public moptimizer::JacobianModel<double, Eigen::Matrix<double, 1, 2>, double> {
  curve_fitting_jacobian(const double* parameters) : JacobianModel(parameters) {}
  Eigen::Matrix<double, 1, 2> operator()(const double& x) const override {
    Eigen::Matrix<double, 1, 2> ret;
    ret[0] = -x * exp(parameters_[0] * x + parameters_[1]);
    ret[1] = -exp(parameters_[0] * x + parameters_[1]);
    return ret;
  }
};
