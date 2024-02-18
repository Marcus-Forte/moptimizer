// #include <thrust/device_vector.h>
// #include <thrust/sequence.h>

#include <Eigen/Dense>
#include <algorithm>
#include <execution>
#include <iostream>
#include <numeric>

#include "cost.hh"

//   F = 1/2 (f1^2 + f2^2 + f3^2 + f4^2)
//
//   f1 = x1 + 10*x2;
//   f2 = sqrt(5) * (x3 - x4)
//   f3 = (x2 - 2*x3)^2
//   f4 = sqrt(10) * (x1 - x4)^2

// struct Model {
//   Eigen::Vector4d operator()(const double* x) const {
//     double&& fx0 = x[0] + 10 * x[1];
//     double&& fx1 = sqrt(5) * (x[2] - x[3]);
//     double&& fx2 = (x[1] - 2 * x[2]) * (x[1] - 2 * x[2]);
//     double&& fx3 = sqrt(10) * (x[0] - x[3]) * (x[0] - x[3]);
//     return {fx0, fx1, fx2, fx3};
//   }
// };

// // Jacobian
// struct ModelJ {
//   Eigen::Matrix4d operator()(const double* x) const {
//     Eigen::Matrix4d jacobian;
//     jacobian(0, 0) = 1;
//     jacobian(1, 0) = 0;
//     jacobian(2, 0) = 0;
//     jacobian(3, 0) = sqrt(10) * 2 * (x[0] - x[3]);

//     // Df / dx1
//     jacobian(0, 1) = 10;
//     jacobian(1, 1) = 0;
//     jacobian(2, 1) = 2 * (x[1] + 2 * x[2]);
//     jacobian(3, 1) = 0;

//     // Df / dx2
//     jacobian(0, 2) = 0;
//     jacobian(1, 2) = sqrt(5);
//     jacobian(2, 2) = 2 * (x[1] + 2 * x[2]) * (-2);
//     jacobian(3, 2) = 0;

//     // Df / dx3
//     jacobian(0, 3) = 0;
//     jacobian(1, 3) = -sqrt(5);
//     jacobian(2, 3) = 0;
//     jacobian(3, 3) = sqrt(10) * 2 * (x[0] - x[3]) * (-1);
//     return jacobian;
//   }
// };

double x_data[7] = {0.038, 0.194, 0.425, 0.626, 1.253, 2.5, 3.74};
double y_data[7] = {0.05, 0.127, 0.094, 0.2122, 0.2729, 0.2665, 0.3317};

struct model {
  model(const double* p) : p_(p) {}
  double operator()(const double& in, const double& measured) const { return measured - (p_[0] * in) / (p_[1] + in); }

  const double* p_;
};

struct model_jacobian {
  model_jacobian(const double* p) : parameters_(p) {}

  Eigen::Matrix<double, 1, 2> operator()(const double& input) const {
    Eigen::Matrix<double, 1, 2> ret;
    ret[0] = -input / (parameters_[1] + input);
    ret[1] = (parameters_[0] * input) / ((parameters_[1] + input) * (parameters_[1] + input));
    return ret;
  }

  const double* parameters_;
};

int main() {
  Eigen::Vector2d parameters{0.1, 0.1};

  moptimizer::Cost<double, double, model> cost(x_data, y_data, 7);

  auto total = cost.sum(parameters.data());
  std::cout << "total = " << total << "\n";

  Eigen::Matrix<double, 2, 2> hessian;
  Eigen::Matrix<double, 2, 1> b;

  cost.linearize<2>(parameters.data(), hessian.data(), b.data());

  std::cout << "Class results: \n";
  std::cout << hessian << std::endl;
  std::cout << b << std::endl;

  Eigen::Matrix<double, 7, 1> f_x;
  Eigen::Matrix<double, 7, 1> f_x_plus;
  Eigen::Matrix<double, 7, 2> jacobian;
  std::transform(x_data, x_data + 7, y_data, f_x.begin(), model(parameters.data()));

  // std::cout << "f_x = " << f_x << std::endl;

  for (int dim = 0; dim < 2; ++dim) {
    const double epsilon = 1e-5;
    Eigen::Vector2d p_plus(parameters);
    p_plus[dim] += epsilon;

    std::transform(x_data, x_data + 7, y_data, f_x_plus.begin(), model(p_plus.data()));

    jacobian.col(dim) = (f_x_plus - f_x) / epsilon;
  }
  // std::cout << "manual jacobian: \n";
  // std::cout << jacobian << "\n";
  Eigen::Matrix<double, 2, 2> manual_hess = jacobian.transpose() * jacobian;
  Eigen::Matrix<double, 2, 1> manual_b = jacobian.transpose() * f_x;
  std::cout << "manual results: \n";
  std::cout << manual_hess << "\n";
  std::cout << manual_b << "\n";
  // Numerical jac
  // for (int it = 0; it < 10; ++it) {
  //   std::transform(x_data, x_data + 7, y_data, f_x.begin(), model(parameters.data()));

  //   for (int dim = 0; dim < 2; ++dim) {
  //     const double epsilon = 1e-5;
  //     Eigen::Vector2d p_plus(parameters);
  //     p_plus[dim] += epsilon;

  //     std::transform(x_data, x_data + 7, y_data, f_x_plus.begin(), model(p_plus.data()));

  //     jacobian.col(dim) = (f_x_plus - f_x) / epsilon;
  //   }
  //   // Numeric
  //   Eigen::Matrix<double, 2, 2> hessian_numeric_reduce;
  //   hessian_numeric_reduce.setZero();
  //   auto sum_numeric =
  //       std::transform_reduce(x_data, x_data + 7, y_data, hessian_numeric_reduce,
  //                             tst_functor(),  // reduce
  //                             [&parameters](const double input, const double measure) {

  //                               Eigen::Matrix<double, 1, 2> ret;
  //                               auto model_ = model(parameters.data());
  //                               const double epsilon = 1e-5;
  //                               for (int dim = 0; dim < 2; ++dim) {
  //                                 Eigen::Vector2d p_plus(parameters);
  //                                 p_plus[dim] += epsilon;
  //                                 auto model_plus = model(p_plus.data());
  //                                 ret[dim] = (model_plus(input, measure) - model_(input, measure)) / epsilon;
  //                               }
  //                               return ret;
  //                             });  // transform

  // std::cout << "Numeric Reduce: " << sum_numeric << std::endl;
  //   // Analytic
  //   Eigen::Matrix<double, 2, 2> hessian_reduce;
  //   hessian_reduce.setZero();
  //   auto sum =
  //       std::transform_reduce( x_data, x_data + 7, y_data, hessian_reduce,
  //                             tst_functor(),  // reduce
  //                             [&parameters](const double input, const double measure) {
  //                               ; return ret;
  //                             });  // transform

  //   std::cout << "Analytic Reduce:" << sum << std::endl;

  //   Eigen::Matrix<double, 2, 2> hessian = jacobian.transpose() * jacobian;
  //   Eigen::Matrix<double, 2, 1> b = jacobian.transpose() * f_x;
  //   std::cout << "numerical:" << hessian << std::endl;

  //   // Solve
  //   Eigen::Vector2d delta = hessian.ldlt().solve(-b);
  //   parameters += delta;

  //   std::cout << "parameters: " << parameters.transpose() << "\n";
  // }
}
// ::transform takes 1 or 2 inputs and needs to assign outputs..
// ::for_each only takes 1 inputsEigen::Matrix<double, 1, 2> ret;
//                               ret[0] = -input / (parameters[1] + input);
//                               ret[1] = (parameters[0] * input) / ((parameters[1] + input) * (parameters[1] +
//                               input))