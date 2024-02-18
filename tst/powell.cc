#include "cost.hh"
#include "gtest/gtest.h"
#include "model.hh"
#include "test_models.h"

TEST(PowellModel, Regression) {
  Eigen::Vector4d parameters = {3, -1, 0, 4};
  constexpr int parameter_dimension = parameters.rows();

  moptimizer::Cost<parameter_dimension, double, Eigen::Vector4d, powell> cost(nullptr, nullptr, 1);

  Eigen::Matrix<double, parameter_dimension, parameter_dimension> hessian_numeric;
  Eigen::Matrix<double, parameter_dimension, 1> b_numeric;

  for (int i = 0; i < 25; ++i) {
    cost.linearize(parameters.data(), hessian_numeric.data(), b_numeric.data());

    Eigen::Vector4d delta = hessian_numeric.ldlt().solve(-b_numeric);

    parameters += delta;
  }

  EXPECT_NEAR(parameters[0], 0.0, 1e-5);
  EXPECT_NEAR(parameters[1], 0.0, 1e-5);
  EXPECT_NEAR(parameters[2], 0.0, 1e-5);
  EXPECT_NEAR(parameters[3], 0.0, 1e-5);
}