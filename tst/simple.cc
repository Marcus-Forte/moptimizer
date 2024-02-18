#include "cost.hh"
#include "gtest/gtest.h"
#include "model.hh"
#include "test_models.h"

const int num_observations = 7;

static double data_x[7] = {0.038, 0.194, 0.425, 0.626, 1.253, 2.5, 3.74};
static double data_y[7] = {0.05, 0.127, 0.094, 0.2122, 0.2729, 0.2665, 0.3317};

TEST(SimpleModel, Derivatives) {
  Eigen::Vector2d parameters = {0., 0.};
  constexpr int parameter_dimension = parameters.rows();

  Eigen::Matrix<double, parameter_dimension, parameter_dimension> hessian_numeric;
  Eigen::Matrix<double, parameter_dimension, 1> b_numeric;
  Eigen::Matrix<double, parameter_dimension, parameter_dimension> hessian_analytic;
  Eigen::Matrix<double, parameter_dimension, 1> b_analytic;
  moptimizer::Cost<double, double, curve_fitting> cost_numeric(data_x, data_y, num_observations);
  moptimizer::Cost<double, double, curve_fitting, curve_fitting_jacobian> cost_analytic(data_x, data_y,
                                                                                        num_observations);

  cost_numeric.linearize<parameter_dimension>(parameters.data(), hessian_numeric.data(), b_numeric.data());
  cost_analytic.linearize<parameter_dimension>(parameters.data(), hessian_analytic.data(), b_analytic.data());
  for (int i = 0; i < hessian_analytic.size(); ++i) {
    EXPECT_NEAR(hessian_analytic(i), hessian_numeric(i), 1e-4);
  }

  EXPECT_NEAR(b_analytic[0], b_numeric[0], 1e-4);
  EXPECT_NEAR(b_analytic[1], b_numeric[1], 1e-4);
}