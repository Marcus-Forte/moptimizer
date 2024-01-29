#include <Eigen/Dense>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <thread>

#include <matplotlibcpp.h>
// #include <mat
namespace plt = matplotlibcpp;

const int data_size = 7;
const std::vector<double> x_data{0.038, 0.194, 0.425, 0.626, 1.253, 2.5, 3.70};
const std::vector<double> y_data{0.05, 0.127, 0.094, 0.2122, 0.2729, 0.2665, 0.3317};

// Define function
void reaction_rate(const double *x, double *f_x, const double *data_x, const double *data_y, int index)
{
    f_x[0] = data_y[index] - (x[0] * data_x[index]) / (x[1] + data_x[index]);
}


int main()
{
    Eigen::Matrix<double, 2, 1> x0;
    Eigen::Matrix<double, data_size, 2> jacobian;
    Eigen::Matrix<double, 2, 2> hessian;
    Eigen::Matrix<double, 2, 1> b;
    const float epsilon = 0.01;
    // define parameter vector.
    x0.setZero();

    // Compute sum
    for (int iterations = 0; iterations < 15; ++iterations)
    {
        double f_x[data_size];
        double f_x_plus[data_size];

        // Comput error
        for (int i = 0; i < data_size; ++i)
        {
            reaction_rate(x0.data(), &f_x[i], x_data.data(), y_data.data(), i);
        }

        // Compute derivative
        for (int p_dim = 0; p_dim < 2; ++p_dim)
        {
            Eigen::Matrix<double, 2, 1> x0_plus(x0);
            x0_plus[p_dim] += epsilon;
            for (int i = 0; i < data_size; ++i)
            {
                reaction_rate(x0_plus.data(), &f_x_plus[i], x_data.data(), y_data.data(), i);
                jacobian(i, p_dim) = (f_x_plus[i] - f_x[i]) / epsilon;
            }
        }

        Eigen::Map<Eigen::Matrix<double, data_size, 1>> residuals(f_x);
        hessian = jacobian.transpose() * jacobian;
        b = jacobian.transpose() * residuals;

        Eigen::Matrix<double, 2, 1> delta;

        Eigen::LDLT<Eigen::Matrix<double, 2, 2>> solver(hessian);

        delta = solver.solve(-b);

        x0 += delta;

        std::vector<double> r_vec(f_x, f_x + data_size);
        plt::clf();

        plt::plot(x_data, y_data, ".");
        // draw graph
        std::vector<double> x_data_fine(100);
        float val = 0.0f;
        for (int j = 0; j < 100; j++, val += 0.037)
        {
            x_data_fine[j] = val;
        }
        std::vector<double> y_data_fine(100);
        std::transform(x_data_fine.begin(), x_data_fine.end(), y_data_fine.begin(), [&x0](double x)
                       { return (x0[0] * x) / (x0[1] + x); });
        plt::plot(x_data_fine, y_data_fine, "r--");

        plt::pause(1.0);
    }

    std::cout << "x = " << x0 << std::endl;

    return 0;
}