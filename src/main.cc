#include <Eigen/Dense>
#include <algorithm>
#include <iostream>
#include <numeric>

#include <matplotlibcpp.h>
namespace plt = matplotlibcpp;

const double x_data[]{0.038, 0.194, 0.425, 0.626, 1.253, 2.5, 3.70};
const double y_data[]{0.05, 0.127, 0.094, 0.2122, 0.2729, 0.2665, 0.3317};

const size_t data_size = sizeof(x_data) / sizeof(double);

struct data_
{
    double x_;
    double y_;
};

double reaction_rate(const double x[2], data_ measured)
{
    return measured.y_ - (x[0] * measured.x_) / (x[1] + measured.x_);
}

struct model
{
    model(const double *x) : x_(x)
    {
    }

    double operator()(data_ measurement)
    {
        return reaction_rate(x_, measurement);
    }

    const double *x_;
};

int main()
{
    Eigen::Matrix<double, 2, 1> x0;
    Eigen::Matrix<double, data_size, 2> jacobian;
    Eigen::Matrix<double, 2, 2> hessian;
    Eigen::Matrix<double, 2, 1> b;
    Eigen::Matrix<double, data_size, 1> f_x;
    const float epsilon = 0.01;
    // define parameter vector.
    x0.setZero();

    std::vector<data_> dataset;

    for (int i = 0; i < data_size; ++i)
    {
        dataset.push_back({x_data[i], y_data[i]});
    }

    // Gauss Newton
    for (int iterations = 0; iterations < 15; ++iterations)
    {
        // Compute error
        std::transform(dataset.begin(), dataset.end(), f_x.begin(), model(x0.data()));

        double error_sum = std::transform_reduce(
            dataset.begin(), dataset.end(), 0.0f, [](double a, double b)
            { return a * a + b * b; },
            model(x0.data()));

        // Compute derivative
        Eigen::Matrix<double, data_size, 1> f_x_plus_;
        for (int p_dim = 0; p_dim < 2; ++p_dim)
        {
            Eigen::Matrix<double, 2, 1> x0_plus(x0);
            x0_plus[p_dim] += epsilon;
            std::transform(dataset.begin(), dataset.end(), f_x_plus_.begin(), model(x0_plus.data()));
            jacobian.col(p_dim) = (f_x_plus_ - f_x) / epsilon;
        }

        hessian = jacobian.transpose() * jacobian;
        b = jacobian.transpose() * f_x;

        Eigen::Matrix<double, 2, 1> delta;

        Eigen::LDLT<Eigen::Matrix<double, 2, 2>> solver(hessian);

        delta = solver.solve(-b);

        x0 += delta;

        std::cout << "Error = " << error_sum << "\n";

        plt::clf();

        plt::plot(x_data, y_data, ".");
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
}