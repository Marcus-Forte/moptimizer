#include <Eigen/Dense>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <thread>
#include <random>

#include <matplotlibcpp.h>
namespace plt = matplotlibcpp;

struct gauss_curve_param
{
    gauss_curve_param(double a0, double a1, double a2, double a3) : peak(a0), height(a1), l_width(a2), r_width(a3) {}
    double peak;
    double height;
    double l_width;
    double r_width;
};

double gauss_dissym(const gauss_curve_param &parameters, const double x)
{
    const double factor = x - parameters.peak;
    double y;
    if (x < parameters.peak)
        y = parameters.height * exp(-factor * factor / parameters.l_width);
    else
        y = parameters.height * exp(-factor * factor / parameters.r_width);

    return y;
}

std::vector<double> gauss_dissym_v(const gauss_curve_param &parameters, const std::vector<double> &x)
{
    std::vector<double> y(x.size());

    for (int i = 0; i < x.size(); ++i)
        y[i] = gauss_dissym(parameters, x[i]);

    return y;
}

int main()
{
    const int data_size = 200;
    gauss_curve_param p(0, 10, 1, 0.3);
    std::vector<double> x(data_size);
    double min = -3.0;
    double max = 3.0;
    double step = (max - min) / data_size;
    for (int i = 0; i < data_size; ++i)
    {
        x[i] = min + i * step;
    }
    std::vector<double> y = gauss_dissym_v(p, x);

    std::default_random_engine generator;
    std::normal_distribution<double> dist(0.0, 0.6);

    std::transform(y.begin(), y.end(), y.begin(), [&](double x)
                   { return x + dist(generator); });

    {
        const int dim_param = 4;
        Eigen::Matrix<double, dim_param, 1> x0;
        Eigen::Matrix<double, data_size, dim_param> jacobian;
        Eigen::Matrix<double, dim_param, dim_param> hessian;
        Eigen::Matrix<double, dim_param, 1> b;

        x0.setZero();
        x0[0] = -0.30;
        x0[1] = 2;
        x0[2] = 0.3;
        x0[3] = 0.1;

        Eigen::Matrix<double, data_size, 1> f_x;
        Eigen::Matrix<double, data_size, 1> f_x_plus;
        Eigen::Matrix<double, data_size, 1> f_x_minus;

        // Build parameters
        for (int iterations = 0; iterations < 15; ++iterations)
        {
            const gauss_curve_param param(x0[0], x0[1], x0[2], x0[3]);

            // Compute error
            for (int i = 0; i < data_size; ++i)
            {
                f_x[i] = y[i] - gauss_dissym(param, x[i]);
            }

            // Compute jacobian
            const double epsilon = 1e-7;
            for (int p_dim = 0; p_dim < dim_param; ++p_dim)
            {
                Eigen::Matrix<double, dim_param, 1> x0_plus(x0);
                Eigen::Matrix<double, dim_param, 1> x0_minus(x0);
                x0_plus[p_dim] += epsilon;
                x0_minus[p_dim] -= epsilon;

                // std::cout << x0_plus << " || " << x0_minus << std::endl;

                // exit(0);
                const gauss_curve_param param_plus(x0_plus[0], x0_plus[1], x0_plus[2], x0_plus[3]);
                const gauss_curve_param param_minus(x0_minus[0], x0_minus[1], x0_minus[2], x0_minus[3]);
                for (int i = 0; i < data_size; ++i)
                {
                    f_x_plus[i] = y[i] - gauss_dissym(param_plus, x[i]);
                    f_x_minus[i] = y[i] - gauss_dissym(param_minus, x[i]);
                    jacobian(i, p_dim) = (f_x_plus[i] - f_x_minus[i]) / (2 * epsilon);
                }
            }

            hessian = jacobian.transpose() * jacobian;
            b = jacobian.transpose() * f_x;

            Eigen::Matrix<double, dim_param, 1> delta;

            Eigen::LDLT<Eigen::Matrix<double, dim_param, dim_param>> solver(hessian);

            delta = solver.solve(-b);

            x0 += delta;

            std::cout << hessian << "\n\n";
            std::cout << x0 << "\n\n";

            plt::clf();

            std::vector<double> y_opt = gauss_dissym_v(param, x);

            plt::plot(x, y);
            plt::plot(x, y_opt, "r--");
            plt::grid(true);

            plt::pause(0.25);
        }
    }
}