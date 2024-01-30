#include <Eigen/Dense>
#include <algorithm>
#include <iostream>
#include <numeric>

#include "util.hh"
#include "data_loader.hh"

#ifdef USE_MATPLOTLIB
#include <matplotlibcpp.h>
namespace plt = matplotlibcpp;
#endif

struct measurement_data_
{
    double x_;
    double y_;
};

struct model
{
    model(const double *beta) : beta_(beta) {}

    // Run error model.
    inline double operator()(const measurement_data_ &measurement) const
    {
        return measurement.y_ - (*this)(measurement.x_) /* (*this)(measurement.x_) */;
    }

    // y = f(x)
    inline double operator()(const double x) const
    {
        const double factor0 = x - beta_[3];
        const double factor1 = x - beta_[6];
        return beta_[0] * exp(-beta_[1] * x) + beta_[2] * exp(-factor0 * factor0 / beta_[4]) + beta_[5] * exp(-factor1 * factor1 / beta_[7]);
    }

    const double *beta_;
};

int main(int argc, char **argv)
{
    double lm_lambda = 10.0;

    if (argc > 1)
        lm_lambda = atoi(argv[1]);

    std::vector<double> data_x;
    std::vector<double> data_y;
    loadData("../data/gauss1.txt", data_x, data_y);

    int data_size = data_x.size();
    std::cout << data_size << "\n";
    const int num_parameters = 8;
    Eigen::Matrix<double, num_parameters, 1> x0;

    Eigen::Matrix<double, Eigen::Dynamic, num_parameters> jacobian;
    Eigen::Matrix<double, num_parameters, num_parameters> hessian;

    Eigen::Matrix<double, num_parameters, 1> b;
    Eigen::Matrix<double, Eigen::Dynamic, 1> f_x;
    Eigen::Matrix<double, Eigen::Dynamic, 1> f_x_plus_;
    const float epsilon = 0.0001;
    // // define parameter vector.
    x0.setZero();
    x0[0] = 100;
    x0[1] = 0.01;
    x0[2] = 1;
    x0[3] = 100;
    x0[4] = 1;
    x0[5] = 1;
    x0[6] = 150;
    x0[7] = 20;

    std::vector<measurement_data_> dataset;
    for (int i = 0; i < data_size; ++i)
    {
        dataset.push_back({data_x[i], data_y[i]});
    }

    f_x.resize(data_size);
    f_x_plus_.resize(data_size);
    jacobian.resize(data_size, Eigen::NoChange);

    // // Gauss Newton
    for (int iterations = 0; iterations < 100; ++iterations)
    {
        // Compute error
        std::transform(dataset.begin(), dataset.end(), f_x.begin(), model(x0.data()));

        // std::cout << f_x << std::endl;
        // exit(0);

        double error_sum = std::transform_reduce(
            dataset.begin(), dataset.end(), 0.0f, [](double init, double b)
            { return init + b * b; },
            model(x0.data()));

        // // Compute derivative

        for (int p_dim = 0; p_dim < num_parameters; ++p_dim)
        {
            Eigen::Matrix<double, num_parameters, 1> x0_plus(x0);
            x0_plus[p_dim] += epsilon;
            std::transform(dataset.begin(), dataset.end(), f_x_plus_.begin(), model(x0_plus.data()));
            jacobian.col(p_dim) = (f_x_plus_ - f_x) / epsilon;
        }

        hessian = jacobian.transpose() * jacobian;

        b = jacobian.transpose() * f_x;

        Eigen::Matrix<double, num_parameters, 1> delta;

        Eigen::Matrix<double, num_parameters, num_parameters> hessian_diagonal;
        hessian_diagonal = hessian.diagonal().asDiagonal();
        Eigen::LDLT<Eigen::Matrix<double, num_parameters, num_parameters>> solver(hessian + lm_lambda * hessian_diagonal);

        delta = solver.solve(-b);

        // std::cout << "delta = " << delta << std::endl;

        x0 += delta;

        std::cout << "Error = " << error_sum << "\n";

#ifdef USE_MATPLOTLIB
        const size_t num_elements_model_curve = 100;
        plt::clf();
        plt::plot(data_x, data_y, ".");
        auto x_data_model = util::linspace(0.0, 250.0, num_elements_model_curve);
        std::vector<double> y_data_model(num_elements_model_curve);
        std::transform(x_data_model.begin(), x_data_model.end(), y_data_model.begin(), model(x0.data()));
        plt::plot(x_data_model, y_data_model, "r--");
        plt::pause(0.1);
#endif
    }
    std::cout << "Final X = " << x0 << std::endl;
}