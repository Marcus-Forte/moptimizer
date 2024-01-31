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

using Point2 = Eigen::Vector3d; // We use homonegeous coordinates.

void build_transform(double tx, double ty, double yaw, Eigen::Matrix<double, 3, 3> &transform)
{
    double cy = cos(yaw);
    double sy = sin(yaw);
    transform.setIdentity();
    transform(0, 2) = tx;
    transform(1, 2) = ty;
    transform(0, 0) = cy;
    transform(0, 1) = sy;
    transform(1, 0) = -sy;
    transform(1, 1) = cy;
}

struct measurement_data_
{
    measurement_data_(double rx, double ry) : reference_(rx, ry, 1.0) {}
    measurement_data_(double rx, double ry, double ix, double iy) : reference_(rx, ry, 1.0), input_(ix, iy, 1.0) {}
    Point2 reference_;
    Point2 input_;
};

struct model
{
    model(const double *beta) : beta_(beta)
    {
        // setup transform.
        build_transform(beta_[0], beta_[1], beta_[2], transform_);
    }

    // Run error model.
    inline Point2 operator()(const measurement_data_ &measurement) const
    {
        Point2 ret = measurement.reference_ - transform_ * measurement.input_; /* (*this)(measurement.x_) */
        return ret;
    }

    const double *beta_;
    Eigen::Matrix<double, 3, 3> transform_;
};

std::pair<std::vector<double>, std::vector<double>> get_pt_coords(const std::vector<Point2> &points);
void print_data(const std::vector<measurement_data_> &data);

int main(int argc, char **argv)
{
    std::vector<measurement_data_> correspondences;
    correspondences.push_back({0, 0});
    correspondences.push_back({0, 1});
    correspondences.push_back({1, 0});
    correspondences.push_back({1, 1});

    // transform origin
    Eigen::Matrix<double, 3, 3> transform;

    build_transform(5.0, 5.0, 0.0, transform);

    std::for_each(correspondences.begin(), correspondences.end(), [&transform](measurement_data_ &corrs)
                  { corrs.input_ = transform * corrs.reference_; });

    // Gauss Newton
    Eigen::Matrix<double, 3, 1> x0;
    x0.setZero();
    std::vector<Point2> f_x;
    std::vector<Point2> f_x_plus_;
    const size_t output_vec_size = correspondences.size() * sizeof(Point2) / sizeof(double);
    f_x.resize(correspondences.size());
    f_x_plus_.resize(correspondences.size());
    const double epsilon = 0.0001;
    Eigen::Matrix<double, Eigen::Dynamic, 3> jacobian;
    Eigen::Matrix<double, 3, 3> hessian;
    Eigen::Matrix<double, 3, 1> b;
    jacobian.resize(output_vec_size, Eigen::NoChange);
    Eigen::Map<Eigen::VectorXd> f_plus_map((double *)f_x_plus_.data(), output_vec_size, 1);
    Eigen::Map<Eigen::VectorXd> f_x_map((double *)f_x.data(), output_vec_size, 1);

    for (int iterations = 0; iterations < 10; ++iterations)
    {
        // Compute error
        std::transform(correspondences.begin(), correspondences.end(), f_x.begin(), model(x0.data()));

        // Compute derivative
        for (int p_dim = 0; p_dim < 3; ++p_dim)
        {
            Eigen::Matrix<double, 3, 1> x0_plus(x0);
            x0_plus[p_dim] += epsilon;
            std::transform(correspondences.begin(), correspondences.end(), f_x_plus_.begin(), model(x0_plus.data()));
            jacobian.col(p_dim) = (f_plus_map - f_x_map) / epsilon;
        }

        hessian = jacobian.transpose() * jacobian;
        b = jacobian.transpose() * f_x_map;

        Eigen::Matrix<double, 3, 1> delta;

        Eigen::LDLT<Eigen::Matrix<double, 3, 3>> solver(hessian);

        delta = solver.solve(-b);

        x0 += delta;

        std::cout << x0 << std::endl;

#ifdef USE_MATPLOTLIB
        // TODO
#endif
    }
}

void print_data(const std::vector<measurement_data_> &data)
{
    std::cout << "input: \n\n";
    std::for_each(data.begin(), data.end(), [](const measurement_data_ &data_)
                  { std::cout << data_.input_ << "\n\n"; });
    std::cout << "refs: \n\n";
    std::for_each(data.begin(), data.end(), [](const measurement_data_ &data_)
                  { std::cout << data_.reference_ << "\n\n"; });
}
