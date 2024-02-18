#pragma once

namespace moptimizer {

/// @brief Model interface
/// @tparam In
/// @tparam Out
/// @tparam Scalar
template <class In, class Out, class Scalar>
class Model {
 public:
  Model(const Scalar* parameters) : parameters_(parameters) {}
  virtual Out operator()(const In& input, const Out& measurement) const = 0;

 protected:
  const Scalar* parameters_;
};

/// @brief Jacobian model interface
/// @tparam In input data type
/// @tparam JacobianOut jacobian matrix type
/// @tparam Scalar
template <class In, class JacobianOut, class Scalar>
class JacobianModel {
 public:
  JacobianModel(const Scalar* parameters) : parameters_(parameters) {}
  virtual JacobianOut operator()(const In& input) const = 0;

 protected:
  const Scalar* parameters_;
};
}  // namespace moptimizer