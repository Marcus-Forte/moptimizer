#pragma once

namespace moptimizer {
template <class Scalar>
class ICost {
 public:
  // virtual Scalar sum(const Scalar* x)
  virtual Scalar linearize(const Scalar* x, Scalar* hessian, Scalar* b) const = 0;
};
}  // namespace moptimizer