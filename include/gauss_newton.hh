#pragma once

#include "Isolver.hh"

namespace moptimizer {
template <class Scalar>
class GaussNewton : public ISolver<Scalar> {
 public:
  GaussNewton(int dim_parameter) : ISolver<Scalar>(dim_parameter) {}
  void step(Scalar* parameters) const override;

 protected:
  using ISolver<Scalar>::costs_;
  using ISolver<Scalar>::parameter_dim_;
};
};  // namespace moptimizer