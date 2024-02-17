#pragma once

namespace moptimizer {

template <class Sum>
struct reduce_functor {
 public:
  Sum operator()(const Sum& init, const Sum& b) const { return init + b; }
};

template <class LinearizePair, class JacobianType>
struct reduce_functor_jacobian {
  LinearizePair operator()(const LinearizePair& init, const LinearizePair& b) const { return init + b; }

  // This one looks that could have been avoided...
  LinearizePair operator()(const JacobianType& a, const JacobianType& b) const {
    return a.transpose() * a + b.transpose() * b;
  }

  LinearizePair operator()(const LinearizePair& init, const JacobianType& b) const { return init + b.transpose() * b; }
};

// template <class Model, class In, class Out, class JacobianType>
// struct transform_jacobian_numeric {
//   transform_jacobian_numeric(const std::vector<Model>& models_plus, const Model& model, size_t p_dim, constexpr size_t dim_output)
//       : models_plus_(models_plus), model_(model), p_dim_(p_dim) ,dim_output_(dim_output) {}

//   JacobianType operator()(const In& in, const Out& out) const {
//     JacobianType local_jacobian;
//     for (size_t p_dim = 0; p_dim < p_dim_; ++p_dim) {
//       const auto&& diff = (models_plus_[p_dim](in, out) - model_(in, out)) / epsilon;
//       if constexpr (dim_output_ == 1)
//         local_jacobian[p_dim] = diff;
//       else
//         local_jacobian.col(p_dim) = diff;
//     }
//     return local_jacobian;
//   }

//   const std::vector<Model>& models_plus_;
//   const Model& model_;
//   size_t p_dim_;
//   constexpr size_t dim_output_;
// };
}  // namespace moptimizer

// struct tst_functor {
//   // All the operators must be implemented.
//   using Sum = Eigen::Matrix<double, 2, 2>;
//   using Ret = Eigen::Matrix<double, 1, 2>;
//   struct ResultType {
//     Sum hessian;
//     Eigen::Matrix<double, 2, 1> b;
//   };

//   Sum operator()(const Sum& init, const Sum& b) const {
//     // std::cout << "Init + b\n";
//     return init + b;
//   }

//   // This one looks that could have been avoided...
//   Sum operator()(const Ret& a, const Ret& b) const {
//     // std::cout << "Ret + Ret\n";
//     return a.transpose() * a + b.transpose() * b;
//   }
//   Sum operator()(const Sum& init, const Ret& b) const {
//     // std::cout << "Init + Ret\n";
//     return init + b.transpose() * b;
//   }
// };