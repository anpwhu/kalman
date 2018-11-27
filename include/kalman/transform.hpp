#ifndef TRANSFORM_HPP
#define TRANSFORM_HPP

#include <Eigen/Geometry>

namespace transform {

/// returns the 3D cross product skew symmetric matrix of a given 3D vector
template<class Derived>
  inline Eigen::Matrix<typename Derived::Scalar, 3, 3> skew(const Eigen::MatrixBase<Derived> & vec)
  {
      typedef typename Derived::Scalar Scalar;
    EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived, 3);
    return (Eigen::Matrix<typename Derived::Scalar, 3, 3>() << Scalar(0.0), -vec[2], vec[1], vec[2],
            Scalar(0.0), -vec[0], -vec[1], vec[0], Scalar(0.0)).finished();
  }

template<class Derived>
  Eigen::Matrix<typename Derived::Scalar, 3, 1> unskew(const Eigen::MatrixBase<Derived> & Omega) {
      typedef typename Derived::Scalar Scalar;
     return Scalar(0.5) * Eigen::Matrix<typename Derived::Scalar, 3, 1>(
                 Omega(2,1) - Omega(1,2),
                 Omega(0,2) - Omega(2,0),
                 Omega(1,0) - Omega(0,1));
  }

template<typename T>
Eigen::Transform<T, 3, Eigen::Isometry> exp(const Eigen::Matrix<T, 6, 1>& delta) {
    // TODO(jhuai): check delta is small
    Eigen::Transform<T, 3, Eigen::Isometry> A_T_B;
    A_T_B.setIdentity();
    Eigen::Quaternion<T> q(Eigen::Matrix<T, 3, 3>::Identity() + skew(delta.template tail<3>()));
    A_T_B.linear() = q.normalized().toRotationMatrix();
    A_T_B.translation() = delta.template head<3>();
    return A_T_B;
}

template<typename T>
Eigen::Matrix<T, 6, 1> log(const Eigen::Transform<T, 3, Eigen::Isometry>& A_T_B) {
    Eigen::Matrix<T, 6, 1> delta;
    delta.setZero();
    delta.template head<3>() = A_T_B.translation();
    delta.template tail<3>() = unskew(A_T_B.linear() - Eigen::Matrix<T, 3, 3>::Identity());
    return delta;
}

} // namespace transform

#endif // TRANSFORM_HPP
