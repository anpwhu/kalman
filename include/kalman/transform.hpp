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

template <typename  T>
static T interpolate(double t0, double t1, double interp_t,T& v0, T& v1){

    if(interp_t < t0 || interp_t > t1)
        throw std::runtime_error("Extrapolation attempted in interpolate");
    if(fabs(interp_t - t0)< 1e-6)
        return v0;
    if(fabs(t1 - interp_t) < 1e-6)
        return v1;

    T interp_v;
    double r = (interp_t - t0)/(t1-t0);
    interp_v = v0 + r*(v1 - v0);

    return interp_v;

}

static Eigen::Matrix3d interpolateSO3 (double t0, double t1, double interp_t, Eigen::Matrix3d& v0, Eigen::Matrix3d& v1){

    if(interp_t < t0 || interp_t > t1)
        throw std::runtime_error("Extrapolation attempted in interpolateSO3");
    if(fabs(interp_t - t0)< 1e-6)
        return v0;
    if(fabs(t1 - interp_t) < 1e-6)
        return v1;

    double r = (interp_t - t0)/(t1-t0);

    Eigen::Quaterniond q0(v0);
    Eigen::Quaterniond q1(v1);
    Eigen::Quaterniond interp_q = q0.slerp(r,q1);
    Eigen::Matrix3d interp_R(interp_q);

    return interp_R;
}

static Eigen::Quaterniond interpolateSO3 (double t0, double t1, double interp_t, Eigen::Quaterniond& q0, Eigen::Quaterniond& q1){
    if(interp_t < t0 || interp_t > t1)
        throw std::runtime_error("Extrapolation attempted in interpolateSO3");
    if(fabs(interp_t - t0)< 1e-6)
        return q0;
    if(fabs(t1 - interp_t) < 1e-6)
        return q1;
    double r = (interp_t - t0)/(t1-t0);
    return q0.slerp(r,q1);
}

} // namespace transform

#endif // TRANSFORM_HPP
