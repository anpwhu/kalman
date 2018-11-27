#ifndef KALMAN_EXAMPLES1_ROBOT_SYSTEMMODEL_HPP_
#define KALMAN_EXAMPLES1_ROBOT_SYSTEMMODEL_HPP_

#include <kalman/transform.hpp>
#include <kalman/LinearizedSystemModel.hpp>

namespace vio_loc_amcl
{

/**
 * @brief System state vector-type for a 3DOF planar robot
 *
 * error state: [\nu, \omega], for G_T_I = [R, t; 0, 1], G_T_I = exp([\nu, \omega]) G_\hat{T}_I
 *
 * @param T Numeric scalar type
 */


template<typename T>
class State
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    static const size_t RowsAtCompileTime = 6;
    static const size_t ColsAtCompileTime = 1;
    typedef T Scalar;
    Eigen::Transform<T, 3, Eigen::Isometry> G_T_I;
    T time;
    void Oplus(const Eigen::Matrix<T, RowsAtCompileTime, ColsAtCompileTime>& delta) {
        G_T_I = transform::exp(delta) * G_T_I;
    }
    void setZero() {
        setIdentity();
    }
    State() {
        setIdentity();
    }
    State(const State& other) : G_T_I(other.G_T_I), time(other.time) {}
    State& operator= (const State& other)
    {
        this->G_T_I = other.G_T_I;
        this->time = other.time;
        return *this;
    }

    T x() const {
        return G_T_I.translation()[0];
    }
    T y() const {
        return G_T_I.translation()[1];
    }
    T theta() const {
        return Eigen::AngleAxis<T>(G_T_I.linear()).angle();
    }
    void setIdentity() {
        G_T_I.setIdentity();
        time = T(0);
    }
};

template <typename T>
class DifferentialEncoderParams {
 public:
    const T dist_drift_ratio;
    const T dist_drift_per_sec;
    const T angle_drift_ratio;
    const T angle_drift_per_sec;
    DifferentialEncoderParams():
        dist_drift_ratio(0.015),
        dist_drift_per_sec(0.015),
        angle_drift_ratio(0.03),
        angle_drift_per_sec(0.03) {
    }

    Eigen::Matrix<T, 6, 6> getCovariance(T dist, T angle, T duration) {
      const T distDriftPerSec = dist_drift_per_sec;
      const T distDriftRatioLeft = dist_drift_ratio;
      const T angleDriftPerSec = angle_drift_per_sec;  // rad/sec, injected fake noise
      const T angleDriftRatio = angle_drift_ratio;  // percentage/100, injected fake noise

      const T distDriftPerSec2 = distDriftPerSec * distDriftPerSec;
      const T distDriftRatioLeft2 = distDriftRatioLeft * distDriftRatioLeft;
      const T angleDriftPerSec2 = angleDriftPerSec * angleDriftPerSec;
      const T angleDriftRatio2 = angleDriftRatio * angleDriftRatio;

      T dt2 = duration * duration;
      T dist2 = dist * dist;

      // TODO(jhuai): add cross terms
      // cf. Probabilistic Robotics Sec. 5.4 alpha2 and alpha4
      Eigen::Matrix<T, 6, 6> covariance;
      covariance.setZero();
      covariance.diagonal() << distDriftRatioLeft2 * dist2 + distDriftPerSec2 * dt2,
          distDriftRatioLeft2 * dist2 + distDriftPerSec2 * dt2, 0.0, 0.0, 0.0,
          angleDriftPerSec2 * dt2 + angleDriftRatio2 * angle * angle;
      return covariance;
    }
};

/**
 * @brief System control-input vector-type for a 3DOF planar robot
 *
 * This is the system control-input of a very simple planar robot that
 * can control the velocity in its current direction as well as the
 * change in direction.
 *
 * @param T Numeric scalar type
 */
template<typename T>
class Control
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    static const size_t RowsAtCompileTime = 6u;
    static const size_t ColsAtCompileTime = 1u;
    typedef T Scalar;

    Eigen::Transform<T, 3, Eigen::Isometry> Ik_T_Ikp1;
    T begin_time;
    T end_time;

};

/**
 * @brief System model for a simple planar 3DOF robot
 *
 * This is the system model defining how our robot moves from one 
 * time-step to the next, i.e. how the system state evolves over time.
 *
 * @param T Numeric scalar type
 * @param CovarianceBase Class template to determine the covariance representation
 *                       (as covariance matrix (StandardBase) or as lower-triangular
 *                       coveriace square root (SquareRootBase))
 */
template<typename T, template<class> class CovarianceBase = Kalman::StandardBase>
class SystemModel : public Kalman::LinearizedSystemModel<State<T>, Control<T>, CovarianceBase>
{
public:
    //! State type shortcut definition
    typedef vio_loc_amcl::State<T> S;
    
    //! Control type shortcut definition
    typedef vio_loc_amcl::Control<T> C;
    
    /**
     * @brief Definition of (non-linear) state transition function
     *
     * This function defines how the system state is propagated through time,
     * i.e. it defines in which state \f$\hat{x}_{k+1}\f$ is system is expected to 
     * be in time-step \f$k+1\f$ given the current state \f$x_k\f$ in step \f$k\f$ and
     * the system control input \f$u\f$.
     *
     * @param [in] x The system state in current time-step
     * @param [in] u The control vector input
     * @returns The (predicted) system state in the next time-step
     */
    S f(const S& x, const C& u) const
    {
        //! Predicted state vector after transition
        S x_;
        x_.G_T_I = x.G_T_I*u.Ik_T_Ikp1;
        return x_;
    }

protected:
    /**
     * @brief Update jacobian matrices for the system state transition function using current state
     *
     * This will re-compute the (state-dependent) elements of the jacobian matrices
     * to linearize the non-linear state transition function \f$f(x,u)\f$ around the
     * current state \f$x\f$.
     *
     * @note This is only needed when implementing a LinearizedSystemModel,
     *       for usage with an ExtendedKalmanFilter or SquareRootExtendedKalmanFilter.
     *       When using a fully non-linear filter such as the UnscentedKalmanFilter
     *       or its square-root form then this is not needed.
     *
     * @param x The current system state around which to linearize
     * @param u The current system control input
     */
    void updateJacobians( const S& x, const C& u )
    {
        // F = df/dx (Jacobian of state transition w.r.t. the state)
        this->F.setIdentity();

        // W = df/dw (Jacobian of state transition w.r.t. the noise)
        this->W.setIdentity();
    }
};

} // namespace Robot

#endif
