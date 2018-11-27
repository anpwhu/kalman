#ifndef KALMAN_EXAMPLES_VIO_LOC_AMCL_POSEMEASUREMENTMODEL_HPP_
#define KALMAN_EXAMPLES_VIO_LOC_AMCL_POSEMEASUREMENTMODEL_HPP_

#include <kalman/transform.hpp>
#include <kalman/LinearizedMeasurementModel.hpp>

namespace vio_loc_amcl
{

/**
 * @brief Measurement vector measuring the robot position
 *
 * @param T Numeric scalar type
 */
template<typename T>
class PoseMeasurement: public Eigen::Matrix<T, 3, 1>
{
public:
    using typename Eigen::Matrix<T, 3, 1>::Scalar;
    static const size_t RowsAtCompileTime = 3u;
    static const size_t ColsAtCompileTime = 1u;
    PoseMeasurement() {}
    PoseMeasurement(const Eigen::Transform<T, 3, Eigen::Isometry>& G_T_I_obs,
                    const Eigen::Transform<T, 3, Eigen::Isometry>& G_T_I_est) {
        Eigen::Matrix<T, 6, 1> z = transform::log(G_T_I_obs * G_T_I_est.inverse());
        (*this)[0] = z[0];
        (*this)[1] = z[1];
        (*this)[2] = z[5];
    }
};

/**
 * @brief Measurement model for measuring the position of the robot
 *        using two beacon-landmarks
 *
 * This is the measurement model for measuring the position of the robot.
 * The measurement is given by two landmarks in the space, whose positions are known.
 * The robot can measure the direct distance to both the landmarks, for instance
 * through visual localization techniques.
 *
 * @param T Numeric scalar type
 * @param CovarianceBase Class template to determine the covariance representation
 *                       (as covariance matrix (StandardBase) or as lower-triangular
 *                       coveriace square root (SquareRootBase))
 */
template<typename T, template<class> class CovarianceBase = Kalman::StandardBase>
class PoseMeasurementModel : public Kalman::LinearizedMeasurementModel<State<T>, PoseMeasurement<T>, CovarianceBase>
{
public:
    //! State type shortcut definition
    typedef  vio_loc_amcl::State<T> S;
    
    //! Measurement type shortcut definition
    typedef  vio_loc_amcl::PoseMeasurement<T> M;
    
    /**
     * @brief Constructor
     *
     * @param landmark1x The x-position of landmark 1
     * @param landmark1y The y-position of landmark 1
     * @param landmark2x The x-position of landmark 2
     * @param landmark2y The y-position of landmark 2
     */
    PoseMeasurementModel()
    {
        // Setup noise jacobian. As this one is static, we can define it once
        // and do not need to update it dynamically
        this->V.setIdentity();
    }


    
    /**
     * @brief Definition of (possibly non-linear) measurement function
     *
     * This function maps the system state to the measurement that is expected
     * to be received from the sensor assuming the system is currently in the
     * estimated state.
     *
     * @param [in] x The system state in current time-step
     * @returns The (predicted) sensor measurement for the system state
     */
    M h(const S& x) const
    {
        M measurement;
        measurement.setZero();
        return measurement;
    }

protected:
    
    /**
     * @brief Update jacobian matrices for the system state transition function using current state
     *
     * This will re-compute the (state-dependent) elements of the jacobian matrices
     * to linearize the non-linear measurement function \f$h(x)\f$ around the
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
    void updateJacobians( const S& x )
    {
        // H = dh/dx (Jacobian of measurement function w.r.t. the state)
        this->H.setZero();

        (this->H)(0, 0) = 1;
        (this->H)(1, 1) = 1;
        (this->H)(2, 5) = 1;
    }
};

} // namespace Robot

#endif
