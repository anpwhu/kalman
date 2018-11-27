// Scenario: The robot is driven to follow a perfect sinusoidal line,
// so the control command is ideal, but the actuator is imperfect.
// The actual robot pose after applying the control deviates from the sinusoidal curve
// The sensor measures the actual robot pose though with some noise
// The ekf tries to estimates the actual robot pose given measured control
// inputs and sensor measurements

// this MUST be first, otherwise there might be problems on windows
// see: https://stackoverflow.com/questions/6563810/m-pi-works-with-math-h-but-not-with-cmath-in-visual-studio/6563891#6563891
#define _USE_MATH_DEFINES
#include <cmath>

#include "SystemModel.hpp"
#include "PoseMeasurementModel.hpp"

#include <kalman/ExtendedKalmanFilter.hpp>

#include <iostream>
#include <random>
#include <chrono>

typedef float T;

// Some type shortcuts
typedef vio_loc_amcl::State<T> State;
typedef vio_loc_amcl::Control<T> Control;
typedef vio_loc_amcl::SystemModel<T> SystemModel;

typedef vio_loc_amcl::PoseMeasurement<T> PoseMeasurement;
typedef vio_loc_amcl::PoseMeasurementModel<T> PoseModel;

template<typename T>
class SinusoidalMotion {
 public:
  SinusoidalMotion(T _dt = T(0.1), size_t _N = 100u) : dt(_dt), N(_N) {}
  // step 0 is the start point
  Eigen::Transform<T, 3, Eigen::Isometry> getStartPose(size_t step = 0u) {
    Eigen::Transform<T, 3, Eigen::Isometry> W_T_k;
    W_T_k.setIdentity();
    W_T_k.translation() = Eigen::Matrix<T, 3, 1>(0, 0, 0);
    W_T_k.linear() = Eigen::AngleAxis<T>(atan2(cos(0), 1),
                                         Eigen::Matrix<T, 3, 1>::UnitZ()).
        toRotationMatrix();
    return W_T_k;
  }

  Eigen::Transform<T, 3, Eigen::Isometry> getDeltaMotion(size_t step) {
    if(step < 1) {
      std::runtime_error("Delta motion only comes with a step greater than 0!");
    }
    Eigen::Transform<T, 3, Eigen::Isometry> W_T_k;
    T time_k = T(2*(step-1)) * T(M_PI) / T(N);
    W_T_k.translation() = Eigen::Matrix<T, 3, 1>(time_k, sin(time_k), 0);
    W_T_k.linear() = Eigen::AngleAxis<T>(atan2(cos(time_k), 1),
                                         Eigen::Matrix<T, 3, 1>::UnitZ()).
        toRotationMatrix();

    Eigen::Transform<T, 3, Eigen::Isometry> W_T_kp1;
    T time = T(2*step) * T(M_PI) / T(N);
    W_T_kp1.translation() = Eigen::Matrix<T, 3, 1>(time, sin(time), 0);
    W_T_kp1.linear() = Eigen::AngleAxis<T>(atan2(cos(time), 1),
                                           Eigen::Matrix<T, 3, 1>::UnitZ()).
        toRotationMatrix();

    return W_T_k.inverse() * W_T_kp1;
  }

  const T dt;
  const size_t N; // total number of steps for a full cycle

};


int main(int argc, char** argv)
{
    // Simulated (true) system state
    
    // Control input
    Control u;
    // System
    SystemModel sys;
    vio_loc_amcl::DifferentialEncoderParams<T> dep;
    // Measurement models
    PoseModel pm;
    
    // Random number generation (for noise simulation)
    std::default_random_engine generator;
    generator.seed(std::chrono::system_clock::now().time_since_epoch().count() );
    std::normal_distribution<T> noise(0, 1);
    
    // Some filters for estimation
    // Pure predictor without measurement updates
    Kalman::ExtendedKalmanFilter<State> predictor;
    // Extended Kalman Filter
    Kalman::ExtendedKalmanFilter<State> ekf;

    // Simulate for 100 steps
    const size_t N = 100;
    const T dt = 0.1;
    SinusoidalMotion<T> sm(dt, N);

    State x;
    x.time = 0;
    x.G_T_I = sm.getStartPose();
    // Init filters with true system state
    predictor.init(x);

    ekf.init(x);
    Eigen::Matrix<T, 6, 6> init_cov;
    init_cov.setIdentity();
    init_cov(2, 2) = T(0);
    init_cov(3, 3) = T(0);
    init_cov(4, 4) = T(0);
    ekf.setCovariance(init_cov);

    // Standard-Deviation of noise added to all state vector components during state transition
    T systemNoise = 0.01;
    // Standard-Deviation of noise added to all measurement vector components in orientation measurements
    T orientationNoise = 0.025;
    // Standard-Deviation of noise added to all measurement vector components in distance measurements
    T distanceNoise = 0.025;
    
    for(size_t i = 1; i <= 2*N; i++)
    {
        // ideal control
        u.Ik_T_Ikp1 = sm.getDeltaMotion(i);
        // Simulate system
        x = sys.f(x, u);
        // Add noise: Our robot move is affected by noise (due to actuator failures)
        Eigen::Matrix<T, 6, 1> delta;
        delta.setZero();
        delta[0] = systemNoise*noise(generator);
        delta[1] = systemNoise*noise(generator);
        delta[5] = systemNoise*noise(generator);
        x.Oplus(delta);

        // Predict state for current time-step using the filters
        auto x_pred = predictor.predict(sys, u);

        Eigen::Matrix<T, 6, 6> control_cov =
            dep.getCovariance(u.Ik_T_Ikp1.translation().norm(),
                              Eigen::AngleAxis<T>(u.Ik_T_Ikp1.linear()).angle(),
                              dt);
        sys.setCovariance(control_cov);
        auto x_ekf = ekf.predict(sys, u);

        // Pose measurement
        {
            // We could have measured the position every 10th step
            PoseMeasurement pose_obs(x.G_T_I, x_ekf.G_T_I);
            // Measurement is affected by noise as well
            pose_obs[0] += distanceNoise * noise(generator);
            pose_obs[1] += distanceNoise * noise(generator);
            pose_obs[2] += orientationNoise * noise(generator);

            Eigen::Matrix<T, 3, 3> cov;
            cov.setIdentity();
            cov(0, 0) *= distanceNoise * distanceNoise;
            cov(1, 1) *= distanceNoise * distanceNoise;
            cov(2, 2) *= orientationNoise * orientationNoise;
            pm.setCovariance(cov);
            x_ekf = ekf.update(pm, pose_obs);
        }
        
        // Print to stdout as csv format
        std::cout   << x.x() << "," << x.y() << "," << x.theta() << ","
                    << x_pred.x() << "," << x_pred.y() << "," << x_pred.theta()  << ","
                    << x_ekf.x() << "," << x_ekf.y() << "," << x_ekf.theta() << ","
                    << "0.0,0.0,0.0"
                    << std::endl;
    }
    return 0;
}
