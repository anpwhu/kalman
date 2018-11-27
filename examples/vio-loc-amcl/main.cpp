
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
    generator.seed( std::chrono::system_clock::now().time_since_epoch().count() );
    std::normal_distribution<T> noise(0, 1);
    
    // Some filters for estimation
    // Pure predictor without measurement updates
    Kalman::ExtendedKalmanFilter<State> predictor;
    // Extended Kalman Filter
    Kalman::ExtendedKalmanFilter<State> ekf;

    Eigen::Transform<T, 3, Eigen::Isometry> W_T_k;
    T time_k = 0;
    W_T_k.translation() = Eigen::Matrix<T, 3, 1>(0, 0, 0);
    W_T_k.linear() = Eigen::AngleAxis<T>(atan2(cos(0), 1),
                                         Eigen::Matrix<T, 3, 1>::UnitZ()).
        toRotationMatrix();
    State x;
    x.time = time_k;
    x.G_T_I = W_T_k;
    // Init filters with true system state
    predictor.init(x);
    ekf.init(x);
    
    // Standard-Deviation of noise added to all state vector components during state transition
    T systemNoise = 0.001;
    // Standard-Deviation of noise added to all measurement vector components in orientation measurements
    T orientationNoise = 0.00025;
    // Standard-Deviation of noise added to all measurement vector components in distance measurements
    T distanceNoise = 0.00025;
    
    // Simulate for 100 steps
    const size_t N = 100;
    const T dt = 0.1;

    for(size_t i = 1; i <= N; i++)
    {
        // Generate some control input
        Eigen::Transform<T, 3, Eigen::Isometry> W_T_k;
        T time_k = T(2*(i-1)) * T(M_PI) / T(N);
        W_T_k.translation() = Eigen::Matrix<T, 3, 1>(time_k, sin(time_k), 0);
        W_T_k.linear() = Eigen::AngleAxis<T>(atan2(cos(time_k), 1),
                                             Eigen::Matrix<T, 3, 1>::UnitZ()).
            toRotationMatrix();

        Eigen::Transform<T, 3, Eigen::Isometry> W_T_kp1;
        T time = T(2*i) * T(M_PI) / T(N);
        W_T_kp1.translation() = Eigen::Matrix<T, 3, 1>(time, sin(time), 0);
        W_T_kp1.linear() = Eigen::AngleAxis<T>(atan2(cos(time), 1),
                                               Eigen::Matrix<T, 3, 1>::UnitZ()).
            toRotationMatrix();

        u.Ik_T_Ikp1 = W_T_k.inverse() * W_T_kp1;

        // Simulate system
        x = sys.f(x, u);
        sys.setCovariance(dep.getCovariance(u.Ik_T_Ikp1.translation().norm(), Eigen::AngleAxis<T>(u.Ik_T_Ikp1.linear()).angle(), dt));
        // Add noise: Our robot move is affected by noise (due to actuator failures)
        Eigen::Matrix<T, 6, 1> delta;
        delta.setZero();
        delta[0] = systemNoise*noise(generator);
        delta[1] = systemNoise*noise(generator);
        delta[5] = systemNoise*noise(generator);
        x.Oplus(delta);
        
        // Predict state for current time-step using the filters
        auto x_pred = predictor.predict(sys, u);
        auto x_ekf = ekf.predict(sys, u);
        // Pose measurement
        {
            // We can measure the position every 10th step
            PoseMeasurement pose_obs(W_T_kp1, x.G_T_I);
            Eigen::Matrix<T, 3, 3> cov;
            cov.setIdentity();
            cov(0, 0) *= distanceNoise * distanceNoise;
            cov(1, 1) *= distanceNoise * distanceNoise;
            cov(2, 2) *= orientationNoise * orientationNoise;
            pm.setCovariance(cov);
            // Measurement is affected by noise as well
            pose_obs[0] += distanceNoise * noise(generator);
            pose_obs[1] += distanceNoise * noise(generator);
            pose_obs[2] += orientationNoise * noise(generator);
            // Update EKF
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
