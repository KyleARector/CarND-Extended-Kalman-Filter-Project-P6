#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // Initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);

  // Measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
              0, 0.0225;

  // Measurement noise covariance matrix
  H_laser_ << 1, 0, 0, 0,
              0, 1, 0, 0;

  // Measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
              0, 0.0009, 0,
              0, 0, 0.09;

  VectorXd x_(4);
  x_ << 1, 1, 1, 1;

  // State covariance matrix
  MatrixXd P_ = MatrixXd(4, 4);
  P_ << 1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1000, 0,
        0, 0, 0, 1000;

  // Transition matrix
  MatrixXd F_ = MatrixXd(4, 4);
  F_ << 1, 0, 1, 0,
			  0, 1, 0, 1,
			  0, 0, 1, 0,
			  0, 0, 0, 1;

// Process covariance matrix
  MatrixXd Q_ = MatrixXd(4, 4);

  ekf_.Init(x_, P_, F_, H_laser_, R_laser_, Q_);
}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {


  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    // First measurement
    cout << "EKF: " << endl;
    ekf_.x_ = VectorXd(4);
    ekf_.x_ << 1, 1, 1, 1;

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      // Convert polar coordinates to x and y
      // X
      ekf_.x_[0] = measurement_pack.raw_measurements_[0]
                   * cos(measurement_pack.raw_measurements_[1]);
      // Y
      ekf_.x_[1] = measurement_pack.raw_measurements_[0]
                   * sin(measurement_pack.raw_measurements_[1]);
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      // If type laser, data is already in x and y
      // X
      ekf_.x_[0] = measurement_pack.raw_measurements_[0];
      // Y
      ekf_.x_[1] = measurement_pack.raw_measurements_[1];
    }

    // Set the previous timestamp to the current from the measurement pack
    // Ensures that no extra time is elapsed between initialization and
    // next steps
    previous_timestamp_ = measurement_pack.timestamp_;

    // Done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  // Calculate delta t in seconds
  float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;

  // Update the timestamp
  previous_timestamp_ = measurement_pack.timestamp_;

  // Update the transition matrix with time delta
  ekf_.F_(0,2) = dt;
  ekf_.F_(1,3) = dt;

  // Update process modeling noise
  const float dt_2 = dt* dt;
  const float dt_3 = dt_2 * dt;
  const float dt_4 = dt_3 * dt;

  const float noise_ax = 9;
  const float noise_ay = 9;

  ekf_.Q_ << dt_4 * noise_ax/4, 0, dt_3 * noise_ax/2, 0,
	           0, dt_4 * noise_ay/4, 0, dt_3 * noise_ay/2,
	           dt_3 * noise_ax/2, 0, dt_2 * noise_ax, 0,
             0, dt_3 * noise_ay/2, 0, dt_2 * noise_ay;

  ekf_.Predict();

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
    Hj_ = tools.CalculateJacobian(ekf_.x_);
    ekf_.H_ = Hj_;
    ekf_.R_ = R_radar_;
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  }
  else {
    // Laser updates
    ekf_.H_ = H_laser_;
    ekf_.R_ = R_laser_;
    ekf_.Update(measurement_pack.raw_measurements_);
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
