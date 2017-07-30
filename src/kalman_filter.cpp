#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::NormalizeAngle(double& phi)
{
  phi = atan2(sin(phi), cos(phi));
}

void KalmanFilter::Predict() {
  x_ = F_ * x_;
	const MatrixXd Ft = F_.transpose();
	P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  const VectorXd z_pred = H_ * x_;
	VectorXd y = z - z_pred;

  // Update Kalman gain
  UpdateCommon(y);
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  VectorXd hx(3);
  const float px = x_(0);
  const float py = x_(1);
  const float vx = x_(2);
  const float vy = x_(3);
  const float phi = atan2(py, px);

  const float rho = sqrt(px * px + py * py) + 0.000001f;

  const float rho_dot = (px * vx + py* vy) / rho;

  hx << rho, phi, rho_dot;

  VectorXd y = z - hx;

  NormalizeAngle(y(1));

  // Update Kalman gain
  UpdateCommon(y);
}

void KalmanFilter::UpdateCommon(const VectorXd& y)
{
  const MatrixXd PHt = P_ * H_.transpose();
  const MatrixXd S = H_ * PHt + R_;
  const MatrixXd K = PHt * S.inverse();

  x_ += K * y;
  P_ -= K * H_ * P_;
}
