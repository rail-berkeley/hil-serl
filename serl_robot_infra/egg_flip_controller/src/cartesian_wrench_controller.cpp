/*
Reference: 
  https://github.com/frankaemika/franka_ros/blob/develop/franka_example_controllers/src/cartesian_wrench_example_controller.cpp
*/

#include <egg_flip_controller/cartesian_wrench_controller.h>

#include <cmath>
#include <memory>

#include <controller_interface/controller_base.h>
#include <franka/robot_state.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>

#include <egg_flip_controller/pseudo_inversion.h>
#include <ros/console.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <std_msgs/Float64MultiArray.h>  // Include header for Float64MultiArray

namespace egg_flip_controller {

bool CartesianWrenchController::init(hardware_interface::RobotHW* robot_hw,
                                               ros::NodeHandle& node_handle) {
  std::vector<double> cartesian_stiffness_vector;
  std::vector<double> cartesian_damping_vector;
  publisher_franka_jacobian_.init(node_handle, "franka_jacobian", 1);

  sub_wrench_target = node_handle.subscribe(
      "wrench_target", 20, &CartesianWrenchController::targetWrenchCallback, this,
      ros::TransportHints().reliable().tcpNoDelay());

  sub_reset = node_handle.subscribe(
      "reset", 20, &CartesianWrenchController::resetCallback, this,
      ros::TransportHints().reliable().tcpNoDelay());

  std::string arm_id;
  if (!node_handle.getParam("arm_id", arm_id)) {
    ROS_ERROR_STREAM("CartesianWrenchController: Could not read parameter arm_id");
    return false;
  }
  std::vector<std::string> joint_names;
  if (!node_handle.getParam("joint_names", joint_names) || joint_names.size() != 7) {
    ROS_ERROR(
        "CartesianWrenchController: Invalid or no joint_names parameters provided, "
        "aborting controller init!");
    return false;
  }

  auto* model_interface = robot_hw->get<franka_hw::FrankaModelInterface>();
  if (model_interface == nullptr) {
    ROS_ERROR_STREAM(
        "CartesianWrenchController: Error getting model interface from hardware");
    return false;
  }
  try {
    model_handle_ = std::make_unique<franka_hw::FrankaModelHandle>(
        model_interface->getHandle(arm_id + "_model"));
  } catch (hardware_interface::HardwareInterfaceException& ex) {
    ROS_ERROR_STREAM(
        "CartesianWrenchController: Exception getting model handle from interface: "
        << ex.what());
    return false;
  }

  auto* state_interface = robot_hw->get<franka_hw::FrankaStateInterface>();
  if (state_interface == nullptr) {
    ROS_ERROR_STREAM(
        "CartesianWrenchController: Error getting state interface from hardware");
    return false;
  }
  try {
    state_handle_ = std::make_unique<franka_hw::FrankaStateHandle>(
        state_interface->getHandle(arm_id + "_robot"));
  } catch (hardware_interface::HardwareInterfaceException& ex) {
    ROS_ERROR_STREAM(
        "CartesianWrenchController: Exception getting state handle from interface: "
        << ex.what());
    return false;
  }

  auto* effort_joint_interface = robot_hw->get<hardware_interface::EffortJointInterface>();
  if (effort_joint_interface == nullptr) {
    ROS_ERROR_STREAM(
        "CartesianWrenchController: Error getting effort joint interface from hardware");
    return false;
  }
  for (size_t i = 0; i < 7; ++i) {
    try {
      joint_handles_.push_back(effort_joint_interface->getHandle(joint_names[i]));
    } catch (const hardware_interface::HardwareInterfaceException& ex) {
      ROS_ERROR_STREAM(
          "CartesianWrenchController: Exception getting joint handles: " << ex.what());
      return false;
    }
  }
  tau_d_publisher_ = node_handle.advertise<std_msgs::Float64MultiArray>("tau_d", 1);

  node_handle.param<double>("max_duration_between_commands", max_duration_between_commands, 0.01);

  // Rate Limiting
  if(!node_handle.getParam("rate_limiting/force", max_force)) {
    ROS_ERROR("CartesianVelocityController: Could not get parameter rate_limiting/linear/velocity");
    return false;
  }
  if(!node_handle.getParam("rate_limiting/torque", max_torque)) {
    ROS_ERROR("CartesianVelocityController: Could not get parameter rate_limiting/acc/acceleration");
    return false;
  }


  if (!node_handle.getParam("k_gains", k_gains_) || k_gains_.size() != 7) {
    ROS_ERROR(
        "JointImpedanceExampleController:  Invalid or no k_gain parameters provided, aborting "
        "controller init!");
    return false;
  }

  if (!node_handle.getParam("d_gains", d_gains_) || d_gains_.size() != 7) {
    ROS_ERROR(
        "JointImpedanceExampleController:  Invalid or no d_gain parameters provided, aborting "
        "controller init!");
    return false;
  }
  std::fill(dq_filtered_.begin(), dq_filtered_.end(), 0);

  std::vector<double> target_positions;
  if (!node_handle.getParam("target_joint_positions", target_positions) || target_positions.size() != 7) {
    ROS_ERROR("JointImpedanceExampleController: Could not read target joint positions from parameter server or incorrect size");
    return false;
  }
  for (size_t i = 0; i < 7; ++i) {
    q_initial_d[i] = target_positions[i];
  }

  // Get bounding box parameters from ROS parameter server
  std::vector<double> min_translation(3), max_translation(3);
  std::vector<double> min_rotation(3), max_rotation(3);

  if (!node_handle.getParam("bounding_box/min_translation", min_translation) || min_translation.size() != 3) {
    ROS_ERROR("CartesianWrenchController: Could not get parameter bounding_box/min_translation");
    return false;
  }
  if (!node_handle.getParam("bounding_box/max_translation", max_translation) || max_translation.size() != 3) {
    ROS_ERROR("CartesianWrenchController: Could not get parameter bounding_box/max_translation");
    return false;
  }
  if (!node_handle.getParam("bounding_box/min_rotation", min_rotation) || min_rotation.size() != 3) {
    ROS_ERROR("CartesianWrenchController: Could not get parameter bounding_box/min_rotation");
    return false;
  }
  if (!node_handle.getParam("bounding_box/max_rotation", max_rotation) || max_rotation.size() != 3) {
    ROS_ERROR("CartesianWrenchController: Could not get parameter bounding_box/max_rotation");
    return false;
  }

  bounding_box_min_translation_ << min_translation[0], min_translation[1], min_translation[2];
  bounding_box_max_translation_ << max_translation[0], max_translation[1], max_translation[2];
  bounding_box_min_rotation_ << min_rotation[0], min_rotation[1], min_rotation[2];
  bounding_box_max_rotation_ << max_rotation[0], max_rotation[1], max_rotation[2];

  // Get PD control gains from ROS parameter server
  if (!node_handle.getParam("p_translation", p_translation_)) {
    ROS_ERROR("CartesianWrenchController: Could not get parameter p_translation");
    return false;
  }
  if (!node_handle.getParam("p_rotation", p_rotation_)) {
    ROS_ERROR("CartesianWrenchController: Could not get parameter p_rotation");
    return false;
  }

  // set wrench target to zero
  wrench_d.setZero();

  return true;
}

void CartesianWrenchController::starting(const ros::Time& /*time*/) {
  // compute initial velocity with jacobian and set x_attractor and q_d_nullspace
  // to initial configuration

  time_since_last_command = ros::Duration(0.0);
  time_since_last_reset_call = ros::Duration(0.0);

  franka::RobotState initial_state = state_handle_->getRobotState();
  // get jacobian
  std::array<double, 42> jacobian_array =
      model_handle_->getZeroJacobian(franka::Frame::kEndEffector);
  // convert to eigen
  Eigen::Affine3d initial_transform(Eigen::Matrix4d::Map(initial_state.O_T_EE.data()));

  // set wrench target to zero
  wrench_d.setZero();

    // Initialize tau_task_target
  tau_task = Eigen::VectorXd::Zero(7);
}

void CartesianWrenchController::update(const ros::Time& time,
                                                 const ros::Duration& period) {
  time_since_last_command += period;
  // get state variables
  franka::RobotState robot_state = state_handle_->getRobotState();
  std::array<double, 7> coriolis_array = model_handle_->getCoriolis();
  std::array<double, 7> gravity = model_handle_->getGravity();

  jacobian_array =
      model_handle_->getZeroJacobian(franka::Frame::kEndEffector);
  publishZeroJacobian(time);

  Eigen::Map<Eigen::Matrix<double, 7, 1>> coriolis(coriolis_array.data());
  Eigen::Map<Eigen::Matrix<double, 6, 7>> jacobian(jacobian_array.data());
  Eigen::Map<Eigen::Matrix<double, 7, 1>> q(robot_state.q.data());
  Eigen::Map<Eigen::Matrix<double, 7, 1>> dq(robot_state.dq.data());
  Eigen::Map<Eigen::Matrix<double, 7, 1>> tau_J_d(  // NOLINT (readability-identifier-naming)
      robot_state.tau_J_d.data());
  Eigen::Affine3d transform(Eigen::Matrix4d::Map(robot_state.O_T_EE.data()));
  Eigen::Vector3d position(transform.translation());


 // Convert Eigen rotation matrix to tf2 quaternion
  Eigen::Quaterniond eigen_quaternion(transform.rotation());
  tf2::Quaternion tf2_quaternion(eigen_quaternion.x(), eigen_quaternion.y(), eigen_quaternion.z(), eigen_quaternion.w());
  // Log the Euler angles
  double roll, pitch, yaw;
  tf2::Matrix3x3(tf2_quaternion).getRPY(roll, pitch, yaw);
  if (roll < 0) {
    roll += 2 * M_PI;
  }
  Eigen::Vector3d euler_angles(roll, pitch, yaw);
  double angle_radians = - M_PI / 4;
  // Create the rotation matrix around the Z-axis
  Eigen::Matrix3d rotation_matrix;
  rotation_matrix = Eigen::AngleAxisd(angle_radians, Eigen::Vector3d::UnitZ());
  // Apply the rotation to the vector
  euler_angles = rotation_matrix * euler_angles;

  // allocate variables
  Eigen::VectorXd tau_task_target(7), tau_nullspace(7), tau_d(7);

  // pseudoinverse for nullspace handling
  // kinematic pseuoinverse
  Eigen::MatrixXd jacobian_transpose_pinv;
  pseudoInverse(jacobian.transpose(), jacobian_transpose_pinv);

  // Check bounding box
  bool outside_bounding_box_translation = (position.array() < bounding_box_min_translation_.array()).any() ||
                                          (position.array() > bounding_box_max_translation_.array()).any();
  bool outside_bounding_box_rotation = (euler_angles.array() < bounding_box_min_rotation_.array()).any() ||
                                       (euler_angles.array() > bounding_box_max_rotation_.array()).any();
  bool outside_bounding_box = outside_bounding_box_translation || outside_bounding_box_rotation;

  // Compute end effector velocity
  Eigen::Matrix<double, 6, 1> end_effector_velocity = jacobian * dq;
  if (!outside_bounding_box) {
    time_since_last_reset_call += period;
  }
  if (!outside_bounding_box && time_since_last_reset_call.toSec() < 1.0) {

    // Do the joint impedance reset
    double alpha = 0.99;
    for (size_t i = 0; i < 7; i++) {
      dq_filtered_[i] = (1 - alpha) * dq_filtered_[i] + alpha * robot_state.dq[i];
    }

    for (size_t i = 0; i < 7; ++i) {
      tau_d[i] = coriolis_array[i] +
                            std::min(std::max(k_gains_[i] * (q_initial_d[i] - robot_state.q[i]), -10.0), 10.0) +
                            std::min(std::max(d_gains_[i] * (-dq_filtered_[i]), -10.0), 10.0);
    }

  } else {
    if (outside_bounding_box) {
    // Clip the position and rotation within the bounding box with margin
    target_position_translation_ = position.cwiseMax(bounding_box_min_translation_ + Eigen::Vector3d(0.02, 0.01, 0.03)).cwiseMin(bounding_box_max_translation_ - Eigen::Vector3d(0.02, 0.01, 0.01));
    target_position_rotation_ = euler_angles.cwiseMax(bounding_box_min_rotation_ + Eigen::Vector3d(0.01, 0.01, 0.01)).cwiseMin(bounding_box_max_rotation_ - Eigen::Vector3d(0.01, 0.01, 0.01));

    // PD control for translation and rotation
    Eigen::Vector3d translation_error = position - target_position_translation_;
    Eigen::Vector3d rotation_error = euler_angles - target_position_rotation_;

    // Combine the translation and rotation errors into a 6D error vector
    Eigen::Matrix<double, 6, 1> error;
    error.head<3>() = translation_error.cwiseMax(-0.03).cwiseMin(0.03);
    error.tail<3>() = rotation_error.cwiseMax(-0.06).cwiseMin(0.06);


    // Apply PD control
    Eigen::Matrix<double, 6, 1> p_control;
    p_control.head<3>() = -p_translation_ * error.head<3>();
    p_control.tail<3>() = -p_rotation_ * error.tail<3>();

    tau_task_target = jacobian.transpose() * (p_control);
    } else {
      target_position_translation_ = (bounding_box_max_translation_ + bounding_box_min_translation_) / 2;
      target_position_rotation_ = (bounding_box_max_rotation_ + bounding_box_min_rotation_) / 2;
      Eigen::Vector3d translation_error = position - target_position_translation_;
      Eigen::Vector3d rotation_error = euler_angles - target_position_rotation_;

      Eigen::Matrix<double, 6, 1> error;
      error.head<3>() = translation_error.cwiseMax(-0.03).cwiseMin(0.03);
      error.tail<3>() = rotation_error.cwiseMax(-0.06).cwiseMin(0.06);

      Eigen::Matrix<double, 6, 1> p_control;
      p_control.head<3>() = -p_translation_ * error.head<3>();
      p_control.tail<3>() = -p_rotation_ * error.tail<3>();
      p_control(2) = 0;
      p_control(4) = 0;

      // Desired force in end-effector frame
      if (time_since_last_command.toSec() > max_duration_between_commands) {
        tau_task_target = jacobian.transpose() * (p_control);
      } else {
        wrench_d(4) = std::min(wrench_d(4), 1.5);
        wrench_d(2) = std::max(wrench_d(2), -1.0);
        tau_task_target = jacobian.transpose() * (wrench_d + p_control);
      }
    }

    // Compute the weighted average
    tau_task = tau_task_target;

  // Nullspace control
    Eigen::Matrix<double, 7, 1> dqe;
    Eigen::Matrix<double, 7, 1> qe;
    qe << q_initial_d - q;
    qe.head(1) << qe.head(1) * joint1_nullspace_stiffness_;
    dqe << dq;
    dqe.head(1) << dqe.head(1) * 2.0 * sqrt(joint1_nullspace_stiffness_);
    tau_nullspace << (Eigen::MatrixXd::Identity(7, 7) -
                      jacobian.transpose() * jacobian_transpose_pinv) *
                        (nullspace_stiffness_ * qe -
                          (2.0 * sqrt(nullspace_stiffness_)) * dqe);

    // Desired torque
    tau_d << tau_task + tau_nullspace + coriolis;
  }

  // Saturate torque rate to avoid discontinuities
  tau_d << saturateTorqueRate(tau_d, tau_J_d);
  
  for (size_t i = 0; i < 7; ++i) {
    joint_handles_[i].setCommand(tau_d(i));
  }
}

void CartesianWrenchController::publishZeroJacobian(const ros::Time& time) {
  if (publisher_franka_jacobian_.trylock()) {
      for (size_t i = 0; i < jacobian_array.size(); i++) {
        publisher_franka_jacobian_.msg_.zero_jacobian[i] = jacobian_array[i];
      }
      publisher_franka_jacobian_.unlockAndPublish();
    }
}

Eigen::Matrix<double, 7, 1> CartesianWrenchController::saturateTorqueRate(
    const Eigen::Matrix<double, 7, 1>& tau_d_calculated,
    const Eigen::Matrix<double, 7, 1>& tau_J_d) {  // NOLINT (readability-identifier-naming)
  Eigen::Matrix<double, 7, 1> tau_d_saturated{};
  for (size_t i = 0; i < 7; i++) {
    double difference = tau_d_calculated[i] - tau_J_d[i];
    tau_d_saturated[i] =
        tau_J_d[i] + std::max(std::min(difference, delta_tau_max_), -delta_tau_max_);
  }
  return tau_d_saturated;
}


void CartesianWrenchController::targetWrenchCallback(
    const geometry_msgs::WrenchStampedConstPtr& msg) {
    wrench_d << msg->wrench.force.x, msg->wrench.force.y, msg->wrench.force.z, msg->wrench.torque.x, msg->wrench.torque.y, msg->wrench.torque.z;

    // clip wrench to max_force and max torque
    wrench_d.head(3) = wrench_d.head(3).cwiseMax(-max_force).cwiseMin(max_force);
    wrench_d.tail(3) = wrench_d.tail(3).cwiseMax(-max_torque).cwiseMin(max_torque);

    time_since_last_command = ros::Duration(0.0);
}

void CartesianWrenchController::resetCallback(const std_msgs::BoolConstPtr& msg) {
  // Reset the controller
  if (msg->data == 1) {
    time_since_last_reset_call = ros::Duration(0.0);
  }
}

}  // namespace egg_flip_controller

PLUGINLIB_EXPORT_CLASS(egg_flip_controller::CartesianWrenchController,
                       controller_interface::ControllerBase)
