// Refered to https://github.com/frankaemika/franka_ros/tree/develop/franka_example_controllers

#pragma once

#include <memory>
#include <string>
#include <vector>
#include <fstream>

#include <controller_interface/multi_interface_controller.h>
#include <dynamic_reconfigure/server.h>
#include <geometry_msgs/WrenchStamped.h>
#include <std_msgs/Bool.h>
#include <hardware_interface/joint_command_interface.h>
#include <hardware_interface/robot_hw.h>
#include <ros/node_handle.h>
#include <ros/time.h>
#include <Eigen/Dense>

#include <franka_hw/franka_model_interface.h>
#include <franka_hw/franka_state_interface.h>
#include <realtime_tools/realtime_publisher.h>
#include <egg_flip_controller/ZeroJacobian.h>

namespace egg_flip_controller {

class CartesianWrenchController : public controller_interface::MultiInterfaceController<
                                                franka_hw::FrankaModelInterface,
                                                hardware_interface::EffortJointInterface,
                                                franka_hw::FrankaStateInterface> {
 public:
  bool init(hardware_interface::RobotHW* robot_hw, ros::NodeHandle& node_handle) override;
  void starting(const ros::Time&) override;
  void update(const ros::Time&, const ros::Duration& period) override;

 private:
  // Saturation
  Eigen::Matrix<double, 7, 1> saturateTorqueRate(
      const Eigen::Matrix<double, 7, 1>& tau_d_calculated,
      const Eigen::Matrix<double, 7, 1>& tau_J_d);  // NOLINT (readability-identifier-naming)

  std::unique_ptr<franka_hw::FrankaStateHandle> state_handle_;
  std::unique_ptr<franka_hw::FrankaModelHandle> model_handle_;
  std::vector<hardware_interface::JointHandle> joint_handles_;
  std::array<double, 42> jacobian_array;
  ros::Publisher tau_d_publisher_;
  Eigen::VectorXd tau_task;
  double alpha;
  double filter_params_{0.005};
  double nullspace_stiffness_{5.0};
  double nullspace_stiffness_target_{5.0};
  double joint1_nullspace_stiffness_{5.0};
  double joint1_nullspace_stiffness_target_{5.0};
  const double delta_tau_max_{1.0};
  Eigen::Matrix<double, 7, 1> q_initial_d;


  // Bounding box parameters
  Eigen::Vector3d bounding_box_min_translation_;
  Eigen::Vector3d bounding_box_max_translation_;
  Eigen::Vector3d bounding_box_min_rotation_;
  Eigen::Vector3d bounding_box_max_rotation_;

    // PD control gains
  double p_translation_;
  double p_rotation_;

  // Target position within the bounding box
  Eigen::Vector3d target_position_translation_;
  Eigen::Vector3d target_position_rotation_;

  ros::Duration time_since_last_command;
  double max_duration_between_commands;
  double max_force;
  double max_torque;


  Eigen::Quaterniond orientation_d_;
  Eigen::Matrix<double, 6, 1> wrench_d;

  Eigen::Quaterniond orientation_d_target_;

  void publishZeroJacobian(const ros::Time& time);
  realtime_tools::RealtimePublisher<egg_flip_controller::ZeroJacobian> publisher_franka_jacobian_;
  void publishDebug(const ros::Time& time);

  // Equilibrium pose subscriber
  ros::Subscriber sub_wrench_target;
  void targetWrenchCallback(const geometry_msgs::WrenchStampedConstPtr& msg);

  ros::Duration time_since_last_reset_call;
  ros::Subscriber sub_reset;
  void resetCallback(const std_msgs::BoolConstPtr& msg);
  std::vector<double> k_gains_;
  std::vector<double> d_gains_;
  std::array<double, 7> dq_filtered_;

};

}  // namespace egg_flip_controller
