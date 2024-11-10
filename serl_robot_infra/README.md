# SERL Robot Infra
![](../docs/images/robot_infra_interfaces.png)

All robot code is structured as follows:
There is a Flask server which sends commands to the robot via ROS. There is a gym env for the robot which communicates with the Flask server via post requests.

- `robot_server`: hosts a Flask server which sends commands to the robot via ROS
- `franka_env`: gym env for the robot which communicates with the Flask server via post requests


### Installation

1. Install `libfranka` and `franka_ros` with instructions [here](https://frankaemika.github.io/docs/requirements.html).

2. Then install the `serl_franka_controllers` from https://github.com/rail-berkeley/serl_franka_controllers

3. Then, install this package and it's dependencies.
    ```bash
    conda activate hilserl
    pip install -e .
    ```

### Usage
To start using the robot, first power on the robot (small switch on the back of robot control box on the floor). Calibrate the end-effector payload in the browser interface before proceeding to ensure accuarcy of the impedance controller. Then, unlock the robot, enable FCI, and put into execution mode (FR3 only). 

The following command are used to start the impedance controller and robot server that the gym environment communicates with. For bimmanual setup, you can run completely independent servers for each arm even if they have different firmware version (we have a Panda and a FR3) by using different catkin_ws, ROS_MASTER_URI, and flask_url. We have provided examples at [launch_left_server.sh](robot_servers/launch_left_server.sh) and [launch_right_server.sh](robot_servers/launch_right_server.sh)

```bash
cd robot_servers
conda activate hilserl

# source the catkin_ws that contains the serl_franka_controllers package
source </path/to/catkin_ws>/devel/setup.bash

# Set ROS master URI to localhost
export ROS_MASTER_URI=http://localhost:<ros_port_number>

# script to start http server and ros controller
python franka_server.py \
    --gripper_type=<Robotiq|Franka|None> \
    --robot_ip=<robot_IP> \
    --gripper_ip=<[Optional] Robotiq_gripper_IP> \
    --reset_joint_target=<[Optional] robot_joints_when_robot_resets> \
    --flask_url=<url_to_serve> \
    --ros_port=<ros_port_number> \
```

This should start ROS node impedence controller and the HTTP server. You can test that things are running by trying to move the end effector around, if the impedence controller is running it should be compliant.

The HTTP server is used to communicate between the ROS controller and gym environments. Possible HTTP requests include:

| Request | Description |
| --- | --- |
| startimp | Start the impedance controller |
| stopimp | Stop the impedance controller |
| pose | Command robot to go to desired end-effector pose given in base frame (xyz+quaternion) |
| getpos | Return current end-effector pose in robot base frame (xyz+rpy)|
| getvel | Return current end-effector velocity in robot base frame |
| getforce | Return estimated force on end-effector in stiffness frame |
| gettorque | Return estimated torque on end-effector in stiffness frame |
| getq | Return current joint position |
| getdq | Return current joint velocity |
| getjacobian | Return current zero-jacobian |
| getstate | Return all robot states |
| jointreset | Perform joint reset |
| activate_gripper | Activate the gripper (Robotiq only) |
| reset_gripper | Reset the gripper (Robotiq only) |
| get_gripper | Return current gripper position |
| close_gripper | Close the gripper completely |
| open_gripper | Open the gripper completely |
| move_gripper | Move the gripper to a given position |
| clearerr | Clear errors |
| update_param | Update the impedance controller parameters |

These commands can also be called in terminal. Useful ones include:
```bash
curl -X POST <flask_url>:5000/activate_gripper # Activate gripper
curl -X POST <flask_url>:5000/close_gripper # Close gripper
curl -X POST <flask_url>:5000/open_gripper # Open gripper
curl -X POST <flask_url>:5000/getpos # Print current end-effector pose in xyz translation and xyzw quaternions
curl -X POST <flask_url>:5000/getpos_euler # Get current end-effector pose in xyz translation and xyz euler angles
curl -X POST <flask_url>:5000/jointreset # Perform joint reset
curl -X POST <flask_url>:5000/stopimp # Stop the impedance controller
curl -X POST <flask_url>:5000/startimp # Start the impedance controller (**Only run this after stopimp**)
```