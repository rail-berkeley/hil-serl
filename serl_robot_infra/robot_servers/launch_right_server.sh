# Source the setup.bash file for the second ROS workspace
source /home/undergrad/code/catkin_ws/devel/setup.bash

# Change the ROS master URI to a different port
export ROS_MASTER_URI=http://localhost:11511

# Run the second instance of franka_server.py in the background
python franka_server.py \
    --robot_ip=172.16.0.2 \
    --gripper_type=Robotiq \
    --gripper_ip=192.168.1.114 \
    --reset_joint_target=0,0,0,-1.9,-0,2,0 \
    --flask_url=127.0.0.2 \
    --ros_port=11511