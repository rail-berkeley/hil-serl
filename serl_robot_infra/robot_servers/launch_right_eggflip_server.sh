# Source the setup.bash file for the second ROS workspace
source /home/undergrad/code/catkin_ws/devel/setup.bash

# Change the ROS master URI to a different port
export ROS_MASTER_URI=http://localhost:11511

# Run the second instance of franka_server.py in the background
python franka_eggflip_server.py \
    --robot_ip=172.16.0.2 \
    --gripper_type=None \
    --flask_url=127.0.0.2 \
    --ros_port=11511