# Run this when you're at the panda robot on the vention table at the center of the lab.

# conda init
conda activate hilserl

sudo chmod a+rw /dev/ttyUSB0
source ~/catkin_ws/devel/setup.sh

export ROS_MASTER_URI=http://localhost:3000

python serl_robot_infra/robot_servers/franka_server.py \
    --gripper_type=Robotiq \
    --robot_ip=172.16.0.2 \
    --gripper_ip=localhost \
    --ros_port=3000 \
