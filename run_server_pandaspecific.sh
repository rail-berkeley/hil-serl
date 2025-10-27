conda activate serl

source /home/robot/catkin_ws/devel/setup.sh

export ROS_MASTER_URI=http://localhost:3000

python serl_robot_infra/robot_servers/franka_server.py \
    --gripper_type=Franka \
    --robot_ip=172.16.0.2 \
    --ros_port=3000 \
