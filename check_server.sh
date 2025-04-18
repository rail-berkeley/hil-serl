curl -X POST localhost:5000/getpos_euler

curl -X POST localhost:5000/activate_gripper

sleep 1

curl -X POST localhost:5000/close_gripper

sleep 3

curl -X POST localhost:5000/reset_gripper
