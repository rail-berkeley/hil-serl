echo "Getting position..."
curl -X POST localhost:5000/getpos_euler

echo ""
echo "Activating gripper..."
curl -X POST localhost:5000/activate_gripper
echo ""
echo ""

echo "Closing gripper in 1s..."
sleep 1

curl -X POST localhost:5000/close_gripper
echo ""
echo ""

echo "Opening gripper in 3s..."
sleep 3

curl -X POST localhost:5000/reset_gripper
echo ""
