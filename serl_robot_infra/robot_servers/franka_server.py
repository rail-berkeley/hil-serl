"""
This file starts a control server running on the real time PC connected to the franka robot.
In a screen run `python franka_server.py`
"""
from flask import Flask, request, jsonify
import numpy as np
import rospy
import time
import subprocess
from scipy.spatial.transform import Rotation as R
from absl import app, flags

from franka_msgs.msg import ErrorRecoveryActionGoal, FrankaState
from franka_msgs.srv import SetLoad
from serl_franka_controllers.msg import ZeroJacobian
import geometry_msgs.msg as geom_msg
from dynamic_reconfigure.client import Client as ReconfClient

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "robot_ip", "172.16.0.2", "IP address of the franka robot's controller box"
)
flags.DEFINE_string(
    "gripper_ip", "192.168.1.114", "IP address of the robotiq gripper if being used"
)
flags.DEFINE_string(
    "gripper_type", "Robotiq", "Type of gripper to use: Robotiq, Franka, or None"
)
flags.DEFINE_list(
    "reset_joint_target",
    [0, 0, 0, -1.9, -0, 2, 0],
    "Target joint angles for the robot to reset to",
)
flags.DEFINE_string("flask_url", 
    "127.0.0.1",
    "URL for the flask server to run on."
)
flags.DEFINE_string("ros_port", "11311", "Port for the ROS master to run on.")


class FrankaServer:
    """Handles the starting and stopping of the impedance controller
    (as well as backup) joint recovery policy."""

    def __init__(self, robot_ip, gripper_type, ros_pkg_name, reset_joint_target):
        self.robot_ip = robot_ip
        self.ros_pkg_name = ros_pkg_name
        self.reset_joint_target = reset_joint_target
        self.gripper_type = gripper_type

        self.eepub = rospy.Publisher(
            "/cartesian_impedance_controller/equilibrium_pose",
            geom_msg.PoseStamped,
            queue_size=10,
        )
        self.resetpub = rospy.Publisher(
            "/franka_control/error_recovery/goal", ErrorRecoveryActionGoal, queue_size=1
        )
        self.jacobian_sub = rospy.Subscriber(
            "/cartesian_impedance_controller/franka_jacobian",
            ZeroJacobian,
            self._set_jacobian,
        )
        time.sleep(1)
        self.state_sub = rospy.Subscriber(
            "franka_state_controller/franka_states", FrankaState, self._set_currpos
        )

    def start_impedance(self):
        """Launches the impedance controller"""
        self.imp = subprocess.Popen(
            [
                "roslaunch",
                self.ros_pkg_name,
                "impedance.launch",
                "robot_ip:=" + self.robot_ip,
                f"load_gripper:={'true' if self.gripper_type == 'Franka' else 'false'}",
            ],
            stdout=subprocess.PIPE,
        )
        time.sleep(3)

    def stop_impedance(self):
        """Stops the impedance controller"""
        self.imp.terminate()
        time.sleep(1)

    def clear(self):
        """Clears any errors"""
        msg = ErrorRecoveryActionGoal()
        self.resetpub.publish(msg)

    def reset_joint(self):
        """Resets Joints (needed after running for hours)"""
        # First Stop impedance
        try:
            self.stop_impedance()
            self.clear()
        except:
            print("impedance Not Running")
        time.sleep(3)
        self.clear()

        # Launch joint controller reset
        # set rosparm with rospkg
        # rosparam set /target_joint_positions '[q1, q2, q3, q4, q5, q6, q7]'
        rospy.set_param("/target_joint_positions", self.reset_joint_target)

        self.joint_controller = subprocess.Popen(
            [
                "roslaunch",
                self.ros_pkg_name,
                "joint.launch",
                "robot_ip:=" + self.robot_ip,
                f"load_gripper:={'true' if self.gripper_type == 'Franka' else 'false'}",
            ],
            stdout=subprocess.PIPE,
        )
        time.sleep(1)
        print("RUNNING JOINT RESET")
        self.clear()

        # Wait until target joint angles are reached
        count = 0
        time.sleep(1)
        while not np.allclose(
            np.array(self.reset_joint_target) - np.array(self.q),
            0,
            atol=1e-2,
            rtol=1e-2,
        ):
            time.sleep(1)
            count += 1
            if count > 30:
                print("joint reset TIMEOUT")
                break

        # Stop joint controller
        print("RESET DONE")
        self.joint_controller.terminate()
        time.sleep(1)
        self.clear()
        print("KILLED JOINT RESET", self.pos)

        # Restart impedece controller
        self.start_impedance()
        print("impedance STARTED")

    def move(self, pose: list):
        """Moves to a pose: [x, y, z, qx, qy, qz, qw]"""
        assert len(pose) == 7
        msg = geom_msg.PoseStamped()
        msg.header.frame_id = "0"
        msg.header.stamp = rospy.Time.now()
        msg.pose.position = geom_msg.Point(pose[0], pose[1], pose[2])
        msg.pose.orientation = geom_msg.Quaternion(pose[3], pose[4], pose[5], pose[6])
        self.eepub.publish(msg)

    def _set_currpos(self, msg):
        tmatrix = np.array(list(msg.O_T_EE)).reshape(4, 4).T
        r = R.from_matrix(tmatrix[:3, :3])
        pose = np.concatenate([tmatrix[:3, -1], r.as_quat()])
        self.pos = pose
        self.dq = np.array(list(msg.dq)).reshape((7,))
        self.q = np.array(list(msg.q)).reshape((7,))
        self.force = np.array(list(msg.K_F_ext_hat_K)[:3])
        self.torque = np.array(list(msg.K_F_ext_hat_K)[3:])
        try:
            self.vel = self.jacobian @ self.dq
        except:
            self.vel = np.zeros(6)
            rospy.logwarn("Jacobian not set, end-effector velocity temporarily not available")

    def _set_jacobian(self, msg):
        jacobian = np.array(list(msg.zero_jacobian)).reshape((6, 7), order="F")
        self.jacobian = jacobian


###############################################################################


def main(_):
    ROS_PKG_NAME = "serl_franka_controllers"

    ROBOT_IP = FLAGS.robot_ip
    GRIPPER_IP = FLAGS.gripper_ip
    GRIPPER_TYPE = FLAGS.gripper_type
    RESET_JOINT_TARGET = FLAGS.reset_joint_target

    webapp = Flask(__name__)

    try:
        roscore = subprocess.Popen(f"roscore -p {FLAGS.ros_port}", shell=True)
        time.sleep(1)
    except Exception as e:
        raise Exception("roscore not running", e)

    # Start ros node
    rospy.init_node("franka_control_api")

    if GRIPPER_TYPE == "Robotiq":
        from robot_servers.robotiq_gripper_server import RobotiqGripperServer

        gripper_server = RobotiqGripperServer(gripper_ip=GRIPPER_IP)
    elif GRIPPER_TYPE == "Franka":
        from robot_servers.franka_gripper_server import FrankaGripperServer

        gripper_server = FrankaGripperServer()
    elif GRIPPER_TYPE == "None":
        pass
    else:
        raise NotImplementedError("Gripper Type Not Implemented")

    """Starts impedance controller"""
    robot_server = FrankaServer(
        robot_ip=ROBOT_IP,
        gripper_type=GRIPPER_TYPE,
        ros_pkg_name=ROS_PKG_NAME,
        reset_joint_target=RESET_JOINT_TARGET,
    )
    robot_server.start_impedance()

    reconf_client = ReconfClient(
        "cartesian_impedance_controllerdynamic_reconfigure_compliance_param_node"
    )

    rospy.wait_for_service('/franka_control/set_load')
    set_load_service = rospy.ServiceProxy('/franka_control/set_load', SetLoad)


    # Route for Setting Load
    @webapp.route("/set_load", methods=["POST"])
    def set_load():
        data = request.json
        mass = data['mass']
        F_x_center_load = data['F_x_center_load']
        load_inertia = data['load_inertia']
        set_load_service(mass, F_x_center_load, load_inertia)
        print("Set mass to", mass)
        return "Set Load"

    # Route for Starting impedance
    @webapp.route("/startimp", methods=["POST"])
    def start_impedance():
        robot_server.clear()
        robot_server.start_impedance()
        return "Started impedance"

    # Route for Stopping impedance
    @webapp.route("/stopimp", methods=["POST"])
    def stop_impedance():
        robot_server.stop_impedance()
        return "Stopped impedance"
    
    # Route for pose in euler angles
    @webapp.route("/getpos_euler", methods=["POST"])
    def get_pose_euler():
        xyz = robot_server.pos[:3]
        r = R.from_quat(robot_server.pos[3:]).as_euler("xyz")
        return jsonify({"pose": np.concatenate([xyz, r]).tolist()})

    # Route for Getting Pose
    @webapp.route("/getpos", methods=["POST"])
    def get_pos():
        return jsonify({"pose": np.array(robot_server.pos).tolist()})

    @webapp.route("/getvel", methods=["POST"])
    def get_vel():
        return jsonify({"vel": np.array(robot_server.vel).tolist()})

    @webapp.route("/getforce", methods=["POST"])
    def get_force():
        return jsonify({"force": np.array(robot_server.force).tolist()})

    @webapp.route("/gettorque", methods=["POST"])
    def get_torque():
        return jsonify({"torque": np.array(robot_server.torque).tolist()})

    @webapp.route("/getq", methods=["POST"])
    def get_q():
        return jsonify({"q": np.array(robot_server.q).tolist()})

    @webapp.route("/getdq", methods=["POST"])
    def get_dq():
        return jsonify({"dq": np.array(robot_server.dq).tolist()})

    @webapp.route("/getjacobian", methods=["POST"])
    def get_jacobian():
        return jsonify({"jacobian": np.array(robot_server.jacobian).tolist()})

    # Route for getting gripper distance
    @webapp.route("/get_gripper", methods=["POST"])
    def get_gripper():
        return jsonify({"gripper": gripper_server.gripper_pos})

    # Route for Running Joint Reset
    @webapp.route("/jointreset", methods=["POST"])
    def joint_reset():
        robot_server.clear()
        robot_server.reset_joint()
        return "Reset Joint"

    # Route for Activating the Gripper
    @webapp.route("/activate_gripper", methods=["POST"])
    def activate_gripper():
        print("activate gripper")
        gripper_server.activate_gripper()
        return "Activated"

    # Route for Resetting the Gripper. It will reset and activate the gripper
    @webapp.route("/reset_gripper", methods=["POST"])
    def reset_gripper():
        print("reset gripper")
        gripper_server.reset_gripper()
        return "Reset"

    # Route for Opening the Gripper
    @webapp.route("/open_gripper", methods=["POST"])
    def open():
        print("open")
        gripper_server.open()
        return "Opened"

    # Route for Closing the Gripper
    @webapp.route("/close_gripper", methods=["POST"])
    def close():
        print("close")
        gripper_server.close()
        return "Closed"

    # Route for Closing the Gripper
    @webapp.route("/close_gripper_slow", methods=["POST"])
    def close_slow():
        print("close")
        gripper_server.close_slow()
        return "Closed"

    # Route for moving the gripper
    @webapp.route("/move_gripper", methods=["POST"])
    def move_gripper():
        gripper_pos = request.json
        pos = np.clip(int(gripper_pos["gripper_pos"]), 0, 255)  # 0-255
        print(f"move gripper to {pos}")
        gripper_server.move(pos)
        return "Moved Gripper"

    # Route for Clearing Errors (Communcation constraints, etc.)
    @webapp.route("/clearerr", methods=["POST"])
    def clear():
        robot_server.clear()
        return "Clear"

    # Route for Sending a pose command
    @webapp.route("/pose", methods=["POST"])
    def pose():
        pos = np.array(request.json["arr"])
        # print("Moving to", pos)
        robot_server.move(pos)
        return "Moved"

    # Route for getting all state information
    @webapp.route("/getstate", methods=["POST"])
    def get_state():
        return jsonify(
            {
                "pose": np.array(robot_server.pos).tolist(),
                "vel": np.array(robot_server.vel).tolist(),
                "force": np.array(robot_server.force).tolist(),
                "torque": np.array(robot_server.torque).tolist(),
                "q": np.array(robot_server.q).tolist(),
                "dq": np.array(robot_server.dq).tolist(),
                "jacobian": np.array(robot_server.jacobian).tolist(),
                "gripper_pos": gripper_server.gripper_pos,
            }
        )

    # Route for updating compliance parameters
    @webapp.route("/update_param", methods=["POST"])
    def update_param():
        reconf_client.update_configuration(request.json)
        return "Updated compliance parameters"

    webapp.run(host=FLAGS.flask_url)


if __name__ == "__main__":
    app.run(main)
