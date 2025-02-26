import time

from piper_sdk import *
from piper_sdk import C_PiperInterface
#from piper_msgs.msg import PiperStatusMsg, PosCmd

class PiperControler:
    def __init__(self, can_port = ""):
        self.can_port = can_port
        self.piper = C_PiperInterface(can_name=self.can_port)
        self.piper.ConnectPort()

    def print_arm_end_pose(self):
        end_pose_msg = self.piper.GetArmEndPoseMsgs()
        #end_pose_msg.end_pose.
        print(end_pose_msg)
        end_vel_msg = self.piper.GetCurrentEndVelAndAccParam()
        print(end_vel_msg)

if __name__ == "__main__":
    controler = PiperControler("can_right")
    while True:
        controler.print_arm_end_pose()
        time.sleep(1.0)

 
