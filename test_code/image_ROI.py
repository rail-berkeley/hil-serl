import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

def pick_roi_from_topic(topic="/wrist/color/image_raw"):
    bridge = CvBridge()
    img = None

    def cb(msg):
        nonlocal img
        try:
            img = bridge.imgmsg_to_cv2(msg, "bgr8")
        except:
            pass

    sub = rospy.Subscriber(topic, Image, cb)
    # 等一帧
    while img is None and not rospy.is_shutdown():
        rospy.sleep(0.1)
    sub.unregister()

    # 让你框 ROI，返回 (x, y, w, h)
    r = cv2.selectROI("Pick ROI", img, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()
    return r  # (x, y, w, h)

if __name__=="__main__":
    rospy.init_node("roi_picker", anonymous=True)
    x, y, w, h = pick_roi_from_topic()
    print(f"你选的 ROI：x={x}, y={y}, x2={x+w}, y2={y+h}")
