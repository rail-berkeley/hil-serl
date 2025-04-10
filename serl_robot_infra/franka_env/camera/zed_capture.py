from copy import deepcopy
import cv2
import numpy as np
try:
    import pyzed.sl as sl
except ModuleNotFoundError:
    print("WARNING: You have not setup the ZED cameras, and currently cannot use them")

resize_func_map = {"cv2": cv2.resize, None: None}
standard_params = dict(
    depth_minimum_distance=0.1, camera_resolution=sl.RESOLUTION.HD720, depth_stabilization=False, camera_fps=60, camera_image_flip=sl.FLIP_MODE.OFF
)

advanced_params = dict(
    depth_minimum_distance=0.1, camera_resolution=sl.RESOLUTION.HD2K, depth_stabilization=False, camera_fps=15, camera_image_flip=sl.FLIP_MODE.OFF
)

def print_yellow(x):
    return print("\033[93m {}\033[00m".format(x))

# https://github.com/droid-dataset/droid/blob/main/droid/camera_utils/camera_readers/zed_camera.py
# Referenced from ^, check that if you want to add more functionality.
class ZedCapture:
    def get_device_serial_numbers(self):
        try:
            devices = sl.Camera.get_device_list()
        except NameError:
            return []
        return [str(d.serial_number) for d in devices]

    def __init__(self, name, serial_number, dim=(640, 480), fps=15, depth=False, exposure=40000):
        self.name = name

        assert isinstance(serial_number, str), f"Got {type(serial_number)}."
        assert serial_number in self.get_device_serial_numbers()
        self.serial_number = serial_number

        self._cam = sl.Camera()
        sl_params = dict(
            depth_minimum_distance = 0.1,
            camera_resolution = sl.RESOLUTION.HD720,
            depth_stabilization = False,
            camera_fps = fps,
            camera_image_flip = sl.FLIP_MODE.OFF,
        )
        sl_params = sl.InitParameters(**sl_params)
        sl_params.set_from_serial_number(int(self.serial_number))
        sl_params.camera_image_flip = sl.FLIP_MODE.OFF
        status = self._cam.open(sl_params)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError("Camera Failed To Open")
        self._cam.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, 50)
        
        self._left_img = sl.Mat()
        self.zed_resolution = sl.Resolution(0, 0)

    def read(self):
        err = self._cam.grab(self._runtime)
        if err != sl.ERROR_CODE.SUCCESS:
            print_yellow(f"Warning: No data from camera {self.name}!")
            return False, None
        
        self._cam.retrieve_image(self._left_img, sl.VIEW.LEFT, resolution=self.zed_resolution)
        frame = deepcopy(frame.get_data())

        return True, frame

    def close(self):
        assert hasattr(self, "_cam")
        self._cam.close()
