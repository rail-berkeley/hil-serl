from easyhid import Enumeration, HIDException
from collections import namedtuple
import timeit
import copy
from typing import Callable, Union, List

# current version number
__version__ = "1.0.3"

# clock for timing
high_acc_clock = timeit.default_timer

# axis mappings are specified as:
# [channel, byte1, byte2, scale]; scale is usually just -1 or 1 and multiplies the result by this value
# (but per-axis scaling can also be achieved by setting this value)
# byte1 and byte2 are indices into the HID array indicating the two bytes to read to form the value for this axis
# For the SpaceNavigator, these are consecutive bytes following the channel number.
AxisSpec = namedtuple("AxisSpec", ["channel", "byte1", "byte2", "scale"])

# button states are specified as:
# [channel, data byte,  bit of byte, index to write to]
# If a message is received on the specified channel, the value of the data byte is set in the button bit array
ButtonSpec = namedtuple("ButtonSpec", ["channel", "byte", "bit"])


## Simple HID code to read data from the 3dconnexion devices

# convert two 8 bit bytes to a signed 16 bit integer
def to_int16(y1, y2):
    x = (y1) | (y2 << 8)
    if x >= 32768:
        x = -(65536 - x)
    return x


# tuple for 6DOF results
SpaceNavigator = namedtuple(
    "SpaceNavigator", ["t", "x", "y", "z", "roll", "pitch", "yaw", "buttons"]
)


class ButtonState(list):
    def __int__(self):
        return sum((b << i) for (i, b) in enumerate(reversed(self)))


class ButtonCallback:
    """Register new button callback"""

    def __init__(
            self, buttons: Union[int, List[int]], callback: Callable[[int, int], None]
    ):
        self.buttons = buttons
        self.callback = callback


class DofCallback:
    """Register new DoF callback"""

    def __init__(
            self,
            axis: str,
            callback: Callable[[int], None],
            sleep: float = 0.0,
            callback_minus: Callable[[int], None] = None,
            filter: float = 0.0
    ):
        self.axis = axis
        self.callback = callback
        self.sleep = sleep
        self.callback_minus = callback_minus
        self.filter = filter


class Config:
    """Create new config file with correct structure and check that the configuration has correct parts"""

    def __init__(
            self,
            callback: Callable[[object], None] = None,
            dof_callback: Callable[[object], None] = None,
            dof_callback_arr: List[DofCallback] = None,
            button_callback: Callable[[object, list], None] = None,
            button_callback_arr: List[ButtonCallback] = None,

    ):
        check_config(callback, dof_callback, dof_callback_arr, button_callback, button_callback_arr)
        self.callback = callback
        self.dof_callback = dof_callback
        self.dof_callback_arr = dof_callback_arr
        self.button_callback = button_callback
        self.button_callback_arr = button_callback_arr


class DeviceSpec(object):
    """Holds the specification of a single 3Dconnexion device"""

    def __init__(
            self, name, hid_id, led_id, mappings, button_mapping, axis_scale=350.0
    ):
        self.name = name
        self.hid_id = hid_id
        self.led_id = led_id
        self.__mappings = mappings
        self.button_mapping = button_mapping
        self.axis_scale = axis_scale
        self.__bytes_to_read = self.__get_num_bytes_to_read()

        # self.led_usage = hid.get_full_usage_id(led_id[0], led_id[1])
        # initialise to a vector of 0s for each state
        self.dict_state = {
            "t": -1,
            "x": 0,
            "y": 0,
            "z": 0,
            "roll": 0,
            "pitch": 0,
            "yaw": 0,
            "buttons": ButtonState([0] * len(self.button_mapping)),
        }
        # initialise to a vector for button_callback_arr timer
        self.dict_state_last = {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": 0.0,
        }
        self.tuple_state = SpaceNavigator(**self.dict_state)

        # start in disconnected state
        self.device = None
        self.callback = None
        self.dof_callback = None
        self.dof_callback_arr = None
        self.button_callback = None
        self.button_callback_arr = None
        self.set_nonblocking_loop = True

    def __get_num_bytes_to_read(self):
        byte_indices = []
        for value in self.__mappings.values():
            byte_indices.extend([value.byte1, value.byte2])

        return max(byte_indices) + 1

    def describe_connection(self):
        """Return string representation of the device, including
        the connection state"""
        if self.device is None:
            return f"{self.name} [disconnected]"
        else:
            return f"{self.name} connected to {self.vendor_name} {self.product_name} version: {self.version_number} [serial: {self.serial_number}]"

    @property
    def mappings(self):
        return self.__mappings

    @mappings.setter
    def mappings(self, val):
        self.__mappings = val
        self.__bytes_to_read = self.__get_num_bytes_to_read()

    @property
    def connected(self):
        """True if the device has been connected"""
        return self.device is not None

    @property
    def state(self):
        """Return the current value of read()

        Returns: state: {t,x,y,z,pitch,yaw,roll,button} namedtuple
                None if the device is not open.
        """
        return self.read()

    def open(self):
        """Open a connection to the device, if possible"""
        if self.device:
            try:
                self.device.open()
            except HIDException as e:
                raise Exception("Failed to open device") from e

        # copy in product details
        self.product_name = self.device.product_string
        self.vendor_name = self.device.manufacturer_string
        self.version_number = self.device.release_number
        # doesn't seem to work on 3dconnexion devices...
        # serial number will be a byte string, we convert to a hex id
        self.serial_number = "".join(
            ["%02X" % ord(char) for char in self.device.serial_number]
        )

    # def set_led(self, state):
    #     """Set the LED state to state (True or False)"""
    #     if self.connected:
    #         reports = self.device.find_output_reports()
    #         for report in reports:
    #             if self.led_usage in report:
    #                 report[self.led_usage] = state
    #                 report.send()

    def close(self):
        """Close the connection, if it is open"""
        if self.connected:
            self.device.close()
            self.device = None

    def read(self):
        """Read data from SpaceMouse and return the current state of this navigation controller.

        Returns:
            state: {t,x,y,z,pitch,yaw,roll,button} namedtuple
            None if the device is not open.
        """
        if not self.connected:
            return None
        # read bytes from SpaceMouse
        ret = self.device.read(self.__bytes_to_read)
        # test for nonblocking read
        if (ret):
            self.process(ret)
        return self.tuple_state

    def process(self, data):
        """
        Update the state based on the incoming data

        This function updates the state of the DeviceSpec object, giving values for each
        axis [x,y,z,roll,pitch,yaw] in range [-1.0, 1.0]
        The state tuple is only set when all 6 DoF have been read correctly.

        The timestamp (in fractional seconds since the start of the program)  is written as element "t"

        If callback is provided, it is called only on DoF state changes with a copy of the current state tuple.
        If dof_callback is provided, it is called only on DOF state changes with the argument (state).
        If dof_callback_arr is provided, it is called only on DOF state changes with the argument (state, axis).
        If button_callback is provided, it is called only on button state changes with the argument (state, buttons).
        If button_callbacks_arr is provided, it is called only on specific button state true with the argument (state, buttons, pressed_buttons).

        Parameters:
            data    The data for this HID event, as returned by the HID callback

        """
        button_changed = False
        dof_changed = False

        for name, (chan, b1, b2, flip) in self.__mappings.items():
            if data[0] == chan:
                dof_changed = True
                #check if b1 or b2 is over the length of the data
                if b1 < len(data) and b2 < len(data):
                    self.dict_state[name] = (
                            flip * to_int16(data[b1], data[b2]) / float(self.axis_scale)
                    )

        for button_index, (chan, byte, bit) in enumerate(self.button_mapping):
            if data[0] == chan:
                button_changed = True
                # update the button vector
                mask = 1 << bit
                self.dict_state["buttons"][button_index] = (
                    1 if (data[byte] & mask) != 0 else 0
                )

        self.dict_state["t"] = high_acc_clock()

        # must receive both parts of the 6DOF state before we return the state dictionary
        if len(self.dict_state) == 8:
            self.tuple_state = SpaceNavigator(**self.dict_state)

        # call any attached callbacks
        if self.callback:
            self.callback(self.tuple_state)

        # only call the DOF callback if the DOF state actually changed
        if self.dof_callback and dof_changed:
            self.dof_callback(self.tuple_state)

        # only call the DoF callback_arr if the specific DoF state actually changed
        if self.dof_callback_arr and dof_changed:
            # foreach all callbacks (ButtonCallback)
            for block_dof_callback in self.dof_callback_arr:
                now = high_acc_clock()
                axis_name = block_dof_callback.axis
                if now >= self.dict_state_last[axis_name] + block_dof_callback.sleep:
                    axis_val = self.dict_state[axis_name]
                    # is minus callback defined
                    if block_dof_callback.callback_minus:
                        # is axis value greater than filter
                        if axis_val > block_dof_callback.filter:
                            block_dof_callback.callback(self.tuple_state, axis_val)
                        elif axis_val < -block_dof_callback.filter:
                            block_dof_callback.callback_minus(self.tuple_state, axis_val)
                    elif axis_val > block_dof_callback.filter or axis_val < -block_dof_callback.filter:
                        block_dof_callback.cafllback(self.tuple_state, axis_val)
                    self.dict_state_last[axis_name] = now

        # only call the button callback if the button state actually changed
        if self.button_callback and button_changed:
            self.button_callback(self.tuple_state, self.tuple_state.buttons)

        # only call the button callback_arr if the specific button state actually changed
        if self.button_callback_arr and button_changed:
            # foreach all callbacks (ButtonCallback)
            for block_button_callback in self.button_callback_arr:
                run = True
                # are buttons list
                if type(block_button_callback.buttons) is list:
                    for button_id in block_button_callback.buttons:
                        if not self.tuple_state.buttons[button_id]:
                            run = False

                # is one button
                elif isinstance(block_button_callback.buttons, int):
                    if not self.tuple_state.buttons[block_button_callback.buttons]:
                        run = False
                # call callback
                if run:
                    block_button_callback.callback(self.tuple_state, self.tuple_state.buttons,
                                                   block_button_callback.buttons)

    def config_set(self, config: Config):
        """Set new configuration of mouse from Config class"""

        self.callback = config.callback
        self.dof_callback = config.dof_callback
        self.dof_callback_arr = config.dof_callback_arr
        self.button_callback = config.button_callback
        self.button_callback_arr = config.button_callback_arr

    def config_set_sep(self, callback=None, dof_callback=None, dof_callback_arr=None, button_callback=None,
                       button_callback_arr=None):
        """Set new configuration of mouse and check that the configuration has correct parts"""

        check_config(callback, dof_callback, dof_callback_arr, button_callback, button_callback_arr)

        self.callback = callback
        self.dof_callback = dof_callback
        self.dof_callback_arr = dof_callback_arr
        self.button_callback = button_callback
        self.button_callback_arr = button_callback_arr

    def config_remove(self):
        """Remove old configuration"""

        self.callback = None
        self.dof_callback = None
        self.dof_callback_arr = None
        self.button_callback = None
        self.button_callback_arr = None


# the IDs for the supported devices
# Each ID maps a device name to a DeviceSpec object
device_specs = {
    "SpaceExplorer": DeviceSpec(
        name="SpaceExplorer",
        # vendor ID and product ID
        hid_id=[0x46D, 0xc627],
        # LED HID usage code pair
        led_id=[0x8, 0x4B],
        mappings={
            "x": AxisSpec(channel=1, byte1=1, byte2=2, scale=1),
            "y": AxisSpec(channel=1, byte1=3, byte2=4, scale=-1),
            "z": AxisSpec(channel=1, byte1=5, byte2=6, scale=-1),
            "pitch": AxisSpec(channel=2, byte1=1, byte2=2, scale=-1),
            "roll": AxisSpec(channel=2, byte1=3, byte2=4, scale=-1),
            "yaw": AxisSpec(channel=2, byte1=5, byte2=6, scale=1),
        },
        button_mapping=[
            ButtonSpec(channel=3, byte=2, bit=0),  # SHIFT
            ButtonSpec(channel=3, byte=1, bit=6),  # ESC
            ButtonSpec(channel=3, byte=2, bit=1),  # CTRL
            ButtonSpec(channel=3, byte=1, bit=7),  # ALT
            ButtonSpec(channel=3, byte=1, bit=0),  # 1
            ButtonSpec(channel=3, byte=1, bit=1),  # 2
            ButtonSpec(channel=3, byte=2, bit=3),  # PANEL
            ButtonSpec(channel=3, byte=2, bit=2),  # FIT
            ButtonSpec(channel=3, byte=2, bit=5),  # -
            ButtonSpec(channel=3, byte=2, bit=4),  # +
            ButtonSpec(channel=3, byte=1, bit=2),  # T
            ButtonSpec(channel=3, byte=1, bit=3),  # L
            ButtonSpec(channel=3, byte=1, bit=5),  # F
            ButtonSpec(channel=3, byte=2, bit=6),  # 2D
            ButtonSpec(channel=3, byte=1, bit=4),  # R
        ],
        axis_scale=350.0,
    ),
    "SpaceNavigator": DeviceSpec(
        name="SpaceNavigator",
        # vendor ID and product ID
        hid_id=[0x46D, 0xC626],
        # LED HID usage code pair
        led_id=[0x8, 0x4B],
        mappings={
            "x": AxisSpec(channel=1, byte1=1, byte2=2, scale=1),
            "y": AxisSpec(channel=1, byte1=3, byte2=4, scale=-1),
            "z": AxisSpec(channel=1, byte1=5, byte2=6, scale=-1),
            "pitch": AxisSpec(channel=2, byte1=1, byte2=2, scale=-1),
            "roll": AxisSpec(channel=2, byte1=3, byte2=4, scale=-1),
            "yaw": AxisSpec(channel=2, byte1=5, byte2=6, scale=1),
        },
        button_mapping=[
            ButtonSpec(channel=3, byte=1, bit=0),  # LEFT
            ButtonSpec(channel=3, byte=1, bit=1),  # RIGHT
        ],
        axis_scale=350.0,
    ),
    "SpaceMouse USB": DeviceSpec(
        name="SpaceMouseUSB",
        # vendor ID and product ID
        hid_id=[0x256f, 0xc641],
        # LED HID usage code pair
        led_id=None,
        mappings={
            "x": AxisSpec(channel=1, byte1=1, byte2=2, scale=1),
            "y": AxisSpec(channel=1, byte1=3, byte2=4, scale=-1),
            "z": AxisSpec(channel=1, byte1=5, byte2=6, scale=-1),
            "pitch": AxisSpec(channel=2, byte1=1, byte2=2, scale=-1),
            "roll": AxisSpec(channel=2, byte1=3, byte2=4, scale=-1),
            "yaw": AxisSpec(channel=2, byte1=5, byte2=6, scale=1),
        },
        button_mapping=[
            ButtonSpec(channel=None, byte=None, bit=None),  # No buttons
            ],
        axis_scale=350.0,
    ),
    "SpaceMouse Compact": DeviceSpec(
        name="SpaceMouse Compact",
        # vendor ID and product ID
        hid_id=[0x256F, 0xC635],
        # LED HID usage code pair
        led_id=[0x8, 0x4B],
        mappings={
            "x": AxisSpec(channel=1, byte1=1, byte2=2, scale=1),
            "y": AxisSpec(channel=1, byte1=3, byte2=4, scale=-1),
            "z": AxisSpec(channel=1, byte1=5, byte2=6, scale=-1),
            "pitch": AxisSpec(channel=2, byte1=1, byte2=2, scale=-1),
            "roll": AxisSpec(channel=2, byte1=3, byte2=4, scale=-1),
            "yaw": AxisSpec(channel=2, byte1=5, byte2=6, scale=1),
        },
        button_mapping=[
            ButtonSpec(channel=3, byte=1, bit=0),  # LEFT
            ButtonSpec(channel=3, byte=1, bit=1),  # RIGHT
        ],
        axis_scale=350.0,
    ),
    "SpaceMouse Pro Wireless": DeviceSpec(
        name="SpaceMouse Pro Wireless",
        # vendor ID and product ID
        hid_id=[0x256F, 0xC632],
        # LED HID usage code pair
        led_id=[0x8, 0x4B],
        mappings={
            "x": AxisSpec(channel=1, byte1=1, byte2=2, scale=1),
            "y": AxisSpec(channel=1, byte1=3, byte2=4, scale=-1),
            "z": AxisSpec(channel=1, byte1=5, byte2=6, scale=-1),
            "pitch": AxisSpec(channel=1, byte1=7, byte2=8, scale=-1),
            "roll": AxisSpec(channel=1, byte1=9, byte2=10, scale=-1),
            "yaw": AxisSpec(channel=1, byte1=11, byte2=12, scale=1),
        },
        button_mapping=[
            ButtonSpec(channel=3, byte=1, bit=0),  # MENU
            ButtonSpec(channel=3, byte=3, bit=7),  # ALT
            ButtonSpec(channel=3, byte=4, bit=1),  # CTRL
            ButtonSpec(channel=3, byte=4, bit=0),  # SHIFT
            ButtonSpec(channel=3, byte=3, bit=6),  # ESC
            ButtonSpec(channel=3, byte=2, bit=4),  # 1
            ButtonSpec(channel=3, byte=2, bit=5),  # 2
            ButtonSpec(channel=3, byte=2, bit=6),  # 3
            ButtonSpec(channel=3, byte=2, bit=7),  # 4
            ButtonSpec(channel=3, byte=2, bit=0),  # ROLL CLOCKWISE
            ButtonSpec(channel=3, byte=1, bit=2),  # TOP
            ButtonSpec(channel=3, byte=4, bit=2),  # ROTATION
            ButtonSpec(channel=3, byte=1, bit=5),  # FRONT
            ButtonSpec(channel=3, byte=1, bit=4),  # REAR
            ButtonSpec(channel=3, byte=1, bit=1),  # FIT
        ],
        axis_scale=350.0,
    ),
    "SpaceMouse Pro": DeviceSpec(
        name="SpaceMouse Pro",
        # vendor ID and product ID
        hid_id=[0x46D, 0xC62b],
        led_id=[0x8, 0x4B],
        mappings={
            "x": AxisSpec(channel=1, byte1=1, byte2=2, scale=1),
            "y": AxisSpec(channel=1, byte1=3, byte2=4, scale=-1),
            "z": AxisSpec(channel=1, byte1=5, byte2=6, scale=-1),
            "pitch": AxisSpec(channel=2, byte1=1, byte2=2, scale=-1),
            "roll": AxisSpec(channel=2, byte1=3, byte2=4, scale=-1),
            "yaw": AxisSpec(channel=2, byte1=5, byte2=6, scale=1),
        },
        button_mapping=[
            ButtonSpec(channel=3, byte=1, bit=0),  # MENU
            ButtonSpec(channel=3, byte=3, bit=7),  # ALT
            ButtonSpec(channel=3, byte=4, bit=1),  # CTRL
            ButtonSpec(channel=3, byte=4, bit=0),  # SHIFT
            ButtonSpec(channel=3, byte=3, bit=6),  # ESC
            ButtonSpec(channel=3, byte=2, bit=4),  # 1
            ButtonSpec(channel=3, byte=2, bit=5),  # 2
            ButtonSpec(channel=3, byte=2, bit=6),  # 3
            ButtonSpec(channel=3, byte=2, bit=7),  # 4
            ButtonSpec(channel=3, byte=2, bit=0),  # ROLL CLOCKWISE
            ButtonSpec(channel=3, byte=1, bit=2),  # TOP
            ButtonSpec(channel=3, byte=4, bit=2),  # ROTATION
            ButtonSpec(channel=3, byte=1, bit=5),  # FRONT
            ButtonSpec(channel=3, byte=1, bit=4),  # REAR
            ButtonSpec(channel=3, byte=1, bit=1),  # FIT
        ],
        axis_scale=350.0,
    ),
    "SpaceMouse Wireless": DeviceSpec(
        name="SpaceMouse Wireless",
        # vendor ID and product ID
        hid_id=[0x256F, 0xC62E],
        # LED HID usage code pair
        led_id=[0x8, 0x4B],
        mappings={
            "x": AxisSpec(channel=1, byte1=1, byte2=2, scale=1),
            "y": AxisSpec(channel=1, byte1=3, byte2=4, scale=-1),
            "z": AxisSpec(channel=1, byte1=5, byte2=6, scale=-1),
            "pitch": AxisSpec(channel=1, byte1=7, byte2=8, scale=-1),
            "roll": AxisSpec(channel=1, byte1=9, byte2=10, scale=-1),
            "yaw": AxisSpec(channel=1, byte1=11, byte2=12, scale=1),
        },
        button_mapping=[
            ButtonSpec(channel=3, byte=1, bit=0),  # LEFT
            ButtonSpec(channel=3, byte=1, bit=1),  # RIGHT
        ],  # FIT
        axis_scale=350.0,
    ),
    "3Dconnexion Universal Receiver": DeviceSpec(
        name="3Dconnexion Universal Receiver",
        # vendor ID and product ID
        hid_id=[0x256F, 0xC652],
        # LED HID usage code pair
        led_id=[0x8, 0x4B],
        mappings={
            "x": AxisSpec(channel=1, byte1=1, byte2=2, scale=1),
            "y": AxisSpec(channel=1, byte1=3, byte2=4, scale=-1),
            "z": AxisSpec(channel=1, byte1=5, byte2=6, scale=-1),
            "pitch": AxisSpec(channel=1, byte1=7, byte2=8, scale=-1),
            "roll": AxisSpec(channel=1, byte1=9, byte2=10, scale=-1),
            "yaw": AxisSpec(channel=1, byte1=11, byte2=12, scale=1),
        },
        button_mapping=[
            ButtonSpec(channel=3, byte=1, bit=0),  # MENU
            ButtonSpec(channel=3, byte=3, bit=7),  # ALT
            ButtonSpec(channel=3, byte=4, bit=1),  # CTRL
            ButtonSpec(channel=3, byte=4, bit=0),  # SHIFT
            ButtonSpec(channel=3, byte=3, bit=6),  # ESC
            ButtonSpec(channel=3, byte=2, bit=4),  # 1
            ButtonSpec(channel=3, byte=2, bit=5),  # 2
            ButtonSpec(channel=3, byte=2, bit=6),  # 3
            ButtonSpec(channel=3, byte=2, bit=7),  # 4
            ButtonSpec(channel=3, byte=2, bit=0),  # ROLL CLOCKWISE
            ButtonSpec(channel=3, byte=1, bit=2),  # TOP
            ButtonSpec(channel=3, byte=4, bit=2),  # ROTATION
            ButtonSpec(channel=3, byte=1, bit=5),  # FRONT
            ButtonSpec(channel=3, byte=1, bit=4),  # REAR
            ButtonSpec(channel=3, byte=1, bit=1),  # FIT
        ],
        axis_scale=350.0,
    ),
    "SpacePilot": DeviceSpec(
        name="SpacePilot",
        # vendor ID and product ID
        hid_id=[0x46D, 0xC625],
        # LED HID usage code pair
        led_id=None,
        mappings={
            "x": AxisSpec(channel=1, byte1=1, byte2=2, scale=1),
            "y": AxisSpec(channel=1, byte1=3, byte2=4, scale=-1),
            "z": AxisSpec(channel=1, byte1=5, byte2=6, scale=-1),
            "pitch": AxisSpec(channel=2, byte1=1, byte2=2, scale=-1),
            "roll": AxisSpec(channel=2, byte1=3, byte2=4, scale=-1),
            "yaw": AxisSpec(channel=2, byte1=5, byte2=6, scale=1),
        },
        button_mapping=[
            ButtonSpec(channel=3, byte=1, bit=0), # 1
            ButtonSpec(channel=3, byte=1, bit=1), # 2
            ButtonSpec(channel=3, byte=1, bit=2), # 3
            ButtonSpec(channel=3, byte=1, bit=3), # 4
            ButtonSpec(channel=3, byte=1, bit=4), # 5
            ButtonSpec(channel=3, byte=1, bit=5), # 6
            ButtonSpec(channel=3, byte=1, bit=6), # T
            ButtonSpec(channel=3, byte=1, bit=7), # L
            ButtonSpec(channel=3, byte=2, bit=0), # R
            ButtonSpec(channel=3, byte=2, bit=1), # F
            ButtonSpec(channel=3, byte=2, bit=2), # Esc
            ButtonSpec(channel=3, byte=2, bit=3), # Alt
            ButtonSpec(channel=3, byte=2, bit=4), # Shift
            ButtonSpec(channel=3, byte=2, bit=5), # Ctrl
            ButtonSpec(channel=3, byte=2, bit=6), # Fit
            ButtonSpec(channel=3, byte=2, bit=7), # Panel
            ButtonSpec(channel=3, byte=3, bit=0), # Zoom -
            ButtonSpec(channel=3, byte=3, bit=1), # Zoom +
            ButtonSpec(channel=3, byte=3, bit=2), # Dom
            ButtonSpec(channel=3, byte=3, bit=3), # 3D Lock
            ButtonSpec(channel=3, byte=3, bit=4), # Config
        ],
        axis_scale=350.0,
    ),
    "SpacePilot Pro": DeviceSpec(
        name="SpacePilot Pro",
        # vendor ID and product ID
        hid_id=[0x46D, 0xC629],
        # LED HID usage code pair
        led_id=[0x8, 0x4B],
        mappings={
            "x": AxisSpec(channel=1, byte1=1, byte2=2, scale=1),
            "y": AxisSpec(channel=1, byte1=3, byte2=4, scale=-1),
            "z": AxisSpec(channel=1, byte1=5, byte2=6, scale=-1),
            "pitch": AxisSpec(channel=2, byte1=1, byte2=2, scale=-1),
            "roll": AxisSpec(channel=2, byte1=3, byte2=4, scale=-1),
            "yaw": AxisSpec(channel=2, byte1=5, byte2=6, scale=1),
        },
        button_mapping=[
            ButtonSpec(channel=3, byte=4, bit=0),  # SHIFT
            ButtonSpec(channel=3, byte=3, bit=6),  # ESC
            ButtonSpec(channel=3, byte=4, bit=1),  # CTRL
            ButtonSpec(channel=3, byte=3, bit=7),  # ALT
            ButtonSpec(channel=3, byte=3, bit=1),  # 1
            ButtonSpec(channel=3, byte=3, bit=2),  # 2
            ButtonSpec(channel=3, byte=2, bit=6),  # 3
            ButtonSpec(channel=3, byte=2, bit=7),  # 4
            ButtonSpec(channel=3, byte=3, bit=0),  # 5
            ButtonSpec(channel=3, byte=1, bit=0),  # MENU
            ButtonSpec(channel=3, byte=4, bit=6),  # -
            ButtonSpec(channel=3, byte=4, bit=5),  # +
            ButtonSpec(channel=3, byte=4, bit=4),  # DOMINANT
            ButtonSpec(channel=3, byte=4, bit=3),  # PAN/ZOOM
            ButtonSpec(channel=3, byte=4, bit=2),  # ROTATION
            ButtonSpec(channel=3, byte=2, bit=0),  # ROLL CLOCKWISE
            ButtonSpec(channel=3, byte=1, bit=2),  # TOP
            ButtonSpec(channel=3, byte=1, bit=5),  # FRONT
            ButtonSpec(channel=3, byte=1, bit=4),  # REAR
            ButtonSpec(channel=3, byte=2, bit=2),  # ISO
            ButtonSpec(channel=3, byte=1, bit=1),  # FIT
        ],
        axis_scale=350.0,
    ),
}

# [For the SpaceNavigator]
# The HID data is in the format
# [id, a, b, c, d, e, f]
# each pair (a,b), (c,d), (e,f) is a 16 bit signed value representing the absolute device state [from -350 to 350]

# if id==1, then the mapping is
# (a,b) = y translation
# (c,d) = x translation
# (e,f) = z translation

# if id==2 then the mapping is
# (a,b) = x tilting (roll)
# (c,d) = y tilting (pitch)
# (d,e) = z tilting (yaw)

# if id==3 then the mapping is
# a = button. Bit 1 = button 1, bit 2 = button 2

# Each movement of the device always causes two HID events, one
# with id 1 and one with id 2, to be generated, one after the other.


supported_devices = list(device_specs.keys())
_active_device = None


def close():
    """Close the active device, if it exists"""
    if _active_device is not None:
        _active_device.close()


def read():
    """Return the current state of the active navigation controller.

    Returns:
        state: {t,x,y,z,pitch,yaw,roll,button} namedtuple
        None if the device is not open.
    """
    return _active_device.read() if _active_device is not None else None

def read_all():
    """Return the current state of the active navigation controller.

    Returns:
        state: {t,x,y,z,pitch,yaw,roll,button} namedtuple
        None if the device is not open.
    """
    return [_active_device[i].read() for i in range(len(_active_device))] if _active_device is not None else None


def list_devices():
    """Return a list of the supported devices connected

    Returns:
        A list of string names of the devices supported which were found. Empty if no supported devices found
    """
    devices = []
    try:
        hid = Enumeration()
    except AttributeError as e:
        raise Exception(
            "HID API is probably not installed. See README.md for details."
        ) from e

    all_hids = hid.find()

    if all_hids:
        for device in all_hids:
            devices.extend(
                device_name
                for device_name, spec in device_specs.items()
                if (
                    device.vendor_id == spec.hid_id[0]
                    and device.product_id == spec.hid_id[1]
                )
            )
    return devices


def openCfg(config: Config, set_nonblocking_loop: bool = True, device=None, DeviceNumber=0):
    """
    Open a 3D space navigator device. Same as open() but input one config file -> class Config

    Returns:
        Device object if the device was opened successfully
        None if the device could not be opened
    """

    return open(config.callback, config.dof_callback, config.dof_callback_arr, config.button_callback,
                config.button_callback_arr, set_nonblocking_loop, device, DeviceNumber)


def open(
        callback: Callable[[object], None] = None,
        dof_callback: Callable[[object], None] = None,
        dof_callback_arr: List[DofCallback] = None,
        button_callback: Callable[[object, list], None] = None,
        button_callback_arr: List[ButtonCallback] = None,
        set_nonblocking_loop=True,
        device: str = None,
        path: str = None,
        DeviceNumber=0) -> Union[None, DeviceSpec]:
    """
    Open a 3D space navigator device. Makes this device the current active device, which enables the module-level read() and close()
    calls. For multiple devices, use the read() and close() calls on the returned object instead, and don't use the module-level calls.

    Parameters:
        callback: If callback is provided, it is called only on DoF state changes with a copy of the current state tuple.
        dof_callback: If dof_callback is provided, it is called only on DOF state changes with the argument (state).
        dof_callback_arr: If dof_callback_arr is provided, it is called only on DOF state changes with the argument (state, axis).
        button_callback: If button_callback is provided, it is called only on button state changes with the argument (state, buttons).
        button_callback_arr: If button_callbacks_arr is provided, it is called only on specific button state true with the argument (state, buttons, pressed_buttons).
        set_nonblocking_loop: Disable waiting for input from SpaceMouse. It is required for using callbacks
        device: name of device to open. Must be one of the values in supported_devices. If None, chooses the first supported device found.
        path: path of the device to open. If path is specified it will try to open at that path regardless of what is connected to it
        DeviceNumber: use the first (DeviceNumber=0) device you find. (for universal wireless receiver)
    Returns:
        Device object if the device was opened successfully
        None if the device could not be opened
    """
    # only used if the module-level functions are used
    global _active_device

    # if no device name specified, look for any matching device and choose the first
    if device is None:
        all_devices = list_devices()
        if len(all_devices) > 0:
            device = all_devices[0]
        else:
            raise Exception("No device connected/supported!")

    found_devices = []
    hid = Enumeration()
    all_hids = hid.find()
    if all_hids:
        for dev in all_hids:
            if path:
                dev.path = path
            spec = device_specs[device]
            if dev.vendor_id == spec.hid_id[0] and dev.product_id == spec.hid_id[1]:
                found_devices.append({"Spec": spec, "HIDDevice": dev})
                print(f"{device} found")

    else:
        print("No HID devices detected")
        return None

    if not found_devices:
        print("No supported devices found")
        return None
    else:
        if len(found_devices) <= DeviceNumber:
            DeviceNumber = 0
        
        if len(found_devices) == 2:
            # Check that the input configuration has the correct components
            # Raise an exception if it encounters incorrect component.
            check_config(callback, dof_callback, dof_callback_arr, button_callback, button_callback_arr)
            # create a copy of the device specification
            spec = found_devices[0]["Spec"]
            dev0 = found_devices[0]["HIDDevice"]
            new_device0 = copy.deepcopy(spec)
            new_device0.device = dev0

            # set the callbacks
            new_device0.callback = callback
            new_device0.dof_callback = dof_callback
            new_device0.dof_callback_arr = dof_callback_arr
            new_device0.button_callback = button_callback
            new_device0.button_callback_arr = button_callback_arr
            # open the device
            new_device0.open()
            # set nonblocking/blocking mode
            new_device0.set_nonblocking_loop = set_nonblocking_loop
            dev0.set_nonblocking(set_nonblocking_loop)


            spec = found_devices[1]["Spec"]
            dev1 = found_devices[1]["HIDDevice"]
            new_device1 = copy.deepcopy(spec)
            new_device1.device = dev1

            # set the callbacks
            new_device1.callback = callback
            new_device1.dof_callback = dof_callback
            new_device1.dof_callback_arr = dof_callback_arr
            new_device1.button_callback = button_callback
            new_device1.button_callback_arr = button_callback_arr
            # open the device
            new_device1.open()
            # set nonblocking/blocking mode
            new_device1.set_nonblocking_loop = set_nonblocking_loop
            dev1.set_nonblocking(set_nonblocking_loop)


            _active_device = [new_device0, new_device1]
            return _active_device

        if len(found_devices) > DeviceNumber:
            # Check that the input configuration has the correct components
            # Raise an exception if it encounters incorrect component.
            check_config(callback, dof_callback, dof_callback_arr, button_callback, button_callback_arr)
            # create a copy of the device specification
            spec = found_devices[DeviceNumber]["Spec"]
            dev = found_devices[DeviceNumber]["HIDDevice"]
            new_device = copy.deepcopy(spec)
            new_device.device = dev

            # set the callbacks
            new_device.callback = callback
            new_device.dof_callback = dof_callback
            new_device.dof_callback_arr = dof_callback_arr
            new_device.button_callback = button_callback
            new_device.button_callback_arr = button_callback_arr
            # open the device
            new_device.open()
            # set nonblocking/blocking mode
            new_device.set_nonblocking_loop = set_nonblocking_loop
            dev.set_nonblocking(set_nonblocking_loop)

            _active_device = [new_device]
            return new_device

    print("Unknown error occured.")
    return None


def check_config(callback=None, dof_callback=None, dof_callback_arr=None, button_callback=None,
                 button_callback_arr=None):
    """Check that the input configuration has the correct components.
    Raise an exception if it encounters incorrect component.
    """
    if dof_callback_arr and check_dof_callback_arr(dof_callback_arr):
        pass
    if button_callback_arr and check_button_callback_arr(button_callback_arr):
        pass


def check_button_callback_arr(button_callback_arr: List[ButtonCallback]) -> List[ButtonCallback]:
    """Check that the button_callback_arr has the correct components.
    Raise an exception if it encounters incorrect component.
    """

    # foreach ButtonCallback
    for num, butt_call in enumerate(button_callback_arr):
        if not isinstance(butt_call, ButtonCallback):
            raise Exception(f"'ButtonCallback[{num}]' is not instance of 'ButtonCallback'")
        if type(butt_call.buttons) is int:
            pass
        elif type(butt_call.buttons) is list:
            for xnum, butt in enumerate(butt_call.buttons):
                if type(butt) is not int:
                    raise Exception(f"'ButtonCallback[{num}]:buttons[{xnum}]' is not type int or list of int")
        else:
            raise Exception(f"'ButtonCallback[{num}]:buttons' is not type int or list of int")
        if not callable(butt_call.callback):
            raise Exception(f"'ButtonCallback[{num}]:callback' is not callable")
    return button_callback_arr

class DofCallback:
    """Register new DoF callback"""

    def __init__(
            self,
            axis: str,
            callback: Callable[[int], None],
            sleep: float = 0.0,
            callback_minus: Callable[[int], None] = None,
            filter: float = 0.0
    ):
        self.axis = axis
        self.callback = callback
        self.sleep = sleep
        self.callback_minus = callback_minus
        self.filter = filter

def check_dof_callback_arr(dof_callback_arr: List[DofCallback]) -> List[DofCallback]:
    """Check that the dof_callback_arr has the correct components.
    Raise an exception if it encounters incorrect component."""

    # foreach DofCallback
    for num, dof_call in enumerate(dof_callback_arr):
        if not isinstance(dof_call, DofCallback):
            raise Exception(f"'DofCallback[{num}]' is not instance of 'DofCallback'")
            # has the correct axis name
        if dof_call.axis not in ["x", "y", "z", "roll", "pitch", "yaw"]:
            raise Exception(
                f"'DofCallback[{num}]:axis' is not string from ['x', 'y', 'z', 'roll', 'pitch', 'yaw']")

            # is callback callable
        if not callable(dof_call.callback):
            raise Exception(f"'DofCallback[{num}]:callback' is not callable")

            # is sleep type float
        if type(dof_call.sleep) is not float:
            raise Exception(f"'DofCallback[{num}]:sleep' is not type float")

            # is callback_minus callable
        if not dof_call.callback_minus or not callable(
            dof_call.callback_minus
        ):
            raise Exception(f"'DofCallback[{num}]:callback_minus' is not callable")

            # is filter type float
        if not dof_call.filter or type(dof_call.filter) is not float:
            raise Exception(f"'DofCallback[{num}]:filter' is not type float")
    return dof_callback_arr


def config_set(config: Config):
    """Set new configuration of mouse from Config class"""

    if _active_device is not None:
        _active_device.config_set(config)


def config_set_sep(callback=None, dof_callback=None, dof_callback_arr=None, button_callback=None,
                   button_callback_arr=None):
    """Set new configuration of mouse and check that the configuration has correct parts"""

    if _active_device is not None:
        _active_device.config_set_sep(callback, dof_callback, dof_callback_arr, button_callback, button_callback_arr)


def config_remove():
    """Remove old configuration"""

    if _active_device is not None:
        _active_device.config_remove()


def print_state(state):
    """Simple default DoF callback
    Print all axis to output
    """
    if state:
        print(
            " ".join(
                [
                    "%4s %+.2f" % (k, getattr(state, k))
                    for k in ["x", "y", "z", "roll", "pitch", "yaw", "t"]
                ]
            )
        )

def silent_callback(state):
    """Silent callback
    Does nothing
    """
    pass

def print_buttons(state, buttons):
    """Simple default button callback
    Print all buttons to output
    """
    # simple default button callback
    print(
        (
            (
                "["
                + " ".join(["%2d, " % buttons[k] for k in range(len(buttons))])
            )
            + "]"
        )
    )


# def toggle_led(state, buttons):
#     print("".join(["buttons=", str(buttons)]))
#     # Switch on the led on left push, off on right push
#     if buttons[0] == 1:
#         set_led(1)
#     if buttons[1] == 1:
#         set_led(0)
#
#
# def set_led(state):
#     if _active_device:
#         _active_device.set_led(state)


if __name__ == "__main__":

    def butt_0(state, buttons, pressed_buttons):
        print("Button 0")


    def butt_2_3(state, buttons, pressed_buttons):
        print("Button 2 and 3")


    button_callbacks_arr = [
        ButtonCallback([0], butt_0),
        ButtonCallback([1], lambda state, buttons, pressed_buttons: print("Button 1")),
        ButtonCallback([2, 3], butt_2_3),
    ]

    dev = open(dof_callback=print_state, button_callback=print_buttons, button_callback_arr=button_callbacks_arr)

    while True:
        state = dev.read()

