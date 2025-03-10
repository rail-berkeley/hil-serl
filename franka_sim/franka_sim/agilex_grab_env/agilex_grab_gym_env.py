from pathlib import Path
from typing import Any, Literal, Tuple, Dict

from trac_ik_python.trac_ik import IK

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
from scipy.spatial.transform import Rotation as R


try:
    import mujoco_py
except ImportError as e:
    MUJOCO_PY_IMPORT_ERROR = e
else:
    MUJOCO_PY_IMPORT_ERROR = None

from franka_sim.controllers import opspace
from franka_sim.mujoco_gym_env import GymRenderingSpec, MujocoGymEnv

_HERE = Path(__file__).parent
_XML_PATH = _HERE / "cobot_magic.xml"
_PANDA_HOME = np.asarray((0, -0.785, 0, -2.35, 0, 1.57))
GRIPPER_OPEN = np.asanyarray((0.02, -0.02))
_CARTESIAN_BOUNDS = np.asarray([[0.2, -0.3, 0], [0.6, 0.3, 0.5]])
_SAMPLING_BOUNDS = np.asarray([[0.25, -0.25], [0.55, 0.25]])


class AgilexGrabGymEnv(MujocoGymEnv):
    metadata = {"render_modes": ["rgb_array", "human"]}  # 

    def __init__(
        self,
        action_scale: np.ndarray = np.asarray([0.1, 1]),
        seed: int = 0,
        control_dt: float = 0.02,
        physics_dt: float = 0.002,
        time_limit: float = 10.0,
        render_spec: GymRenderingSpec = GymRenderingSpec(),
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
        image_obs: bool = False,
    ):
        self._action_scale = action_scale

        super().__init__(
            xml_path=_XML_PATH,
            seed=seed,
            control_dt=control_dt,
            physics_dt=physics_dt,
            time_limit=time_limit,
            render_spec=render_spec,
        )
        print(dir(self._model))
        print("--------joint--------")
        print(self._model.joint)
        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
            ],
            "render_fps": int(np.round(1.0 / self.control_dt)),
        }

        self.render_mode = render_mode
        self.camera_id = (0, 1)
        self.image_obs = image_obs

        # Caching.
        self._left_dof_ids = np.asarray(
            [self._model.joint(f"fl_joint{i}").id for i in range(1, 7)]
        )
        self._left_gripper_dof_ids = np.asarray(
            [self._model.joint(f"fl_joint{i}").id for i in range(7, 9)]
        )
        self._left_ctrl_ids = np.asarray(
            [self._model.actuator(f"fl_joint{i}").id for i in range(1, 7)]
        )
        self._left_gripper_ctrl_id = [
            self._model.actuator(f"fl_joint{i}").id for i in range(7, 9)
        ]
        self._pinch_site_id = 0  # self._model.site("pinch").id
        self._block_z = self._model.geom("block").size[2]

        image_dict = {
            "front": gym.spaces.Box(
                low=0,
                high=255,
                shape=(render_spec.height, render_spec.width, 3),
                dtype=np.uint8,
            ),
            "wrist": gym.spaces.Box(
                low=0,
                high=255,
                shape=(render_spec.height, render_spec.width, 3),
                dtype=np.uint8,
            ),
        }
        state_dict = {
            "pos": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
            "vel": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
            "gripper_pos": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
        }
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(state_dict),
                "images": gym.spaces.Dict(image_dict),
            }
        )

        self.action_space = gym.spaces.Box(
            np.ones((7,), dtype=np.float32) * -1,
            np.ones((7,), dtype=np.float32),
        )


        self.ik_solver = IK("base_link", "link6", urdf_string=open("/Users/anker/Robotics/cobot_magic/Piper_ros/src/piper_description/urdf/piper_description.urdf", 'r').read())

        print("init observation ", self.observation_space)
        # NOTE: gymnasium is used here since MujocoRenderer is not available in gym. It
        # is possible to add a similar viewer feature with gym, but that can be a future TODO
        from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

        self._viewer = MujocoRenderer(
            self.model,
            self.data,
        )
        self._viewer.render(self.render_mode)

    def reset(
        self, seed=None, **kwargs
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the environment."""
        mujoco.mj_resetData(self._model, self._data)

        # Reset arm to home position.
        print(f"left_dof {self._left_dof_ids}")
        self._data.qpos[self._left_dof_ids] = _PANDA_HOME
        self._data.qpos[self._left_gripper_dof_ids] = GRIPPER_OPEN
        mujoco.mj_forward(self._model, self._data)

        # Reset mocap body to home position. useless?
        tcp_pos = self._data.sensor("left_pinch_pos").data
        self._data.mocap_pos[0] = tcp_pos

        # Sample a new block position.
        block_xy = np.random.uniform(*_SAMPLING_BOUNDS)
        self._data.jnt("block").qpos[:3] = (*block_xy, self._block_z)
        mujoco.mj_forward(self._model, self._data)

        # Cache the initial block height.
        self._z_init = self._data.sensor("block_pos").data[2]
        self._z_success = self._z_init + 0.2

        obs = self._compute_observation()
        return obs, {}


    def compute_ik(self, ik_solver, pos, curr_joint):
        # 机器人 URDF 里的链定义
        try:
            # 目标末端位姿
            assert len(pos) == 7
            # 计算逆运动学
            qinit = [0.0] * 6
            joint_angles = ik_solver.get_ik(
                curr_joint, *pos, 0.01, 0.01, 0.01, 999, 999, 999
            )
            if joint_angles is not None:
                return joint_angles
            else:
                print("failed")
        except:
            print("failed")
            return [0.0] * 6

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        take a step in the environment.
        Params:
            action: np.ndarray

        Returns:
            observation: dict[str, np.ndarray],
            reward: float,
            done: bool,
            truncated: bool,
            info: dict[str, Any]
        """
        single_arm = action[0:7]
        x, y, z = single_arm[0:3]
        grasp = single_arm[6]

        # Set the mocap position.
        pos = self._data.mocap_pos[0].copy()
        dpos = np.asarray([x, y, z]) * self._action_scale[0]
        npos = np.clip(pos + dpos, *_CARTESIAN_BOUNDS)
        self._data.mocap_pos[0] = npos

        # Set gripper grasp.
        g = self._data.ctrl[self._left_gripper_ctrl_id[0]] / 255
        dg = grasp * self._action_scale[1]
        ng = np.clip(g + dg, 0.0, 1.0)
        self._data.ctrl[self._left_gripper_ctrl_id[0]] = ng * 255
        quat = R.from_euler('xyz', action[3:7]).as_quat()
        quat_pos = single_arm[0:3] + quat
        joint = self.compute_ik(self.ik_solver, quat_pos, [0.0]*6)
        self._data.ctrl[self._left_ctrl_ids] = joint 
        mujoco.mj_step(self._model, self._data)

        obs = self._compute_observation()
        rew = self._compute_reward()
        terminated = self.time_limit_exceeded()

        return obs, rew, terminated, False, {}

    def render(self):
        rendered_frames = []
        for cam_id in self.camera_id:
            rendered_frames.append(
                self._viewer.render(render_mode="rgb_array", camera_id=cam_id)
            )
        return rendered_frames

    # Helper methods.

    def _compute_observation(self) -> dict:
        obs = {}
        obs["state"] = {}

        left_pos = self._data.sensor("left_pinch_pos").data
        obs["state"]["pos"] = left_pos.astype(np.float32)

        left_vel = self._data.sensor("left_pinch_vel").data
        obs["state"]["vel"] = left_vel.astype(np.float32)

        left_gripper_pos = np.array(
            sum(self._data.ctrl[self._left_gripper_ctrl_id])
            / len(self._left_gripper_ctrl_id)
            / 255,
            dtype=np.float32,
        )
        obs["state"]["gripper_pos"] = left_gripper_pos

        # joint_pos = np.stack(
        #     [self._data.sensor(f"panda/joint{i}_pos").data for i in range(1, 8)],
        # ).ravel()
        # obs["panda/joint_pos"] = joint_pos.astype(np.float32)

        # joint_vel = np.stack(
        #     [self._data.sensor(f"panda/joint{i}_vel").data for i in range(1, 8)],
        # ).ravel()
        # obs["panda/joint_vel"] = joint_vel.astype(np.float32)

        # joint_torque = np.stack(
        # [self._data.sensor(f"panda/joint{i}_torque").data for i in range(1, 8)],
        # ).ravel()
        # obs["panda/joint_torque"] = symlog(joint_torque.astype(np.float32))

        # wrist_force = self._data.sensor("panda/wrist_force").data.astype(np.float32)
        # obs["panda/wrist_force"] = symlog(wrist_force.astype(np.float32))

        if self.image_obs:
            obs["images"] = {}
            images = self.render()
            obs["images"]["front"], obs["images"]["wrist"] = (
                images[0],
                images[1],
            )
        else:
            block_pos = self._data.sensor("block_pos").data.astype(np.float32)
            obs["state"]["block_pos"] = block_pos

        if self.render_mode == "human":
            self._viewer.render(self.render_mode)

        return obs

    def _compute_reward(self) -> float:
        block_pos = self._data.sensor("block_pos").data
        tcp_pos = self._data.sensor("left_pinch_pos").data
        dist = np.linalg.norm(block_pos - tcp_pos)
        r_close = np.exp(-20 * dist)
        r_lift = (block_pos[2] - self._z_init) / (self._z_success - self._z_init)
        r_lift = np.clip(r_lift, 0.0, 1.0)
        rew = 0.3 * r_close + 0.7 * r_lift
        return rew


if __name__ == "__main__":
    env = PandaPickCubeGymEnv(render_mode="human")
    env.reset()
    for i in range(100):
        env.step(np.random.uniform(-1, 1, 4))
        env.render()
    env.close()
