import logging
from functools import cached_property

from hsl_leap.motors import Motor, MotorNormMode
from hsl_leap.motors.dynamixel import (
    DynamixelMotorsBus,
    OperatingMode,
)

from dataclasses import dataclass
from hsl_leap.robots import RobotConfig, Robot

import time
import numpy as np

from typing import Any

from hsl_leap import CALIBRATION_PATH

logger = logging.getLogger(__name__)

MJ_ZERO_POSITION = {
    "if_mcp.pos": 0.0,
    "if_rot.pos": 0.0,
    "if_pip.pos": 0.0,
    "if_dip.pos": 0.0,
    "mf_mcp.pos": 0.0,
    "mf_rot.pos": 0.0,
    "mf_pip.pos": 0.0,
    "mf_dip.pos": 0.0,
    "rf_mcp.pos": 0.0,
    "rf_rot.pos": 0.0,
    "rf_pip.pos": 0.0,
    "rf_dip.pos": 0.0,
    "th_cmc.pos": 0.0,
    "th_axl.pos": 0.0,
    "th_mcp.pos": 0.0,
    "th_ipl.pos": 0.0,
}

MJ_MOTOR_CONFIG = {
    "if_mcp": Motor(1, "xc330-m288", MotorNormMode.RANGE_M100_100),
    "if_rot": Motor(0, "xc330-m288", MotorNormMode.RANGE_M100_100),
    "if_pip": Motor(2, "xc330-m288", MotorNormMode.RANGE_M100_100),
    "if_dip": Motor(3, "xc330-m288", MotorNormMode.RANGE_M100_100),
    "mf_mcp": Motor(5, "xc330-m288", MotorNormMode.RANGE_M100_100),
    "mf_rot": Motor(4, "xc330-m288", MotorNormMode.RANGE_M100_100),
    "mf_pip": Motor(6, "xc330-m288", MotorNormMode.RANGE_M100_100),
    "mf_dip": Motor(7, "xc330-m288", MotorNormMode.RANGE_M100_100),
    "rf_mcp": Motor(9, "xc330-m288", MotorNormMode.RANGE_M100_100),
    "rf_rot": Motor(8, "xc330-m288", MotorNormMode.RANGE_M100_100),
    "rf_pip": Motor(10, "xc330-m288", MotorNormMode.RANGE_M100_100),
    "rf_dip": Motor(11, "xc330-m288", MotorNormMode.RANGE_M100_100),
    "th_cmc": Motor(12, "xc330-m288", MotorNormMode.RANGE_M100_100),    
    "th_axl": Motor(13, "xc330-m288", MotorNormMode.RANGE_M100_100),
    "th_mcp": Motor(14, "xc330-m288", MotorNormMode.RANGE_M100_100),
    "th_ipl": Motor(15, "xc330-m288", MotorNormMode.RANGE_M100_100),
}

DEFAULT_ZERO_POSITION = {
    f"joint_{i}.pos": 0.0 for i in range(16)
}

DEFAULT_MOTOR_CONFIG = {f"joint_{i}": Motor(i, "xc330-m288", MotorNormMode.RANGE_M100_100) for i in range(16)}

@dataclass
class LeapHandConfig(RobotConfig):
    # Port to connect to the arm
    port: str
    baudrate: int = 4_000_000
    disable_torque_on_disconnect: bool = True

    kP: int = 600
    kI: int = 0
    kD: int = 200
    curr_lim: int = 550  # set this to 550 if you are using full motors!!!!

    use_mj_motor_config: bool = True

    def __post_init__(self):
        self.id = "leap_hand_mj" if self.use_mj_motor_config else "leap_hand"
        print(f"setting calibration dir to {CALIBRATION_PATH}")
        self.calibration_dir = CALIBRATION_PATH


class LeapHand(Robot):
    config_class = LeapHandConfig
    name = "leap_hand"

    def __init__(self, config: LeapHandConfig):
        super().__init__(config)
        self.config = config
        self.bus = DynamixelMotorsBus(
            port=self.config.port,
            motors=MJ_MOTOR_CONFIG if self.config.use_mj_motor_config else DEFAULT_MOTOR_CONFIG,
            calibration=self.calibration,
        )

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.bus.motors}

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return self._motors_ft

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected

    def connect(self) -> None:
        # self.bus.connect() this doesn't work
        self.bus.set_baudrate(self.config.baudrate)
        self.bus._assert_motors_exist()
        if not self.is_calibrated:
            self.calibrate()
        self.configure()

    def calibrate(self) -> None:
        self.bus.write_calibration(self.calibration)

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def configure(self) -> None:
        with self.bus.torque_disabled():
            self.bus.configure_motors()
            for motor in self.bus.motors:
                self.bus.write("Operating_Mode", motor, OperatingMode.CURRENT_POSITION.value)
                self.bus.write("Position_P_Gain", motor, self.config.kP)
                self.bus.write("Position_I_Gain", motor, self.config.kI)
                self.bus.write("Position_D_Gain", motor, self.config.kD)
                if motor in [
                    "joint_0", 
                    "joint_4", 
                    "joint_8", 
                    "if_rot",
                    "mf_rot",
                    "rf_rot",
                ]:
                    self.bus.write("Position_P_Gain", motor, int(self.config.kP * 0.75))  # 75% of kP
                    self.bus.write("Position_D_Gain", motor, int(self.config.kD * 0.75))  # 75% of kD
                self.bus.write("Current_Limit", motor, self.config.curr_lim) 

    def normalize(self, val: int) -> float:
        return val / 4095.0 * 360.0 - 180.0
    
    def denormalize(self, val: float) -> int:
        return int((val + 180.0) / 360.0 * 4095.0)
    
    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise ConnectionError(f"{self} is not connected.")

        # Read arm position
        obs_dict = self.bus.sync_read("Present_Position", normalize=False, num_retry=3)
        obs_dict = {f"{motor}.pos": self.normalize(val) for motor, val in obs_dict.items()}

        return obs_dict
    
    def get_observation_scaled(self) -> dict[str, Any]:
        if not self.is_connected:
            raise ConnectionError(f"{self} is not connected.")
        # Read arm position
        obs_dict = self.bus.sync_read("Present_Position", normalize=True) 
        obs_dict = {f"{motor}.pos": val / 100.0 for motor, val in obs_dict.items()}
        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        goal_pos = {key.removesuffix(".pos"): self.denormalize(val) for key, val in action.items()}

        goal_pos = {
            key: max(
                min(
                    val,
                    self.calibration[key].range_max,
                ),
                self.calibration[key].range_min,
            )
            for key, val in goal_pos.items()
        }

        # Send goal position to the arm
        self.bus.sync_write("Goal_Position", goal_pos, normalize=False)

        return action
    
    def send_action_scaled(self, action: dict[str, Any]) -> dict[str, Any]:
        goal_pos = {key.removesuffix(".pos"): val * 100.0 for key, val in action.items()}
        # Send goal position to the arm
        self.bus.sync_write("Goal_Position", goal_pos, normalize=True)
        return action
    
    def move(self, action: dict[str, Any], duration: float = 1.0, scaled: bool = False) -> None:
        if not self.is_connected:
            raise ConnectionError(f"{self} is not connected.")

        if duration <= 0:
            # just send once
            if scaled:
                self.send_action_scaled(action)
            else:
                self.send_action(action)
            return

        # ---- 1. Get current positions ----
        if scaled:
            current_obs = self.get_observation_scaled()
        else:
            current_obs = self.get_observation()

        # Only interpolate joints present in action
        start = {k: current_obs[k] for k in action.keys()}
        target = action

        # ---- 2. Timing setup ----
        control_rate = 100.0  # Hz (adjust if needed)
        dt = 1.0 / control_rate
        steps = max(1, int(duration * control_rate))

        start_time = time.time()

        # ---- 3. Interpolate ----
        for i in range(steps):
            alpha = (i + 1) / steps  # linear ramp 0 → 1

            interp_action = {
                k: (1 - alpha) * start[k] + alpha * target[k]
                for k in target.keys()
            }

            if scaled:
                self.send_action_scaled(interp_action)
            else:
                self.send_action(interp_action)

            # maintain timing
            next_time = start_time + (i + 1) * dt
            sleep_time = next_time - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)

        # ---- 4. Ensure exact final position ----
        if scaled:
            self.send_action_scaled(target)
        else:
            self.send_action(target)

    def torque_off(self):
        self.bus.disable_torque()

    def torque_on(self):
        self.bus.enable_torque()

    def disconnect(self):
        self.bus.disconnect(self.config.disable_torque_on_disconnect)


if __name__ == "__main__":
    import time
    
    # allow debug logging
    logging.basicConfig(level=logging.DEBUG)

    hand = LeapHand(
        LeapHandConfig(
            port="/dev/ttyDXL_leap_hand",
        )
    )
    hand.connect()

    print(hand.get_observation())

    
    actions = MJ_ZERO_POSITION
    # actions["th_ipl.pos"] = 30.0
    # actions = {
    #     f"joint_{i}.pos": 0.0 for i in range(16)
    # }
    # actions['joint_0.pos'] = -40

    # actions = {
    #     "joint_1.pos": 0.0,
    # }

    # hand.send_action(actions)
    hand.move(actions, duration=2.0)

    input()

    

    hand.disconnect()
