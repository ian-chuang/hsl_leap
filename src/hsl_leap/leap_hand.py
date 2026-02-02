import logging
from functools import cached_property

from lerobot.motors import Motor, MotorNormMode
from hsl_leap.motors.dynamixel import (
    DynamixelMotorsBus,
    OperatingMode,
)

from dataclasses import dataclass
from lerobot.robots import RobotConfig, Robot

from typing import Any

logger = logging.getLogger(__name__)

@RobotConfig.register_subclass("leap_hand")
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


class LeapHand(Robot):
    config_class = LeapHandConfig
    name = "leap_hand"

    def __init__(self, config: LeapHandConfig):
        super().__init__(config)
        self.config = config
        self.bus = DynamixelMotorsBus(
            port=self.config.port,
            motors={f"joint_{i}": Motor(i, "xc330-m288", MotorNormMode.RANGE_M100_100) for i in range(16)},
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
                if motor in ["joint_0", "joint_4", "joint_8"]:
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
        obs_dict = self.bus.sync_read("Present_Position", normalize=False) 
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

    def disconnect(self):
        self.bus.disconnect(self.config.disable_torque_on_disconnect)


if __name__ == "__main__":

    from pathlib import Path
    import time
    
    # allow debug logging
    logging.basicConfig(level=logging.DEBUG)

    hand = LeapHand(
        LeapHandConfig(
            port="/dev/tty.usbserial-FTAO51BR",
            calibration_dir=Path(__file__).parent / "calibration",
            id="leap_hand",
        )
    )
    hand.connect()

    print(hand.get_observation())

    

    actions = {
        f"joint_{i}.pos": 0.0 for i in range(16)
    }
    # actions['joint_0.pos'] = -40

    hand.send_action(actions)

    input()

    

    hand.disconnect()
