#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import time
from functools import cached_property

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from hsl_leap.motors.dynamixel import (
    DynamixelMotorsBus,
    OperatingMode,
)
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from dataclasses import dataclass, field
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

    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a dictionary that maps motor
    # names to the max_relative_target value for that motor.
    # max_relative_target: float | dict[str, float] | None = None

    # cameras
    # cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Set to `True` for backward compatibility with previous policies/dataset
    use_degrees: bool = True

class LeapHand(Robot):
    config_class = LeapHandConfig
    name = "leap_hand"

    def __init__(self, config: LeapHandConfig):
        super().__init__(config)
        self.config = config
        self.bus = DynamixelMotorsBus(
            port=self.config.port,
            motors={
                "motor_13": Motor(13, "xc330-m288", MotorNormMode.DEGREES),
            },
            calibration=self.calibration,
        )

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.bus.motors}

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected

    def connect(self, calibrate: bool = True) -> None:
        self.bus.set_baudrate(self.config.baudrate)
        # self.bus.connect()
        if not self.is_calibrated and calibrate:
            self.calibrate()

        self.configure()

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        self.bus.disable_torque()
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

        input(f"Move {self} to the middle of its range of motion and press ENTER....")
        homing_offsets = self.bus.set_half_turn_homings()

        print(
            "Move all joints sequentially through their entire ranges "
            "of motion.\nRecording positions. Press ENTER to stop..."
        )
        range_mins, range_maxes = self.bus.record_ranges_of_motion()

        self.calibration = {}
        for motor, m in self.bus.motors.items():
            self.calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=homing_offsets[motor],
                range_min=range_mins[motor],
                range_max=range_maxes[motor],
            )

        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        print("Calibration saved to", self.calibration_fpath)

    def configure(self) -> None:
        with self.bus.torque_disabled():
            self.bus.configure_motors()
            for motor in self.bus.motors:
                self.bus.write("Operating_Mode", motor, OperatingMode.CURRENT_POSITION.value)
                # self.bus.write("P_Coefficient", motor, 16)
                # self.bus.write("I_Coefficient", motor, 0)
                # self.bus.write("D_Coefficient", motor, 32)
    
    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise ConnectionError(f"{self} is not connected.")

        # Read arm position
        obs_dict = self.bus.sync_read("Present_Position")
        obs_dict = {f"{motor}.pos": val for motor, val in obs_dict.items()}

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items()}

        # Send goal position to the arm
        self.bus.sync_write("Goal_Position", goal_pos)

        return action

    def disconnect(self):
        self.bus.disconnect(self.config.disable_torque_on_disconnect)


if __name__ == "__main__":

    from pathlib import Path
    
    # allow debug logging
    logging.basicConfig(level=logging.DEBUG)

    hand = LeapHand(
        LeapHandConfig(
            port="/dev/tty.usbserial-FTAO51BR",
            calibration_dir=Path(__file__).parent / "calibration",
            id="leap_hand",
        )
    )
    hand.connect(calibrate=True)


    # bus = DynamixelMotorsBus(
    #     port="/dev/tty.usbserial-FTAO51BR",
    #     motors={
    #         "motor_1": Motor(1, "xc330-m288", MotorNormMode.DEGREES),
    #     },
    # )

    # # bus.scan_port("/dev/tty.usbserial-FTAO51BR")
    # bus.set_baudrate(4_000_000)
    # bus.disconnect()
    # bus.connect()
    # bus._assert_motors_exist()

    # bus.enable_torque()

    # print(bus.sync_read("Present_Position", normalize=True))

    # bus.disconnect()

