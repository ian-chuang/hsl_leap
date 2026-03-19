"""Viser URDF + real LEAP hand teleop with smooth target tracking in radians.

Per control step:
1) Read desired joint targets from GUI sliders (radians).
2) Clamp desired targets to joint limits.
3) EMA smooth desired target using previous applied command.
4) Apply per-step delta clamp using a radians clip magnitude.
5) Send resulting command to robot.

The controller uses only the previous command + current target,
and does not rely on proprioception for command generation.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro
import yourdfpy
import viser
from viser.extras import ViserUrdf

from hsl_leap.leap_hand import LeapHand, LeapHandConfig, MJ_ZERO_POSITION


URDF_PATH = (
	Path(__file__).parent
	/ "assets"
	/ "leap_hand"
	/ "urdf"
	/ "leap_hand_right.urdf"
)

logger = logging.getLogger(__name__)


@dataclass
class Args:
	urdf_path: str = str(URDF_PATH.absolute())
	load_meshes: bool = True
	load_collision_meshes: bool = False
	robot_port: str = "/dev/ttyDXL_leap_hand"
	use_mj_motor_config: bool = True
	read_hz: float = 20.0
	send_hz: float = 20.0
	default_clip_rad: float = 0.05
	torque_off_on_start: bool = False
	default_ema: float = 0.2
	log_level: int = logging.INFO


def _extract_joint_positions_rad(
	obs_deg: dict[str, float],
	joint_names: tuple[str, ...],
	missing_warning_once: list[bool],
) -> np.ndarray | None:
	positions_deg: list[float] = []
	missing: list[str] = []

	for name in joint_names:
		if name in obs_deg:
			positions_deg.append(float(obs_deg[name]))
		elif f"{name}.pos" in obs_deg:
			positions_deg.append(float(obs_deg[f"{name}.pos"]))
		elif name.endswith(".pos") and name[:-4] in obs_deg:
			positions_deg.append(float(obs_deg[name[:-4]]))
		else:
			missing.append(name)

	if missing:
		if not missing_warning_once[0]:
			logger.warning(
				"Missing joint(s) in observation: %s",
				", ".join(missing[:8]),
			)
			missing_warning_once[0] = True
		return None

	return np.deg2rad(np.asarray(positions_deg, dtype=np.float32))


def _safe_disconnect(hand: LeapHand | None) -> None:
	if hand is None:
		return
	try:
		if hand.is_connected:
			hand.disconnect()
	except Exception as exc:
		logger.warning("Disconnect failed: %s", exc)
		try:
			hand.bus.disconnect(False)
		except Exception as exc2:
			logger.warning("Forced bus disconnect failed: %s", exc2)


def _to_deg_action(joint_names: tuple[str, ...], q_target_rad: np.ndarray) -> dict[str, float]:
	return {
		f"{joint}.pos": float(np.rad2deg(q_target_rad[idx]))
		for idx, joint in enumerate(joint_names)
	}


def main(args: Args) -> None:
	logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s: %(message)s")

	server = viser.ViserServer()
	server.scene.add_grid("/grid", width=2.0, height=2.0)

	urdf = yourdfpy.URDF.load(args.urdf_path)
	viser_urdf = ViserUrdf(
		server,
		urdf_or_path=urdf,
		load_meshes=args.load_meshes,
		load_collision_meshes=args.load_collision_meshes,
		collision_mesh_color_override=(1.0, 0.0, 0.0, 0.5),
		root_node_name="/robot",
	)

	joint_names = viser_urdf.get_actuated_joint_names()
	joint_limits_map = viser_urdf.get_actuated_joint_limits()

	low_limits = np.zeros(len(joint_names), dtype=np.float32)
	high_limits = np.zeros(len(joint_names), dtype=np.float32)
	for idx, joint in enumerate(joint_names):
		lower, upper = joint_limits_map[joint]
		lower = -np.pi if lower is None else float(lower)
		upper = np.pi if upper is None else float(upper)
		if lower > upper:
			lower, upper = upper, lower
		low_limits[idx] = lower
		high_limits[idx] = upper
	max_joint_span = float(np.max(high_limits - low_limits)) if len(joint_names) > 0 else float(np.pi)
	clip_init = float(np.clip(args.default_clip_rad, 0.0, max_joint_span))

	with server.gui.add_folder("Robot Control"):
		ema_slider = server.gui.add_slider(
			"EMA alpha",
			min=0.0,
			max=1.0,
			step=0.01,
			initial_value=float(np.clip(args.default_ema, 0.0, 1.0)),
		)
		delta_clip_slider = server.gui.add_slider(
			"Delta clip |rad/step|",
			min=0.0,
			max=max_joint_span,
			step=0.001,
			initial_value=clip_init,
		)
		reset_button = server.gui.add_button("Reset commands to zero")

	slider_handles: dict[str, viser.GuiInputHandle[float]] = {}
	with server.gui.add_folder("Joint Commands [rad]"):
		for idx, joint in enumerate(joint_names):
			slider_handles[joint] = server.gui.add_slider(
				label=joint,
				min=float(low_limits[idx]),
				max=float(high_limits[idx]),
				step=0.001,
				initial_value=0.0,
			)

	with server.gui.add_folder("URDF Visibility"):
		show_meshes_cb = server.gui.add_checkbox("Show meshes", viser_urdf.show_visual)
		show_collision_cb = server.gui.add_checkbox("Show collision", viser_urdf.show_collision)

	@show_meshes_cb.on_update
	def _(_):
		viser_urdf.show_visual = show_meshes_cb.value

	@show_collision_cb.on_update
	def _(_):
		viser_urdf.show_collision = show_collision_cb.value

	show_meshes_cb.visible = args.load_meshes
	show_collision_cb.visible = args.load_collision_meshes

	@reset_button.on_click
	def _(_):
		for joint in slider_handles:
			slider_handles[joint].value = float(np.clip(0.0, slider_handles[joint].min, slider_handles[joint].max))

	hand: LeapHand | None = None
	missing_warning_once = [False]

	try:
		hand = LeapHand(
			LeapHandConfig(
				port=args.robot_port,
				use_mj_motor_config=args.use_mj_motor_config,
			)
		)
		hand.connect()

		logger.info("Moving robot to MJ_ZERO_POSITION with hand.move(duration=3.0)...")
		hand.move(MJ_ZERO_POSITION, duration=3.0)

		if args.torque_off_on_start:
			hand.torque_off()

		obs_deg = hand.get_observation()
		q_obs_rad = _extract_joint_positions_rad(obs_deg, joint_names, missing_warning_once)
		if q_obs_rad is None:
			q_obs_rad = np.zeros(len(joint_names), dtype=np.float32)

		q_obs_rad = np.clip(q_obs_rad, low_limits, high_limits)
		viser_urdf.update_cfg(q_obs_rad)
		for idx, joint in enumerate(joint_names):
			slider_handles[joint].value = float(q_obs_rad[idx])

		logger.info("Running EMA control loop. Open viser UI and move sliders.")

		prev_applied = q_obs_rad.astype(np.float32).copy()
		last_send_time = time.time()
		last_read_time = 0.0

		while True:
			now = time.time()

			send_period = 1.0 / max(args.send_hz, 1e-3)
			if now - last_send_time >= send_period:
				last_send_time = now
				alpha = float(np.clip(ema_slider.value, 0.0, 1.0))

				target_actions = np.asarray(
					[float(slider_handles[joint].value) for joint in joint_names],
					dtype=np.float32,
				)
				target_actions = np.clip(target_actions, low_limits, high_limits)

				ema_actions = alpha * target_actions + (1.0 - alpha) * prev_applied

				max_delta = max(0.0, float(delta_clip_slider.value))
				delta = ema_actions - prev_applied
				next_actions = prev_applied + np.clip(delta, -max_delta, max_delta)
				next_actions = np.clip(next_actions, low_limits, high_limits)

				hand.send_action(_to_deg_action(joint_names, next_actions))
				prev_applied = next_actions

			read_period = 1.0 / max(args.read_hz, 1e-3)
			if now - last_read_time >= read_period:
				last_read_time = now
				obs_deg = hand.get_observation()
				q_rad = _extract_joint_positions_rad(obs_deg, joint_names, missing_warning_once)
				if q_rad is not None:
					viser_urdf.update_cfg(np.clip(q_rad, low_limits, high_limits))

			time.sleep(0.001)

	finally:
		_safe_disconnect(hand)


if __name__ == "__main__":
	main(tyro.cli(Args))
