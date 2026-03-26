from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hsl_leap.leap_hand import LeapHand, LeapHandConfig, MJ_ZERO_POSITION


logger = logging.getLogger(__name__)

# XC330-M288 conversions from e-Manual control-table units.
DXL_TICKS_PER_REV = 4096.0
RAD_PER_TICK = 2.0 * math.pi / DXL_TICKS_PER_REV
RPM_PER_VELOCITY_LSB = 0.229
RAD_S_PER_VELOCITY_LSB = RPM_PER_VELOCITY_LSB * 2.0 * math.pi / 60.0
MA_PER_CURRENT_LSB = 1.0

# Datasheet-derived proportional conversion (approximate for XC330 family).
# Note: XC330 Present Current is input current, not direct motor phase current.
TORQUE_NM_PER_AMP = 0.515


@dataclass
class WaveSegment:
	name: str
	kind: str
	duration_s: float
	amplitude: float
	frequency_hz: float
	offset: float = 0.0
	phase_rad: float = 0.0
	duty_cycle: float = 0.5


@dataclass
class Args:
	port: str = "/dev/ttyDXL_leap_hand"
	baudrate: int = 4_000_000
	motor: str = "if_mcp"
	use_mj_motor_config: bool = True
	control_hz: float = 20.0
	out_dir: str = "sysid/outputs/traj"
	run_name: str = ""
	log_level: str = "INFO"


def _ticks_to_rad(ticks: float) -> float:
	return (ticks * RAD_PER_TICK) - math.pi


def _velocity_lsb_to_rad_s(vel_lsb: float) -> float:
	return vel_lsb * RAD_S_PER_VELOCITY_LSB


def _current_lsb_to_milliamp(cur_lsb: float) -> float:
	return cur_lsb * MA_PER_CURRENT_LSB


def _current_milliamp_to_torque_nm(current_mA: float) -> float:
	return (current_mA / 1000.0) * TORQUE_NM_PER_AMP


def _default_wave_schedule() -> list[WaveSegment]:
	return [
		WaveSegment(
			name="sine_low_freq",
			kind="sine",
			duration_s=4.0,
			amplitude=1.0,
			frequency_hz=0.35,
			offset=0.0,
			phase_rad=0.0,
		),
		WaveSegment(
			name="sine_medium_freq",
			kind="sine",
			duration_s=4.0,
			amplitude=1.0,
			frequency_hz=0.7,
			offset=0.0,
			phase_rad=0.0,
		),
		WaveSegment(
			name="sine_high_freq",
			kind="sine",
			duration_s=4.0,
			amplitude=1.0,
			frequency_hz=1.1,
			offset=0.0,
			phase_rad=0.0,
		),
		WaveSegment(
			name="square_final",
			kind="square",
			duration_s=6.0,
			amplitude=1.0,
			frequency_hz=0.8,
			offset=0.0,
			phase_rad=0.0,
			duty_cycle=0.5,
		),
	]


def _eval_wave_scaled(segment: WaveSegment, t_seg: float) -> float:
	omega = 2.0 * math.pi * segment.frequency_hz
	phase = omega * t_seg + segment.phase_rad

	if segment.kind == "sine":
		val = segment.offset + segment.amplitude * math.sin(phase)
	elif segment.kind == "square":
		frac = ((phase / (2.0 * math.pi)) % 1.0 + 1.0) % 1.0
		high = frac < segment.duty_cycle
		val = segment.offset + (segment.amplitude if high else -segment.amplitude)
	else:
		raise ValueError(f"Unsupported segment kind: {segment.kind}")

	return float(np.clip(val, -1.0, 1.0))


def _read_joint_telemetry(hand: LeapHand, motor: str) -> tuple[float, float, float, float]:
	position_tick = float(
		hand.bus.sync_read(
			"Present_Position",
			motors=[motor],
			normalize=False,
			num_retry=hand.config.read_num_retries,
		)[motor]
	)
	velocity_lsb = float(
		hand.bus.sync_read(
			"Present_Velocity",
			motors=[motor],
			normalize=False,
			num_retry=hand.config.read_num_retries,
		)[motor]
	)
	current_lsb = float(
		hand.bus.sync_read(
			"Present_Current",
			motors=[motor],
			normalize=False,
			num_retry=hand.config.read_num_retries,
		)[motor]
	)
	# Read motor's internal trajectory target position if available
	position_traj_tick = float(
		hand.bus.sync_read(
			"Position_Trajectory",
			motors=[motor],
			normalize=False,
			num_retry=hand.config.read_num_retries,
		)[motor]
	)
	return position_tick, velocity_lsb, current_lsb, position_traj_tick


def _save_csv(path: Path, rows: list[dict[str, float | str]], fieldnames: list[str]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)



def _save_plot(rows: list[dict[str, float | str]], out_path: Path, motor: str) -> None:
	t = np.asarray([float(r["time_s"]) for r in rows], dtype=np.float64)
	cmd_scaled = np.asarray([float(r["command_scaled"]) for r in rows], dtype=np.float64)
	pos_rad = np.asarray([float(r["position_rad"]) for r in rows], dtype=np.float64)
	pos_traj_rad = np.asarray([float(r.get("position_trajectory_rad", float('nan'))) for r in rows], dtype=np.float64)
	vel_rad_s = np.asarray([float(r["velocity_rad_s"]) for r in rows], dtype=np.float64)
	cur_mA = np.asarray([float(r["current_mA"]) for r in rows], dtype=np.float64)
	torque_nm = np.asarray([float(r["torque_nm_from_current"]) for r in rows], dtype=np.float64)

	fig, axes = plt.subplots(5, 1, figsize=(12, 14), sharex=True)

	axes[0].plot(t, cmd_scaled, label="command_scaled [-1,1]", linewidth=1.2)
	axes[0].set_ylabel("command [-1,1]")
	axes[0].legend(loc="upper right", fontsize="small")
	axes[0].grid(True)

	axes[1].plot(t, pos_rad, label="measured position_rad [rad]", linewidth=1.0)
	# Plot internal trajectory target if present
	if not np.all(np.isnan(pos_traj_rad)):
		axes[1].plot(t, pos_traj_rad, label="position_trajectory_rad [rad] (target)", linestyle="--", linewidth=1.0)
	axes[1].set_ylabel("rad")
	axes[1].legend(loc="upper right", fontsize="small")
	axes[1].grid(True)

	axes[2].plot(t, vel_rad_s, color="tab:orange", label="velocity_rad_s")
	axes[2].set_ylabel("rad/s")
	axes[2].legend(loc="upper right", fontsize="small")
	axes[2].grid(True)

	axes[3].plot(t, cur_mA, color="tab:green", label="current_mA")
	axes[3].set_ylabel("mA")
	axes[3].legend(loc="upper right", fontsize="small")
	axes[3].grid(True)

	axes[4].plot(
		t,
		torque_nm,
		color="tab:red",
		label=f"torque_nm = current_A * {TORQUE_NM_PER_AMP:.3f}",
	)
	axes[4].set_xlabel("time [s]")
	axes[4].set_ylabel("Nm")
	axes[4].legend(loc="upper right", fontsize="small")
	axes[4].grid(True)

	fig.suptitle(
		(
			f"Single-joint trajectory capture ({motor})\n"
			"Units: rad, rad/s, mA, Nm. "
			f"Torque uses {TORQUE_NM_PER_AMP:.3f} Nm/A from XC330 datasheet specs."
		),
		fontsize=11,
	)
	fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])

	out_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(out_path)
	plt.close(fig)


def _build_output_dir(base_out: str, motor: str, run_name: str) -> Path:
	stamp = time.strftime("%Y%m%d_%H%M%S")
	suffix = run_name if run_name else f"{motor}_{stamp}"
	return Path(base_out) / suffix


def main() -> None:
	parser = argparse.ArgumentParser(
		description=(
			"Run stitched sine+square scaled commands on one LEAP joint and record replayable trajectory telemetry."
		)
	)
	parser.add_argument("--port", default=Args.port)
	parser.add_argument("--baudrate", type=int, default=Args.baudrate)
	parser.add_argument("--motor", default=Args.motor)
	parser.add_argument(
		"--use_mj_motor_config",
		action=argparse.BooleanOptionalAction,
		default=Args.use_mj_motor_config,
		help="Use LEAP MJ motor naming/config (default: true).",
	)
	parser.add_argument("--control_hz", type=float, default=Args.control_hz)
	parser.add_argument("--out_dir", default=Args.out_dir)
	parser.add_argument("--run_name", default=Args.run_name)
	parser.add_argument(
		"--log_level",
		default=Args.log_level,
		choices=["DEBUG", "INFO", "WARNING", "ERROR"],
	)
	args = parser.parse_args()

	logging.basicConfig(
		level=getattr(logging, args.log_level),
		format="%(asctime)s %(levelname)s %(message)s",
	)

	if args.control_hz <= 0.0:
		raise ValueError("control_hz must be > 0")

	hand = LeapHand(
		LeapHandConfig(
			port=args.port,
			baudrate=args.baudrate,
			use_mj_motor_config=args.use_mj_motor_config,
		)
	)

	schedule = _default_wave_schedule()
	output_dir = _build_output_dir(args.out_dir, args.motor, args.run_name)
	output_dir.mkdir(parents=True, exist_ok=True)

	logger.info("Connecting to LEAP hand on %s", args.port)
	hand.connect()
	try:
		if args.motor not in hand.bus.motors:
			raise KeyError(
				f"Unknown motor '{args.motor}'. Available motors: {list(hand.bus.motors.keys())}"
			)

		logger.info("Moving hand to MJ_ZERO_POSITION with hand.move(duration=3.0)")
		hand.move(MJ_ZERO_POSITION, duration=3.0)

		total_duration = sum(seg.duration_s for seg in schedule)
		logger.info("Running stitched trajectory on %s, total duration %.2f s", args.motor, total_duration)
		for seg in schedule:
			logger.info(
				(
					"Segment %-16s kind=%s duration=%.2fs amp=%.3f freq=%.3fHz "
					"offset=%.3f phase=%.3f duty=%.3f"
				),
				seg.name,
				seg.kind,
				seg.duration_s,
				seg.amplitude,
				seg.frequency_hz,
				seg.offset,
				seg.phase_rad,
				seg.duty_cycle,
			)

		dt = 1.0 / args.control_hz
		zero_scaled_action = {}
		command_key = f"{args.motor}.pos"
		hand.move({command_key: 0.0}, duration=1.0,scaled=True)

		rows: list[dict[str, float | str]] = []
		t_global = 0.0
		run_start = time.monotonic()

		for segment in schedule:
			n_steps = max(1, int(round(segment.duration_s * args.control_hz)))
			segment_start = time.monotonic()

			for step in range(n_steps):
				t_seg = step * dt
				command_scaled = _eval_wave_scaled(segment, t_seg)
				print(command_scaled)
				action = dict(zero_scaled_action)
				action[command_key] = command_scaled
				hand.send_action_scaled(action)

				pos_tick, vel_lsb, cur_lsb, pos_traj_tick = _read_joint_telemetry(hand, args.motor)

				pos_rad = _ticks_to_rad(pos_tick)
				pos_traj_rad = _ticks_to_rad(pos_traj_tick)
				vel_rad_s = _velocity_lsb_to_rad_s(vel_lsb)
				cur_mA = _current_lsb_to_milliamp(cur_lsb)
				torque_nm = _current_milliamp_to_torque_nm(cur_mA)

				now = time.monotonic()
				t_global = now - run_start

				rows.append(
					{
						"time_s": t_global,
						"segment": segment.name,
						"segment_kind": segment.kind,
						"segment_t_s": t_seg,
						"command_scaled": command_scaled,
						"position_tick": pos_tick,
						"position_rad": pos_rad,
						"position_trajectory_tick": pos_traj_tick,
						"position_trajectory_rad": pos_traj_rad,
						"velocity_lsb": vel_lsb,
						"velocity_rad_s": vel_rad_s,
						"current_lsb": cur_lsb,
						"current_mA": cur_mA,
						"torque_nm_from_current": torque_nm,
					}
				)

				next_time = segment_start + (step + 1) * dt
				sleep_s = next_time - time.monotonic()
				if sleep_s > 0.0:
					time.sleep(sleep_s)

		# hand.send_action_scaled(zero_scaled_action)
		hand.move(MJ_ZERO_POSITION, duration=2.0)

		trajectory_csv = output_dir / "trajectory.csv"
		replay_csv = output_dir / "replay_command.csv"
		metadata_json = output_dir / "metadata.json"
		data_npz = output_dir / "trajectory.npz"
		plot_png = output_dir / "trajectory_plot.png"

		_save_csv(
			trajectory_csv,
			rows,
			[
				"time_s",
				"segment",
				"segment_kind",
				"segment_t_s",
				"command_scaled",
				"position_tick",
				"position_rad",
				"position_trajectory_tick",
				"position_trajectory_rad",
				"velocity_lsb",
				"velocity_rad_s",
				"current_lsb",
				"current_mA",
				"torque_nm_from_current",
			],
		)

		replay_rows = [
			{
				"time_s": float(r["time_s"]),
				"motor": args.motor,
				"command_scaled": float(r["command_scaled"]),
			}
			for r in rows
		]
		_save_csv(
			replay_csv,
			replay_rows,
			["time_s", "motor", "command_scaled", "command_rad"],
		)

		metadata = {
			"script": "sysid/collect_trajectory.py",
			"created_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
			"port": args.port,
			"baudrate": args.baudrate,
			"motor": args.motor,
			"use_mj_motor_config": args.use_mj_motor_config,
			"control_hz": args.control_hz,
			"total_samples": len(rows),
			"total_duration_s": float(rows[-1]["time_s"]) if rows else 0.0,
			"start_pose": "MJ_ZERO_POSITION via hand.move(duration=3.0)",
			"command_mode": "scaled via hand.send_action_scaled, range [-1, 1]",
			"wave_schedule": [asdict(seg) for seg in schedule],
			"units": {
				"Present_Position": "1 pulse",
				"Present_Velocity": "0.229 rev/min per LSB",
				"Present_Current": "1.0 mA per LSB",
				"position_rad": "tick * 2*pi/4096 - pi",
				"velocity_rad_s": "vel_lsb * 0.229 * 2*pi/60",
				"current_mA": "current_lsb * 1.0",
				"torque_nm": f"(current_mA/1000) * {TORQUE_NM_PER_AMP}",
			},
			"torque_note": (
				"Torque conversion uses XC330 datasheet slope ~0.515 Nm/A from stall points; "
				"for XC330, Present Current is input current, so torque is an approximation."
			),
			"files": {
				"trajectory_csv": str(trajectory_csv),
				"replay_csv": str(replay_csv),
				"data_npz": str(data_npz),
				"plot_png": str(plot_png),
			},
		}
		with metadata_json.open("w") as f:
			json.dump(metadata, f, indent=2)

		np.savez(
			data_npz,
			time_s=np.asarray([float(r["time_s"]) for r in rows], dtype=np.float64),
			command_scaled=np.asarray([float(r["command_scaled"]) for r in rows], dtype=np.float64),
			position_tick=np.asarray([float(r["position_tick"]) for r in rows], dtype=np.float64),
			position_rad=np.asarray([float(r["position_rad"]) for r in rows], dtype=np.float64),
			position_trajectory_tick=np.asarray([float(r["position_trajectory_tick"]) for r in rows], dtype=np.float64),
			position_trajectory_rad=np.asarray([float(r["position_trajectory_rad"]) for r in rows], dtype=np.float64),
			velocity_lsb=np.asarray([float(r["velocity_lsb"]) for r in rows], dtype=np.float64),
			velocity_rad_s=np.asarray([float(r["velocity_rad_s"]) for r in rows], dtype=np.float64),
			current_lsb=np.asarray([float(r["current_lsb"]) for r in rows], dtype=np.float64),
			current_mA=np.asarray([float(r["current_mA"]) for r in rows], dtype=np.float64),
			torque_nm=np.asarray([float(r["torque_nm_from_current"]) for r in rows], dtype=np.float64),
		)

		_save_plot(rows, plot_png, args.motor)

		logger.info("Saved trajectory CSV: %s", trajectory_csv)
		logger.info("Saved replay CSV: %s", replay_csv)
		logger.info("Saved metadata JSON: %s", metadata_json)
		logger.info("Saved NPZ: %s", data_npz)
		logger.info("Saved plot: %s", plot_png)

	finally:
		logger.info("Disconnecting hand...")
		hand.disconnect()


if __name__ == "__main__":
	main()
