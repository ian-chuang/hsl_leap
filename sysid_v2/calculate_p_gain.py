import argparse
import logging
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hsl_leap.leap_hand import LeapHand, LeapHandConfig
from hsl_leap.motors.dynamixel import OperatingMode

logger = logging.getLogger(__name__)

# XC330 control table conversions
DXL_TICKS_PER_REV = 4096.0
RAD_PER_TICK = 2.0 * np.pi / DXL_TICKS_PER_REV
AMP_PER_CURRENT_LSB = 0.001
TORQUE_NM_PER_AMP = 0.515


def _fit_through_origin(x: np.ndarray, y: np.ndarray) -> float:
	denom = float(np.dot(x, x))
	if denom < 1e-12:
		return 0.0
	return float(np.dot(x, y) / denom)


def _set_motor_gains(hand: LeapHand, motor: str, kp: int, current_limit: int) -> None:
	hand.bus.disable_torque(motor)
	hand.bus.write("Operating_Mode", motor, OperatingMode.CURRENT_POSITION.value)
	hand.bus.write("Position_P_Gain", motor, kp)
	hand.bus.write("Position_I_Gain", motor, 0)
	hand.bus.write("Position_D_Gain", motor, 0)
	hand.bus.write("Current_Limit", motor, current_limit)
	hand.bus.enable_torque(motor)


def _sample_register(hand: LeapHand, data_name: str, motor: str) -> float:
	vals = hand.bus.sync_read(data_name, motors=[motor], normalize=False, num_retry=hand.config.read_num_retries)
	return float(vals[motor])


def _move_to_start(hand: LeapHand, motor: str, settle_s: float) -> None:
	obs = hand.get_observation()
	joint_key = f"{motor}.pos"
	if joint_key not in obs:
		raise KeyError(f"Joint '{joint_key}' not found in observation.")
	hand.move({joint_key: 0}, duration=1.0, scaled=True)
	if settle_s > 0:
		time.sleep(settle_s)


def _run_square_test(
	hand: LeapHand,
	motor: str,
	sample_hz: float,
	record_s: float,
	period_s: float,
	current_limit: int,
	amplitude: float,
) -> dict[str, np.ndarray]:
	cal = hand.calibration[motor]
	rng_min = int(cal.range_min)
	rng_max = int(cal.range_max)

	dt = 1.0 / sample_hz
	n = max(1, int(round(record_s * sample_hz)))

	times = []
	traj_ticks = []
	present_ticks = []
	present_cur = []

	t0 = time.monotonic()
	for _ in range(n):
		start = time.monotonic()
		t = start - t0
		phase = (t % period_s) / period_s
		action = amplitude if phase < 0.5 else 0.0
		desired = int(round(((action + 1.0) / 2.0) * (rng_max - rng_min) + rng_min))

		hand.bus.sync_write("Goal_Position", {motor: desired}, normalize=False)

		traj = _sample_register(hand, "Position_Trajectory", motor)
		pres = _sample_register(hand, "Present_Position", motor)
		cur = _sample_register(hand, "Present_Current", motor)

		if abs(cur) >= current_limit * 0.9:
			logger.warning(
				"Current near limit: motor=%s current_lsb=%.1f limit_lsb=%d",
				motor,
				cur,
				current_limit,
			)

		times.append(t)
		traj_ticks.append(traj)
		present_ticks.append(pres)
		present_cur.append(cur)

		sleep = dt - (time.monotonic() - start)
		if sleep > 0:
			time.sleep(sleep)

	return {
		"time_s": np.asarray(times),
		"position_trajectory_tick": np.asarray(traj_ticks),
		"present_position_tick": np.asarray(present_ticks),
		"present_current_lsb": np.asarray(present_cur),
	}


def _plot_results(data: dict[str, np.ndarray], out_png: Path) -> float:
	t = data["time_s"]
	traj_tick = data["position_trajectory_tick"]
	pres_tick = data["present_position_tick"]
	cur_lsb = data["present_current_lsb"]

	err_tick = traj_tick - pres_tick
	err_rad = err_tick * RAD_PER_TICK
	torque_nm = cur_lsb * AMP_PER_CURRENT_LSB * TORQUE_NM_PER_AMP

	mask = np.abs(err_tick) > 1.0
	slope_nm_per_rad = _fit_through_origin(err_rad[mask], torque_nm[mask])

	fig, ax = plt.subplots(1, 2, figsize=(12, 4))

	ax[0].plot(t, traj_tick * RAD_PER_TICK, label="trajectory_rad")
	ax[0].plot(t, pres_tick * RAD_PER_TICK, label="present_rad")
	ax[0].set_xlabel("time [s]")
	ax[0].set_ylabel("position [rad]")
	ax[0].grid(True)
	ax[0].legend(fontsize="small")

	ax_torque = ax[0].twinx()
	ax_torque.plot(t, torque_nm, color="tab:red", alpha=0.6, label="torque_nm")
	ax_torque.set_ylabel("torque [Nm]")
	ax_torque.legend(fontsize="small", loc="lower right")

	ax[1].scatter(err_rad, torque_nm, s=6, alpha=0.7)
	if err_rad.size:
		lo = float(np.min(err_rad))
		hi = float(np.max(err_rad))
		xline = np.linspace(lo, hi, 200)
		yline = slope_nm_per_rad * xline
		ax[1].plot(xline, yline, "r", label=f"fit slope = {slope_nm_per_rad:.4f} Nm/rad")
		ax[1].legend(fontsize="small")
	ax[1].set_xlabel("position_error [rad]")
	ax[1].set_ylabel("torque [Nm]")
	ax[1].grid(True)

	fig.tight_layout()
	out_png.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(out_png)
	plt.close(fig)

	return slope_nm_per_rad

# 0.632743
# 0.667444
# 0.661797
# 0.642701
# like 0.6 or something...
def main() -> None:
	parser = argparse.ArgumentParser(description="Square-wave P gain measurement")
	parser.add_argument("--port", default="/dev/ttyDXL_leap_hand")
	parser.add_argument("--baudrate", type=int, default=4_000_000)
	parser.add_argument("--motor", default="if_dip")
	parser.add_argument("--sample_hz", type=float, default=50.0)
	parser.add_argument("--record_s", type=float, default=6.0)
	parser.add_argument("--period_s", type=float, default=1.0)
	parser.add_argument("--amplitude", type=float, default=0.05)
	parser.add_argument("--current_limit", type=int, default=300)
	parser.add_argument("--kp_test", type=int, default=1100)
	parser.add_argument("--out_dir", default="sysid/outputs/p_gains")
	parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
	args = parser.parse_args()

	logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")

	cfg = LeapHandConfig(port=args.port, baudrate=args.baudrate)
	hand = LeapHand(cfg)

	out_dir = Path(args.out_dir)
	stamp = time.strftime("%Y%m%d_%H%M%S")
	out_png = out_dir / f"kp_square_{args.motor}_{stamp}.png"

	logger.info("Connecting to hand...")
	hand.connect()
	try:
		if args.motor not in hand.bus.motors:
			raise KeyError(f"Unknown motor '{args.motor}'. Available: {list(hand.bus.motors.keys())}")

		_set_motor_gains(hand, args.motor, kp=args.kp_test, current_limit=args.current_limit)
		_move_to_start(hand, args.motor, settle_s=max(0.75, 1.5 / max(args.sample_hz, 1.0)))

		data = _run_square_test(
			hand=hand,
			motor=args.motor,
			sample_hz=args.sample_hz,
			record_s=args.record_s,
			period_s=args.period_s,
			current_limit=args.current_limit,
			amplitude=args.amplitude,
		)

		slope_nm_per_rad = _plot_results(data, out_png)
		logger.info("P gain slope: %.6f Nm/rad", slope_nm_per_rad)
		logger.info("Saved plot: %s", out_png)
	finally:
		logger.info("Disconnecting...")
		hand.disconnect()


if __name__ == "__main__":
	main()
