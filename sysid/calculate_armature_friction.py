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
class SineSegment:
    name: str
    duration_s: float
    amplitude: float
    frequency_hz: float
    offset: float = 0.0
    phase_rad: float = 0.0


@dataclass
class Args:
    port: str = "/dev/ttyDXL_leap_hand"
    baudrate: int = 4_000_000
    motor: str = "if_dip"
    use_mj_motor_config: bool = True
    control_hz: float = 60.0
    out_dir: str = "sysid/outputs/armature_friction"
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


def _scaled_to_tick(command_scaled: float, range_min: int, range_max: int) -> float:
    bounded = float(np.clip(command_scaled, -1.0, 1.0))
    normalized = bounded * 100.0
    return ((normalized + 100.0) / 200.0) * (range_max - range_min) + range_min


def _default_sine_schedule() -> list[SineSegment]:
    return [
        SineSegment(name="sine_0", duration_s=6.0, amplitude=0.5, frequency_hz=0.30),
        SineSegment(name="sine_1", duration_s=6.0, amplitude=0.8, frequency_hz=0.55),
        SineSegment(name="sine_2", duration_s=6.0, amplitude=1.0, frequency_hz=0.85),
    ]


def _eval_sine_scaled(segment: SineSegment, t_seg: float) -> float:
    omega = 2.0 * math.pi * segment.frequency_hz
    phase = omega * t_seg + segment.phase_rad
    command = segment.offset + segment.amplitude * math.sin(phase)
    return float(np.clip(command, -1.0, 1.0))


def _read_joint_telemetry(hand: LeapHand, motor: str) -> tuple[float, float, float, float, float]:
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
    trajectory_position_tick = float(
        hand.bus.sync_read(
            "Position_Trajectory",
            motors=[motor],
            normalize=False,
            num_retry=hand.config.read_num_retries,
        )[motor]
    )
    trajectory_velocity_lsb = float(
        hand.bus.sync_read(
            "Velocity_Trajectory",
            motors=[motor],
            normalize=False,
            num_retry=hand.config.read_num_retries,
        )[motor]
    )
    return position_tick, velocity_lsb, current_lsb, trajectory_position_tick, trajectory_velocity_lsb


def _build_output_dir(base_out: str, motor: str, run_name: str) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    suffix = run_name if run_name else f"{motor}_{stamp}"
    return Path(base_out) / suffix


def _save_csv(path: Path, rows: list[dict[str, float | str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fit_armature_friction_pd(
    time_s: np.ndarray,
    trajectory_position_rad: np.ndarray,
    trajectory_velocity_rad_s: np.ndarray,
    position_rad: np.ndarray,
    velocity_rad_s: np.ndarray,
    torque_nm: np.ndarray,
    min_abs_velocity_rad_s: float,
    min_abs_error_rad: float,
) -> dict[str, float | np.ndarray]:
    dt = float(np.median(np.diff(time_s))) if time_s.size > 1 else 0.0
    if dt <= 0.0:
        raise ValueError("Insufficient or invalid timestamps for fitting.")

    accel_rad_s2 = np.gradient(velocity_rad_s, dt)
    position_error_rad = trajectory_position_rad - position_rad
    velocity_error_rad_s = trajectory_velocity_rad_s - velocity_rad_s

    # Use tanh-based sign approximation for smoother Coulomb estimation near zero speed.
    tanh_scale = max(min_abs_velocity_rad_s, 1e-3)
    sign_smooth = np.tanh(velocity_rad_s / tanh_scale)

    fit_mask_base = np.abs(velocity_rad_s) >= min_abs_velocity_rad_s
    if int(np.sum(fit_mask_base)) < 20:
        raise ValueError(
            "Not enough informative samples for robust base fit. Increase excitation or reduce velocity threshold."
        )

    # Stage 1: fit armature/friction model.
    x_base = np.column_stack(
        [
            accel_rad_s2[fit_mask_base],
            velocity_rad_s[fit_mask_base],
            sign_smooth[fit_mask_base],
            np.ones(int(np.sum(fit_mask_base))),
        ]
    )
    y_base = torque_nm[fit_mask_base]

    beta_base, _, _, _ = np.linalg.lstsq(x_base, y_base, rcond=None)
    inertia, viscous, coulomb, bias = [float(v) for v in beta_base]

    torque_base_full = (
        inertia * accel_rad_s2
        + viscous * velocity_rad_s
        + coulomb * sign_smooth
        + bias
    )

    # Stage 2: fit PD gains on residual torque.
    residual_torque = torque_nm - torque_base_full
    fit_mask_pd = (np.abs(position_error_rad) >= min_abs_error_rad) | (
        np.abs(velocity_error_rad_s) >= min_abs_velocity_rad_s
    )
    if int(np.sum(fit_mask_pd)) < 20:
        raise ValueError(
            "Not enough informative samples for PD fit. Increase tracking excitation or reduce error threshold."
        )

    x_pd = np.column_stack(
        [
            position_error_rad[fit_mask_pd],
            velocity_error_rad_s[fit_mask_pd],
            np.ones(int(np.sum(fit_mask_pd))),
        ]
    )
    y_pd = residual_torque[fit_mask_pd]

    beta_pd, _, _, _ = np.linalg.lstsq(x_pd, y_pd, rcond=None)
    kp_eff, kd_eff, pd_bias = [float(v) for v in beta_pd]

    torque_pred_full = torque_base_full + kp_eff * position_error_rad + kd_eff * velocity_error_rad_s + pd_bias

    fit_mask = fit_mask_base | fit_mask_pd
    y_eval = torque_nm[fit_mask]
    y_hat = torque_pred_full[fit_mask]

    ss_res = float(np.sum((y_eval - y_hat) ** 2))
    ss_tot = float(np.sum((y_eval - np.mean(y_eval)) ** 2))
    r2 = 0.0 if ss_tot < 1e-12 else 1.0 - (ss_res / ss_tot)
    rmse = float(np.sqrt(np.mean((y_eval - y_hat) ** 2)))

    return {
        "inertia": inertia,
        "viscous": viscous,
        "coulomb": coulomb,
        "kp_eff": kp_eff,
        "kd_eff": kd_eff,
        "bias": bias,
        "pd_bias": pd_bias,
        "r2": r2,
        "rmse": rmse,
        "accel_rad_s2": accel_rad_s2,
        "position_error_rad": position_error_rad,
        "velocity_error_rad_s": velocity_error_rad_s,
        "torque_base_full": torque_base_full,
        "torque_pred_full": torque_pred_full,
        "fit_mask": fit_mask,
        "fit_mask_base": fit_mask_base,
        "fit_mask_pd": fit_mask_pd,
    }


def _save_plots(
    rows: list[dict[str, float | str]],
    out_traj: Path,
    out_fit: Path,
    motor: str,
    fit_result: dict[str, float | np.ndarray],
) -> None:
    t = np.asarray([float(r["time_s"]) for r in rows], dtype=np.float64)
    cmd_scaled = np.asarray([float(r["command_scaled"]) for r in rows], dtype=np.float64)
    pos_rad = np.asarray([float(r["position_rad"]) for r in rows], dtype=np.float64)
    vel_rad_s = np.asarray([float(r["velocity_rad_s"]) for r in rows], dtype=np.float64)
    cur_mA = np.asarray([float(r["current_mA"]) for r in rows], dtype=np.float64)
    torque_nm = np.asarray([float(r["torque_nm_from_current"]) for r in rows], dtype=np.float64)

    accel = np.asarray(fit_result["accel_rad_s2"], dtype=np.float64)
    torque_pred = np.asarray(fit_result["torque_pred_full"], dtype=np.float64)
    fit_mask = np.asarray(fit_result["fit_mask"], dtype=bool)

    fig, axes = plt.subplots(6, 1, figsize=(12, 15), sharex=True)

    axes[0].plot(t, cmd_scaled, label="command_scaled [-1,1]")
    axes[0].set_ylabel("cmd")
    axes[0].legend(fontsize="small")
    axes[0].grid(True)

    axes[1].plot(t, pos_rad, label="position_rad")
    axes[1].set_ylabel("rad")
    axes[1].legend(fontsize="small")
    axes[1].grid(True)

    axes[2].plot(t, vel_rad_s, color="tab:orange", label="velocity_rad_s")
    axes[2].set_ylabel("rad/s")
    axes[2].legend(fontsize="small")
    axes[2].grid(True)

    axes[3].plot(t, accel, color="tab:purple", label="acceleration_rad_s2")
    axes[3].set_ylabel("rad/s²")
    axes[3].legend(fontsize="small")
    axes[3].grid(True)

    axes[4].plot(t, cur_mA, color="tab:green", label="current_mA")
    axes[4].set_ylabel("mA")
    axes[4].legend(fontsize="small")
    axes[4].grid(True)

    axes[5].plot(t, torque_nm, color="tab:red", label="torque_meas_nm")
    axes[5].plot(t, torque_pred, color="black", alpha=0.7, label="torque_fit_nm")
    axes[5].fill_between(t, np.min(torque_nm), np.max(torque_nm), where=~fit_mask, color="gray", alpha=0.15, label="excluded")
    axes[5].set_xlabel("time [s]")
    axes[5].set_ylabel("Nm")
    axes[5].legend(fontsize="small")
    axes[5].grid(True)

    fig.suptitle(f"Armature + friction identification traces ({motor})", fontsize=11)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])
    out_traj.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_traj)
    plt.close(fig)

    fig2, ax2 = plt.subplots(1, 1, figsize=(6, 6))
    ax2.scatter(torque_nm[fit_mask], torque_pred[fit_mask], s=8, alpha=0.6)
    low = float(min(np.min(torque_nm[fit_mask]), np.min(torque_pred[fit_mask])))
    high = float(max(np.max(torque_nm[fit_mask]), np.max(torque_pred[fit_mask])))
    ax2.plot([low, high], [low, high], "k--", linewidth=1.0)
    ax2.set_xlabel("Measured torque [Nm]")
    ax2.set_ylabel("Predicted torque [Nm]")
    ax2.set_title("Fit quality scatter")
    ax2.grid(True)
    fig2.tight_layout()
    out_fit.parent.mkdir(parents=True, exist_ok=True)
    fig2.savefig(out_fit)
    plt.close(fig2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate armature and friction terms from sine-driven single-joint data.")
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
    parser.add_argument("--min_abs_velocity_rad_s", type=float, default=0.2)
    parser.add_argument("--min_abs_error_rad", type=float, default=0.02)
    parser.add_argument("--out_dir", default=Args.out_dir)
    parser.add_argument("--run_name", default=Args.run_name)
    parser.add_argument(
        "--log_level",
        default=Args.log_level,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")

    if args.control_hz <= 0.0:
        raise ValueError("control_hz must be > 0")

    hand = LeapHand(
        LeapHandConfig(
            port=args.port,
            baudrate=args.baudrate,
            use_mj_motor_config=args.use_mj_motor_config,
        )
    )

    schedule = _default_sine_schedule()
    output_dir = _build_output_dir(args.out_dir, args.motor, args.run_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Connecting to LEAP hand on %s", args.port)
    hand.connect()
    try:
        if args.motor not in hand.bus.motors:
            raise KeyError(f"Unknown motor '{args.motor}'. Available motors: {list(hand.bus.motors.keys())}")

        logger.info("Moving hand to MJ_ZERO_POSITION with hand.move(duration=3.0)")
        hand.move(MJ_ZERO_POSITION, duration=3.0)

        logger.info("Centering target joint to scaled 0.0")
        hand.move({f"{args.motor}.pos": 0.0}, duration=1.0, scaled=True)

        cal = hand.calibration[args.motor]

        total_duration = sum(seg.duration_s for seg in schedule)
        logger.info("Running sine schedule on %s for %.2f s", args.motor, total_duration)
        for seg in schedule:
            logger.info(
                "Segment %-10s duration=%.2fs amp=%.3f freq=%.3fHz offset=%.3f phase=%.3f",
                seg.name,
                seg.duration_s,
                seg.amplitude,
                seg.frequency_hz,
                seg.offset,
                seg.phase_rad,
            )

        dt = 1.0 / args.control_hz
        command_key = f"{args.motor}.pos"

        rows: list[dict[str, float | str]] = []
        run_start = time.monotonic()

        for segment in schedule:
            n_steps = max(1, int(round(segment.duration_s * args.control_hz)))
            seg_start = time.monotonic()

            for step in range(n_steps):
                t_seg = step * dt
                command_scaled = _eval_sine_scaled(segment, t_seg)
                command_tick = _scaled_to_tick(command_scaled, cal.range_min, cal.range_max)
                command_rad = _ticks_to_rad(command_tick)

                hand.send_action_scaled({command_key: command_scaled})

                pos_tick, vel_lsb, cur_lsb, traj_pos_tick, traj_vel_lsb = _read_joint_telemetry(hand, args.motor)
                pos_rad = _ticks_to_rad(pos_tick)
                vel_rad_s = _velocity_lsb_to_rad_s(vel_lsb)
                cur_mA = _current_lsb_to_milliamp(cur_lsb)
                torque_nm = _current_milliamp_to_torque_nm(cur_mA)
                traj_pos_rad = _ticks_to_rad(traj_pos_tick)
                traj_vel_rad_s = _velocity_lsb_to_rad_s(traj_vel_lsb)

                rows.append(
                    {
                        "time_s": time.monotonic() - run_start,
                        "segment": segment.name,
                        "segment_t_s": t_seg,
                        "command_scaled": command_scaled,
                        "command_tick": command_tick,
                        "command_rad": command_rad,
                        "trajectory_position_tick": traj_pos_tick,
                        "trajectory_position_rad": traj_pos_rad,
                        "trajectory_velocity_lsb": traj_vel_lsb,
                        "trajectory_velocity_rad_s": traj_vel_rad_s,
                        "position_tick": pos_tick,
                        "position_rad": pos_rad,
                        "velocity_lsb": vel_lsb,
                        "velocity_rad_s": vel_rad_s,
                        "current_lsb": cur_lsb,
                        "current_mA": cur_mA,
                        "torque_nm_from_current": torque_nm,
                    }
                )

                next_time = seg_start + (step + 1) * dt
                sleep_s = next_time - time.monotonic()
                if sleep_s > 0.0:
                    time.sleep(sleep_s)

        logger.info("Returning hand to MJ_ZERO_POSITION")
        hand.move(MJ_ZERO_POSITION, duration=2.0)

        time_s = np.asarray([float(r["time_s"]) for r in rows], dtype=np.float64)
        trajectory_position_rad = np.asarray([float(r["trajectory_position_rad"]) for r in rows], dtype=np.float64)
        trajectory_velocity_rad_s = np.asarray([float(r["trajectory_velocity_rad_s"]) for r in rows], dtype=np.float64)
        position_rad = np.asarray([float(r["position_rad"]) for r in rows], dtype=np.float64)
        vel_rad_s = np.asarray([float(r["velocity_rad_s"]) for r in rows], dtype=np.float64)
        torque_nm = np.asarray([float(r["torque_nm_from_current"]) for r in rows], dtype=np.float64)

        fit_result = _fit_armature_friction_pd(
            time_s=time_s,
            trajectory_position_rad=trajectory_position_rad,
            trajectory_velocity_rad_s=trajectory_velocity_rad_s,
            position_rad=position_rad,
            velocity_rad_s=vel_rad_s,
            torque_nm=torque_nm,
            min_abs_velocity_rad_s=args.min_abs_velocity_rad_s,
            min_abs_error_rad=args.min_abs_error_rad,
        )

        results = {
            "inertia_kgm2": float(fit_result["inertia"]),
            "viscous_Nm_per_rad_s": float(fit_result["viscous"]),
            "coulomb_Nm": float(fit_result["coulomb"]),
            "kp_eff_Nm_per_rad": float(fit_result["kp_eff"]),
            "kd_eff_Nm_s_per_rad": float(fit_result["kd_eff"]),
            "bias_Nm": float(fit_result["bias"]),
            "pd_bias_Nm": float(fit_result["pd_bias"]),
            "fit_r2": float(fit_result["r2"]),
            "fit_rmse_Nm": float(fit_result["rmse"]),
            "fit_samples": int(np.sum(np.asarray(fit_result["fit_mask"], dtype=bool))),
            "fit_samples_base": int(np.sum(np.asarray(fit_result["fit_mask_base"], dtype=bool))),
            "fit_samples_pd": int(np.sum(np.asarray(fit_result["fit_mask_pd"], dtype=bool))),
            "total_samples": int(len(rows)),
        }

        logger.info("Estimated inertia [kg*m^2]: %.6e", results["inertia_kgm2"])
        logger.info("Estimated viscous [Nm/(rad/s)]: %.6e", results["viscous_Nm_per_rad_s"])
        logger.info("Estimated coulomb [Nm]: %.6e", results["coulomb_Nm"])
        logger.info("Estimated Kp_eff [Nm/rad]: %.6e", results["kp_eff_Nm_per_rad"])
        logger.info("Estimated Kd_eff [Nm*s/rad]: %.6e", results["kd_eff_Nm_s_per_rad"])
        logger.info("Estimated bias [Nm]: %.6e", results["bias_Nm"])
        logger.info("Estimated pd_bias [Nm]: %.6e", results["pd_bias_Nm"])
        logger.info("Fit quality: R2=%.4f RMSE=%.6f Nm", results["fit_r2"], results["fit_rmse_Nm"])

        trajectory_csv = output_dir / "trajectory.csv"
        summary_json = output_dir / "fit_summary.json"
        summary_csv = output_dir / "fit_summary.csv"
        data_npz = output_dir / "fit_data.npz"
        plot_traj = output_dir / "trajectory_plot.png"
        plot_fit = output_dir / "fit_scatter.png"
        metadata_json = output_dir / "metadata.json"

        _save_csv(
            trajectory_csv,
            rows,
            [
                "time_s",
                "segment",
                "segment_t_s",
                "command_scaled",
                "command_tick",
                "command_rad",
                "trajectory_position_tick",
                "trajectory_position_rad",
                "trajectory_velocity_lsb",
                "trajectory_velocity_rad_s",
                "position_tick",
                "position_rad",
                "velocity_lsb",
                "velocity_rad_s",
                "current_lsb",
                "current_mA",
                "torque_nm_from_current",
            ],
        )

        _save_csv(
            summary_csv,
            [
                {"name": "inertia_kgm2", "value": results["inertia_kgm2"]},
                {"name": "viscous_Nm_per_rad_s", "value": results["viscous_Nm_per_rad_s"]},
                {"name": "coulomb_Nm", "value": results["coulomb_Nm"]},
                {"name": "kp_eff_Nm_per_rad", "value": results["kp_eff_Nm_per_rad"]},
                {"name": "kd_eff_Nm_s_per_rad", "value": results["kd_eff_Nm_s_per_rad"]},
                {"name": "bias_Nm", "value": results["bias_Nm"]},
                {"name": "pd_bias_Nm", "value": results["pd_bias_Nm"]},
                {"name": "fit_r2", "value": results["fit_r2"]},
                {"name": "fit_rmse_Nm", "value": results["fit_rmse_Nm"]},
            ],
            ["name", "value"],
        )

        with summary_json.open("w") as f:
            json.dump(results, f, indent=2)

        accel = np.asarray(fit_result["accel_rad_s2"], dtype=np.float64)
        pos_err = np.asarray(fit_result["position_error_rad"], dtype=np.float64)
        vel_err = np.asarray(fit_result["velocity_error_rad_s"], dtype=np.float64)
        torque_base = np.asarray(fit_result["torque_base_full"], dtype=np.float64)
        torque_pred = np.asarray(fit_result["torque_pred_full"], dtype=np.float64)
        fit_mask = np.asarray(fit_result["fit_mask"], dtype=bool)
        fit_mask_base = np.asarray(fit_result["fit_mask_base"], dtype=bool)
        fit_mask_pd = np.asarray(fit_result["fit_mask_pd"], dtype=bool)

        np.savez(
            data_npz,
            time_s=time_s,
            command_scaled=np.asarray([float(r["command_scaled"]) for r in rows], dtype=np.float64),
            command_rad=np.asarray([float(r["command_rad"]) for r in rows], dtype=np.float64),
            trajectory_position_rad=trajectory_position_rad,
            trajectory_velocity_rad_s=trajectory_velocity_rad_s,
            position_rad=position_rad,
            velocity_rad_s=vel_rad_s,
            acceleration_rad_s2=accel,
            position_error_rad=pos_err,
            velocity_error_rad_s=vel_err,
            current_mA=np.asarray([float(r["current_mA"]) for r in rows], dtype=np.float64),
            torque_measured_nm=torque_nm,
            torque_base_nm=torque_base,
            torque_predicted_nm=torque_pred,
            fit_mask=fit_mask.astype(np.int8),
            fit_mask_base=fit_mask_base.astype(np.int8),
            fit_mask_pd=fit_mask_pd.astype(np.int8),
        )

        _save_plots(rows, plot_traj, plot_fit, args.motor, fit_result)

        metadata = {
            "script": "sysid/calculate_armature_friction.py",
            "created_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
            "port": args.port,
            "baudrate": args.baudrate,
            "motor": args.motor,
            "control_hz": args.control_hz,
            "start_pose": "MJ_ZERO_POSITION via hand.move(duration=3.0)",
            "command_mode": "scaled via hand.send_action_scaled, range [-1, 1]",
            "sine_schedule": [asdict(seg) for seg in schedule],
            "fit_model": "two-stage: tau_base=J*qdd + B*qd + Fc*tanh(qd/v_eps)+tau_bias, then tau_residual=Kp*(q_traj-q)+Kd*(qd_traj-qd)+pd_bias",
            "fit_velocity_threshold_rad_s": args.min_abs_velocity_rad_s,
            "fit_position_error_threshold_rad": args.min_abs_error_rad,
            "units": {
                "Present_Position": "1 pulse",
                "Present_Velocity": "0.229 rev/min per LSB",
                "Present_Current": "1.0 mA per LSB",
                "command_rad": "command_scaled mapped with motor calibration limits",
                "trajectory_position_rad": "Position_Trajectory tick * 2*pi/4096 - pi",
                "trajectory_velocity_rad_s": "Velocity_Trajectory lsb * 0.229 * 2*pi/60",
                "position_rad": "tick * 2*pi/4096 - pi",
                "velocity_rad_s": "vel_lsb * 0.229 * 2*pi/60",
                "torque_nm": f"(current_mA/1000) * {TORQUE_NM_PER_AMP}",
            },
            "torque_note": (
                "Torque conversion uses XC330 datasheet slope ~0.515 Nm/A from stall points; "
                "for XC330, Present Current is input current, so torque is approximate."
            ),
            "results": results,
            "files": {
                "trajectory_csv": str(trajectory_csv),
                "summary_json": str(summary_json),
                "summary_csv": str(summary_csv),
                "fit_data_npz": str(data_npz),
                "trajectory_plot": str(plot_traj),
                "fit_scatter_plot": str(plot_fit),
            },
        }
        with metadata_json.open("w") as f:
            json.dump(metadata, f, indent=2)

        logger.info("Saved trajectory CSV: %s", trajectory_csv)
        logger.info("Saved fit summary JSON: %s", summary_json)
        logger.info("Saved fit summary CSV: %s", summary_csv)
        logger.info("Saved NPZ: %s", data_npz)
        logger.info("Saved trajectory plot: %s", plot_traj)
        logger.info("Saved fit scatter plot: %s", plot_fit)
        logger.info("Saved metadata JSON: %s", metadata_json)

    finally:
        logger.info("Disconnecting hand...")
        hand.disconnect()


if __name__ == "__main__":
    main()
