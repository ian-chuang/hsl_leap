import argparse
import csv
import logging
import time
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from hsl_leap.leap_hand import LeapHand, LeapHandConfig
from hsl_leap.motors.dynamixel import OperatingMode

logger = logging.getLogger(__name__)

# Constants
DXL_TICKS_PER_REV = 4096.0
RAD_PER_TICK = 2.0 * np.pi / DXL_TICKS_PER_REV
RPM_PER_VELOCITY_LSB = 0.229
RAD_S_PER_VELOCITY_LSB = RPM_PER_VELOCITY_LSB * 2.0 * np.pi / 60.0
AMP_PER_CURRENT_LSB = 0.001
YELLOW = "\033[93m"
RESET = "\033[0m"


def _warn_near_current_limit(motor: str, present_current_lsb: float, current_limit_lsb: int, t_s: float) -> None:
    threshold = 0.9 * float(current_limit_lsb)
    abs_current = abs(float(present_current_lsb))
    if abs_current <= threshold:
        return

    logger.warning(
        "%s" + "!" * 68 + "\n"
        "! CURRENT NEAR LIMIT on motor=%s at t=%.3fs: |Present_Current|=%.1f LSB, threshold=%.1f LSB (90%% of limit=%d) !\n"
        + "!" * 68
        + "%s",
        YELLOW,
        motor,
        t_s,
        abs_current,
        threshold,
        current_limit_lsb,
        RESET,
    )


def _fit_through_origin(x: np.ndarray, y: np.ndarray) -> float:
    denom = float(np.dot(x, x))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(x, y) / denom)


def _fit_affine_2d(x1: np.ndarray, x2: np.ndarray, y: np.ndarray) -> tuple[float, float, float, np.ndarray]:
    """Fit y = a*x1 + b*x2 + c using least squares."""
    if not (x1.size == x2.size == y.size):
        raise ValueError("Input arrays must have the same length")
    if y.size == 0:
        return 0.0, 0.0, 0.0, np.asarray([])

    X = np.column_stack([x1, x2, np.ones_like(y)])
    coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_pred = X @ coeffs
    return float(coeffs[0]), float(coeffs[1]), float(coeffs[2]), y_pred


def _rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x))))


def _fit_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if x.size == 0:
        return 0.0, 0.0
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    x_centered = x - x_mean
    denom = float(np.dot(x_centered, x_centered))
    if denom < 1e-12:
        return 0.0, y_mean
    slope = float(np.dot(x_centered, y - y_mean) / denom)
    intercept = y_mean - slope * x_mean
    return slope, intercept


def _set_single_motor_gains(hand: LeapHand, motor: str, kp: int, ki: int, kd: int, curr_lim: int) -> None:
    logger.info("Setting gains for %s: kp=%d ki=%d kd=%d current_limit=%d", motor, kp, ki, kd, curr_lim)
    hand.bus.disable_torque(motor)
    hand.bus.write("Operating_Mode", motor, OperatingMode.CURRENT_POSITION.value)
    hand.bus.write("Position_P_Gain", motor, kp)
    hand.bus.write("Position_I_Gain", motor, ki)
    hand.bus.write("Position_D_Gain", motor, kd)
    hand.bus.write("Current_Limit", motor, curr_lim)
    hand.bus.enable_torque(motor)


def _sample_register(hand: LeapHand, data_name: str, motor: str) -> float:
    vals = hand.bus.sync_read(data_name, motors=[motor], normalize=False, num_retry=hand.config.read_num_retries)
    return float(vals[motor])


def _move_motor_to_position(hand: LeapHand, motor: str, settle_s: float = 0.75) -> None:
    current_obs = hand.get_observation()
    joint_key = f"{motor}.pos"
    if joint_key not in current_obs:
        raise KeyError(f"Joint '{joint_key}' not found in current observation.")

    initial_pos = float(current_obs[joint_key])
    logger.info("Moving %s to initial position %.3f before starting", motor, initial_pos)
    hand.move({joint_key: 0}, duration=1.0, scaled=True)
    if settle_s > 0:
        time.sleep(settle_s)


def run_kp_square_test(
    hand: LeapHand,
    motor: str,
    kp_test: int,
    current_limit: int,
    sample_hz: float,
    record_s: float,
    period_s: float,
) -> Dict[str, np.ndarray]:
    """Set P gain, zero I/D, drive `Position_Trajectory` with a square wave in [-1,1].

    Records `Position_Trajectory`, `Present_Position`, and `Present_Current` and fits
    current_lsb = Kp_lsb_per_tick * (position_trajectory_tick - present_position_tick)
    using a fit through the origin.
    """
    _set_single_motor_gains(hand, motor, kp=kp_test, ki=0, kd=0, curr_lim=current_limit)

    cal = hand.calibration[motor]
    rng_min = int(cal.range_min)
    rng_max = int(cal.range_max)

    dt = 1.0 / sample_hz
    n = max(1, int(round(record_s * sample_hz)))

    t0 = time.monotonic()
    times: List[float] = []
    traj_ticks: List[float] = []
    present_ticks: List[float] = []
    present_cur: List[float] = []
    warned_near_limit = False

    for i in range(n):
        start = time.monotonic()
        t = start - t0
        phase = (t % period_s) / period_s
        action = 0.2 if phase < 0.5 else 0.0

        # map [-1,1] to ticks [rng_min, rng_max]
        desired = int(round(((action + 1.0) / 2.0) * (rng_max - rng_min) + rng_min))

        # Write Goal_Position to actually move the joint; Position_Trajectory is often read-only.
        hand.bus.sync_write("Goal_Position", {motor: desired}, normalize=False)

        # sample Position_Trajectory (if available), Present_Position and Present_Current
        traj = _sample_register(hand, "Position_Trajectory", motor)
        pres = _sample_register(hand, "Present_Position", motor)
        cur = _sample_register(hand, "Present_Current", motor)

        if not warned_near_limit and abs(cur) > 0.9 * current_limit:
            _warn_near_current_limit(motor=motor, present_current_lsb=cur, current_limit_lsb=current_limit, t_s=t)
            warned_near_limit = True

        times.append(t)
        traj_ticks.append(float(traj))
        present_ticks.append(float(pres))
        present_cur.append(float(cur))

        sleep = dt - (time.monotonic() - start)
        if sleep > 0:
            time.sleep(sleep)

    times_np = np.asarray(times)
    traj_np = np.asarray(traj_ticks)
    pres_np = np.asarray(present_ticks)
    cur_np = np.asarray(present_cur)

    error_tick = traj_np - pres_np

    # Filter out tiny errors to avoid noise-dominated fit
    mask = np.abs(error_tick) > 1.0
    x = error_tick[mask]
    y = cur_np[mask]

    kp_lsb_per_tick = _fit_through_origin(x, y)
    kp_amp_per_rad = kp_lsb_per_tick * AMP_PER_CURRENT_LSB / RAD_PER_TICK

    logger.info("Measured Kp: %.6f [current_lsb/tick], %.6f [A/rad]", kp_lsb_per_tick, kp_amp_per_rad)

    return {
        "time_s": times_np,
        "position_trajectory_tick": traj_np,
        "present_position_tick": pres_np,
        "present_current_lsb": cur_np,
        "position_error_tick": error_tick,
        "kp_lsb_per_tick": np.asarray([kp_lsb_per_tick]),
        "kp_amp_per_rad": np.asarray([kp_amp_per_rad]),
    }


def run_kd_after_kp(
    hand: LeapHand,
    motor: str,
    kp_lsb_per_tick: float,
    kd_test: int,
    current_limit: int,
    sample_hz: float,
    record_s: float,
    period_s: float,
    kd_amp_frac: float,
    kd_warmup_cycles: float,
    kd_auto_target_current_frac: float,
    kd_auto_min_amp_frac: float,
    kd_auto_max_amp_frac: float,
    kd_auto_probe_cycles: float,
    kd_auto_steps: int,
) -> Dict[str, np.ndarray]:
    """Identify D using a centered sine trajectory and a regression on the residual current.

    The fit is:
        current_lsb - (prior_kP * position_error_tick) = kD * velocity_error_lsb + bias

    A small sine motion around the current position keeps the joint near its operating point
    while giving a clean phase-separated position/velocity excitation.
    """
    # Note: kp_lsb_per_tick is kept as a diagnostic reference from the prior KP run.
    hand.bus.disable_torque(motor)
    hand.bus.write("Operating_Mode", motor, OperatingMode.CURRENT_POSITION.value)
    hand.bus.enable_torque(motor)

    cal = hand.calibration[motor]
    rng_min = int(cal.range_min)
    rng_max = int(cal.range_max)
    span_ticks = max(1, rng_max - rng_min)

    dt = 1.0 / sample_hz
    n = max(1, int(round(record_s * sample_hz)))
    warmup_s = max(0.0, float(kd_warmup_cycles)) * float(period_s)
    auto_mode = kd_amp_frac <= 0.0

    times: List[float] = []
    traj_ticks: List[float] = []
    traj_vel: List[float] = []
    pres_pos: List[float] = []
    pres_vel: List[float] = []
    pres_cur: List[float] = []
    warned_near_limit = False

    def _run_kd_trial(amp_frac: float, record_s_trial: float) -> Dict[str, np.ndarray]:
        trial_n = max(1, int(round(record_s_trial * sample_hz)))
        trial_center_tick = int(round(_sample_register(hand, "Present_Position", motor)))
        trial_amp_ticks = max(2, int(round(span_ticks * amp_frac)))
        logger.info(
            "KD trial setup: center_tick=%d amp_ticks=%d (amp_frac=%.4f of range=%d, duration=%.2fs)",
            trial_center_tick,
            trial_amp_ticks,
            amp_frac,
            span_ticks,
            record_s_trial,
        )

        trial_times: List[float] = []
        trial_traj_ticks: List[float] = []
        trial_traj_vel: List[float] = []
        trial_pres_pos: List[float] = []
        trial_pres_vel: List[float] = []
        trial_pres_cur: List[float] = []

        trial_t0 = time.monotonic()
        for _ in range(trial_n):
            start = time.monotonic()
            t = start - trial_t0
            desired = int(round(trial_center_tick + trial_amp_ticks * np.sin(2.0 * np.pi * t / period_s)))
            desired = int(np.clip(desired, rng_min, rng_max))
            hand.bus.sync_write("Goal_Position", {motor: desired}, normalize=False)

            traj = _sample_register(hand, "Position_Trajectory", motor)
            vtraj = _sample_register(hand, "Velocity_Trajectory", motor)
            pres = _sample_register(hand, "Present_Position", motor)
            vel = _sample_register(hand, "Present_Velocity", motor)
            cur = _sample_register(hand, "Present_Current", motor)

            trial_times.append(t)
            trial_traj_ticks.append(float(traj))
            trial_traj_vel.append(float(vtraj))
            trial_pres_pos.append(float(pres))
            trial_pres_vel.append(float(vel))
            trial_pres_cur.append(float(cur))

            if abs(cur) > 0.9 * current_limit:
                _warn_near_current_limit(motor=motor, present_current_lsb=cur, current_limit_lsb=current_limit, t_s=t)

            sleep = dt - (time.monotonic() - start)
            if sleep > 0:
                time.sleep(sleep)

        return {
            "time_s": np.asarray(trial_times),
            "position_trajectory_tick": np.asarray(trial_traj_ticks),
            "velocity_trajectory_lsb": np.asarray(trial_traj_vel),
            "present_position_tick": np.asarray(trial_pres_pos),
            "present_velocity_lsb": np.asarray(trial_pres_vel),
            "present_current_lsb": np.asarray(trial_pres_cur),
            "center_tick": np.asarray([trial_center_tick]),
            "amp_ticks": np.asarray([trial_amp_ticks]),
            "amp_frac": np.asarray([amp_frac]),
        }

    selected_amp_frac = float(kd_amp_frac)
    probe_peak_current_lsb = np.asarray([np.nan])
    probe_peak_ratio = np.asarray([np.nan])
    auto_target_current_lsb = float(kd_auto_target_current_frac) * float(current_limit)

    if auto_mode:
        logger.info(
            "KD auto mode enabled: target=%.1f LSB (%.2f of current limit), amp range=[%.4f, %.4f]",
            auto_target_current_lsb,
            kd_auto_target_current_frac,
            kd_auto_min_amp_frac,
            kd_auto_max_amp_frac,
        )

        selected_amp_frac = max(1e-4, float(kd_auto_min_amp_frac))
        final_probe: Dict[str, np.ndarray] | None = None
        for step in range(max(1, int(kd_auto_steps))):
            probe = _run_kd_trial(selected_amp_frac, max(period_s * float(kd_auto_probe_cycles), 2.0 * dt))
            final_probe = probe
            peak_current = float(np.max(np.abs(probe["present_current_lsb"]))) if probe["present_current_lsb"].size else 0.0
            probe_peak_current_lsb = np.asarray([peak_current])
            probe_peak_ratio = np.asarray([peak_current / float(current_limit) if current_limit else 0.0])
            logger.info(
                "KD auto probe %d/%d: amp_frac=%.5f amp_ticks=%d peak_current=%.1f LSB (%.1f%% of limit)",
                step + 1,
                int(kd_auto_steps),
                selected_amp_frac,
                int(probe["amp_ticks"][0]),
                peak_current,
                100.0 * probe_peak_ratio[0],
            )

            if peak_current >= auto_target_current_lsb:
                break

            if peak_current <= 1e-6:
                next_amp_frac = min(float(kd_auto_max_amp_frac), selected_amp_frac * 2.0)
            else:
                scale = auto_target_current_lsb / peak_current
                next_amp_frac = min(float(kd_auto_max_amp_frac), selected_amp_frac * min(max(scale, 1.25), 2.0))

            if next_amp_frac <= selected_amp_frac + 1e-6:
                break

            selected_amp_frac = next_amp_frac

        if final_probe is not None and probe_peak_current_lsb.size:
            logger.info(
                "KD auto selected amp_frac=%.5f after probe peak %.1f LSB (target %.1f LSB)",
                selected_amp_frac,
                float(probe_peak_current_lsb[0]),
                auto_target_current_lsb,
            )
    else:
        logger.info("KD manual mode: using amp_frac=%.5f", selected_amp_frac)

    final = _run_kd_trial(selected_amp_frac, record_s)

    times_np = final["time_s"]
    traj_np = final["position_trajectory_tick"]
    traj_vel_np = final["velocity_trajectory_lsb"]
    pres_pos_np = final["present_position_tick"]
    pres_vel_np = final["present_velocity_lsb"]
    pres_cur_np = final["present_current_lsb"]

    pos_error = traj_np - pres_pos_np

    vel_error_raw = traj_vel_np - pres_vel_np

    # Mask out the initial transient and any near-limit points that would bias the linear fit.
    fit_mask = np.ones_like(pres_cur_np, dtype=bool)
    fit_mask &= times_np >= warmup_s
    fit_mask &= np.abs(pres_cur_np) <= 0.9 * float(current_limit)

    if int(np.count_nonzero(fit_mask)) < 10:
        logger.warning(
            "KD fit has too few samples after masking (n=%d); falling back to all samples.",
            int(np.count_nonzero(fit_mask)),
        )
        fit_mask = np.ones_like(pres_cur_np, dtype=bool)

    pos_fit = pos_error[fit_mask]
    cur_fit = pres_cur_np[fit_mask]
    residual_fit = cur_fit - kp_lsb_per_tick * pos_fit

    fit_candidates = []
    for label, vel_err in (("as_is", vel_error_raw), ("flipped", -vel_error_raw)):
        vel_fit = vel_err[fit_mask]
        kd_fit, bias_fit = _fit_line(vel_fit, residual_fit)
        pred_residual_fit = kd_fit * vel_fit + bias_fit
        pred_fit = kp_lsb_per_tick * pos_fit + pred_residual_fit
        residual_fit_model = cur_fit - pred_fit
        fit_candidates.append(
            {
                "label": label,
                "kd": kd_fit,
                "bias": bias_fit,
                "pred": pred_fit,
                "rms": _rms(residual_fit_model),
                "vel_err": vel_err,
                "residual": residual_fit_model,
            }
        )

    fit_candidates.sort(key=lambda d: d["rms"])
    best = fit_candidates[0]
    if len(fit_candidates) > 1:
        alt = fit_candidates[1]
        if best["kd"] < 0 <= alt["kd"] and alt["rms"] <= 1.05 * best["rms"]:
            best = alt

    kd_vel_gain_lsb_per_vel = best["kd"]
    kd_bias_lsb = best["bias"]
    vel_error_lsb = best["vel_err"]
    predicted_current_lsb = kp_lsb_per_tick * pos_error + kd_vel_gain_lsb_per_vel * vel_error_lsb + kd_bias_lsb
    residual_cur = pres_cur_np - predicted_current_lsb

    kd_amp_s_per_rad = kd_vel_gain_lsb_per_vel * AMP_PER_CURRENT_LSB / RAD_S_PER_VELOCITY_LSB
    fit_rms_lsb = best["rms"]

    logger.info(
        "KD fit selected=%s, amp_frac=%.5f, using prior kP=%.6f [current_lsb/tick] (%.6f A/rad), kD=%.6f [current_lsb/vel_lsb] (%.6f A/(rad/s)), bias=%.3f [current_lsb], RMS=%.3f [current_lsb]",
        best["label"],
        selected_amp_frac,
        kp_lsb_per_tick,
        kp_lsb_per_tick * AMP_PER_CURRENT_LSB / RAD_PER_TICK,
        kd_vel_gain_lsb_per_vel,
        kd_amp_s_per_rad,
        kd_bias_lsb,
        fit_rms_lsb,
    )
    logger.info("Reference prior Kp from the KP run: %.6f [current_lsb/tick]", kp_lsb_per_tick)

    return {
        "time_s": times_np,
        "position_trajectory_tick": traj_np,
        "velocity_trajectory_lsb": traj_vel_np,
        "present_position_tick": pres_pos_np,
        "present_velocity_lsb": pres_vel_np,
        "present_current_lsb": pres_cur_np,
        "position_error_tick": pos_error,
        "predicted_current_lsb": predicted_current_lsb,
        "residual_current_lsb": residual_cur,
        "velocity_error_lsb": vel_error_lsb,
        "kd_kp_lsb_per_tick": np.asarray([kp_lsb_per_tick]),
        "kd_kp_amp_per_rad": np.asarray([kp_lsb_per_tick * AMP_PER_CURRENT_LSB / RAD_PER_TICK]),
        "kd_lsb_per_vel": np.asarray([kd_vel_gain_lsb_per_vel]),
        "kd_amp_s_per_rad": np.asarray([kd_amp_s_per_rad]),
        "kd_current_bias_lsb": np.asarray([kd_bias_lsb]),
        "kd_fit_rms_lsb": np.asarray([fit_rms_lsb]),
        "kd_fit_label": np.asarray([0.0 if best["label"] == "as_is" else 1.0]),
        "kd_warmup_s": np.asarray([warmup_s]),
        "kd_selected_amp_frac": np.asarray([selected_amp_frac]),
        "kd_probe_peak_current_lsb": probe_peak_current_lsb,
        "kd_probe_peak_ratio": probe_peak_ratio,
        "kd_auto_mode": np.asarray([1.0 if auto_mode else 0.0]),
    }


def _save_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _plot_kp(data: Dict[str, np.ndarray], out_png: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)

    t = data["time_s"]
    traj = data["position_trajectory_tick"]
    pres = data["present_position_tick"]
    cur = data["present_current_lsb"]
    err = data["position_error_tick"]

    slope = float(data["kp_lsb_per_tick"][0])

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(t, traj, label="position_trajectory_tick")
    ax[0].plot(t, pres, label="present_position_tick")
    ax[0].plot(t, cur, label="present_current_lsb")
    ax[0].set_xlabel("time [s]")
    ax[0].legend(fontsize="small")
    ax[0].grid(True)

    ax[1].scatter(err, cur, s=6, alpha=0.7)
    if err.size:
        xline = np.linspace(float(np.min(err)), float(np.max(err)), 200)
        yline = slope * xline
        ax[1].plot(xline, yline, "r", label=f"fit: y={slope:.4f}x")
        ax[1].legend(fontsize="small")
    ax[1].set_xlabel("position_error [tick]")
    ax[1].set_ylabel("present_current [LSB]")
    ax[1].grid(True)

    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def _plot_kd(data: Dict[str, np.ndarray], out_png: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)

    t = data["time_s"]
    vel = data.get("present_velocity_lsb", np.zeros_like(t))
    vel_traj = data.get("velocity_trajectory_lsb", np.zeros_like(t))
    cur = data["present_current_lsb"]
    pred = data.get("predicted_current_lsb", cur * 0)
    resid = data.get("residual_current_lsb", cur - pred)
    vel_err = data.get("velocity_error_lsb", vel * 0)
    pos_err = data.get("position_error_tick", np.zeros_like(t))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(t, vel_traj, label="velocity_trajectory_lsb")
    ax[0].plot(t, vel, label="present_velocity_lsb")
    ax[0].plot(t, cur, label="present_current_lsb")
    ax[0].plot(t, pred, label="predicted_current_lsb")
    ax[0].set_xlabel("time [s]")
    ax[0].legend(fontsize="small")
    ax[0].grid(True)

    ax[1].scatter(cur, pred, s=6, alpha=0.7)
    if cur.size:
        lo = float(min(np.min(cur), np.min(pred)))
        hi = float(max(np.max(cur), np.max(pred)))
        ax[1].plot([lo, hi], [lo, hi], "r", label="ideal: y=x")
        ax[1].legend(fontsize="small")
    ax[1].set_xlabel("measured_current [LSB]")
    ax[1].set_ylabel("predicted_current [LSB]")
    ax[1].grid(True)

    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def _save_summary(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "value", "unit", "note"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main() -> None:
    parser = argparse.ArgumentParser(description="KP square-wave and KD sine-wave measurement")
    parser.add_argument("--port", default="/dev/ttyDXL_leap_hand")
    parser.add_argument("--baudrate", type=int, default=4_000_000)
    parser.add_argument("--motor", default="if_pip")
    parser.add_argument("--sample_hz", type=float, default=50.0)
    parser.add_argument("--record_s", type=float, default=6.0)
    parser.add_argument("--period_s", type=float, default=1.0)
    parser.add_argument("--current_limit", type=int, default=550)
    parser.add_argument("--kp_test", type=int, default=600)
    parser.add_argument("--kd_test", type=int, default=200)
    parser.add_argument("--kd_amp_frac", type=float, default=0.0)
    parser.add_argument("--kd_warmup_cycles", type=float, default=1.0)
    parser.add_argument("--kd_auto_target_current_frac", type=float, default=0.9)
    parser.add_argument("--kd_auto_min_amp_frac", type=float, default=0.005)
    parser.add_argument("--kd_auto_max_amp_frac", type=float, default=0.2)
    parser.add_argument("--kd_auto_probe_cycles", type=float, default=1.5)
    parser.add_argument("--kd_auto_steps", type=int, default=4)
    parser.add_argument("--out_dir", default="sysid/outputs/pd_gains")
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")

    cfg = LeapHandConfig(port=args.port, baudrate=args.baudrate)
    hand = LeapHand(cfg)

    out_dir = Path(args.out_dir)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    stem = f"kp_square_{args.motor}_{stamp}"

    logger.info("Connecting to hand...")
    hand.connect()
    try:
        if args.motor not in hand.bus.motors:
            raise KeyError(f"Unknown motor '{args.motor}'. Available: {list(hand.bus.motors.keys())}")

        _move_motor_to_position(hand, args.motor, settle_s=max(0.75, 1.5 / max(args.sample_hz, 1.0)))

        data = run_kp_square_test(
            hand=hand,
            motor=args.motor,
            kp_test=args.kp_test,
            current_limit=args.current_limit,
            sample_hz=args.sample_hz,
            record_s=args.record_s,
            period_s=args.period_s,
        )

        csv_path = out_dir / f"{stem}.csv"
        png_path = out_dir / f"{stem}.png"
        summary_path = out_dir / f"{stem}_summary.csv"

        rows = []
        for i in range(data["time_s"].size):
            rows.append(
                {
                    "time_s": float(data["time_s"][i]),
                    "position_trajectory_tick": float(data["position_trajectory_tick"][i]),
                    "present_position_tick": float(data["present_position_tick"][i]),
                    "position_error_tick": float(data["position_error_tick"][i]),
                    "present_current_lsb": float(data["present_current_lsb"][i]),
                }
            )

        _save_csv(csv_path, rows, ["time_s", "position_trajectory_tick", "present_position_tick", "position_error_tick", "present_current_lsb"])
        _plot_kp(data, png_path)

        _save_summary(
            summary_path,
            [
                {"name": "kp_lsb_per_tick", "value": float(data["kp_lsb_per_tick"][0]), "unit": "current_lsb/tick", "note": "fit through origin"},
                {"name": "kp_amp_per_rad", "value": float(data["kp_amp_per_rad"][0]), "unit": "A/rad", "note": "derived"},
            ],
        )

        logger.info("Saved CSV: %s", csv_path)
        logger.info("Saved summary: %s", summary_path)
        logger.info("Saved plot: %s", png_path)

        # --- KD measurement using the measured P contribution ---
        kp_lsb_measured = float(data["kp_lsb_per_tick"][0])
        logger.info("Running KD measurement with Position_D_Gain=%d (register) and using measured P mapping %.6f", args.kd_test, kp_lsb_measured)

        _move_motor_to_position(hand, args.motor, settle_s=max(0.75, 1.5 / max(args.sample_hz, 1.0)))

        # set register-level gains: P stays at args.kp_test, D set to args.kd_test
        _set_single_motor_gains(hand, args.motor, kp=args.kp_test, ki=0, kd=args.kd_test, curr_lim=args.current_limit)

        kd_data = run_kd_after_kp(
            hand=hand,
            motor=args.motor,
            kp_lsb_per_tick=kp_lsb_measured,
            kd_test=args.kd_test,
            current_limit=args.current_limit,
            sample_hz=args.sample_hz,
            record_s=args.record_s,
            period_s=args.period_s,
            kd_amp_frac=args.kd_amp_frac,
            kd_warmup_cycles=args.kd_warmup_cycles,
            kd_auto_target_current_frac=args.kd_auto_target_current_frac,
            kd_auto_min_amp_frac=args.kd_auto_min_amp_frac,
            kd_auto_max_amp_frac=args.kd_auto_max_amp_frac,
            kd_auto_probe_cycles=args.kd_auto_probe_cycles,
            kd_auto_steps=args.kd_auto_steps,
        )

        stem_kd = f"kd_sine_{args.motor}_{stamp}"
        csv_kd = out_dir / f"{stem_kd}.csv"
        png_kd = out_dir / f"{stem_kd}.png"
        summary_kd = out_dir / f"{stem_kd}_summary.csv"

        rows_kd = []
        for i in range(kd_data["time_s"].size):
            rows_kd.append(
                {
                    "time_s": float(kd_data["time_s"][i]),
                    "position_trajectory_tick": float(kd_data["position_trajectory_tick"][i]),
                    "present_position_tick": float(kd_data["present_position_tick"][i]),
                    "velocity_trajectory_lsb": float(kd_data["velocity_trajectory_lsb"][i]),
                    "present_velocity_lsb": float(kd_data["present_velocity_lsb"][i]),
                    "position_error_tick": float(kd_data["position_error_tick"][i]),
                    "present_current_lsb": float(kd_data["present_current_lsb"][i]),
                    "predicted_current_lsb": float(kd_data["predicted_current_lsb"][i]),
                    "residual_current_lsb": float(kd_data["residual_current_lsb"][i]),
                    "velocity_error_lsb": float(kd_data["velocity_error_lsb"][i]),
                }
            )

        _save_csv(
            csv_kd,
            rows_kd,
            [
                "time_s",
                "position_trajectory_tick",
                "present_position_tick",
                "velocity_trajectory_lsb",
                "present_velocity_lsb",
                "position_error_tick",
                "present_current_lsb",
                "predicted_current_lsb",
                "residual_current_lsb",
                "velocity_error_lsb",
            ],
        )

        _plot_kd(kd_data, png_kd)

        _save_summary(
            summary_kd,
            [
                {"name": "kd_kp_lsb_per_tick", "value": float(kd_data["kd_kp_lsb_per_tick"][0]), "unit": "current_lsb/tick", "note": "fixed prior Kp estimate"},
                {"name": "kd_kp_amp_per_rad", "value": float(kd_data["kd_kp_amp_per_rad"][0]), "unit": "A/rad", "note": "derived from fixed prior Kp"},
                {"name": "kd_lsb_per_vel", "value": float(kd_data["kd_lsb_per_vel"][0]), "unit": "current_lsb/velocity_lsb", "note": "line fit with intercept"},
                {"name": "kd_current_bias_lsb", "value": float(kd_data["kd_current_bias_lsb"][0]), "unit": "current_lsb", "note": "line-fit intercept"},
                {"name": "kd_amp_s_per_rad", "value": float(kd_data["kd_amp_s_per_rad"][0]), "unit": "A/(rad/s)", "note": "derived"},
                {"name": "kd_fit_rms_lsb", "value": float(kd_data["kd_fit_rms_lsb"][0]), "unit": "current_lsb", "note": "fit residual RMS"},
                {"name": "kd_fit_label", "value": float(kd_data["kd_fit_label"][0]), "unit": "0/1", "note": "0=as_is velocity sign, 1=flipped"},
                {"name": "kd_warmup_s", "value": float(kd_data["kd_warmup_s"][0]), "unit": "s", "note": "fit excludes early transient"},
                {"name": "kd_selected_amp_frac", "value": float(kd_data["kd_selected_amp_frac"][0]), "unit": "fraction_of_range", "note": "selected excitation amplitude"},
                {"name": "kd_probe_peak_current_lsb", "value": float(kd_data["kd_probe_peak_current_lsb"][0]), "unit": "current_lsb", "note": "peak current during auto probe"},
                {"name": "kd_probe_peak_ratio", "value": float(kd_data["kd_probe_peak_ratio"][0]), "unit": "fraction_of_current_limit", "note": "peak current / current limit"},
                {"name": "kd_auto_mode", "value": float(kd_data["kd_auto_mode"][0]), "unit": "0/1", "note": "1 means auto amplitude search was used"},
            ],
        )

        logger.info("Saved KD CSV: %s", csv_kd)
        logger.info("Saved KD summary: %s", summary_kd)
        logger.info("Saved KD plot: %s", png_kd)

    finally:
        logger.info("Disconnecting...")
        hand.disconnect()


if __name__ == "__main__":
    main()
