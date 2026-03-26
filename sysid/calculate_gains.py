#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import math

import numpy as np
import matplotlib.pyplot as plt

# Same conversions as collect_trajectory
DXL_TICKS_PER_REV = 4096.0
RAD_PER_TICK = 2.0 * math.pi / DXL_TICKS_PER_REV
RPM_PER_VELOCITY_LSB = 0.229
RAD_S_PER_VELOCITY_LSB = RPM_PER_VELOCITY_LSB * 2.0 * math.pi / 60.0
MA_PER_CURRENT_LSB = 1.0
TORQUE_NM_PER_AMP = 0.515


def _safe_load_npz(p: Path):
    arr = np.load(p)
    return arr


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2)))


def main():
    parser = argparse.ArgumentParser(description="Estimate PD gains from a collected trajectory run")
    parser.add_argument("--run_dir", required=True, help="Path to trajectory run folder (contains trajectory.npz)")
    parser.add_argument("--out_dir", default=None, help="Directory to save results (defaults to run_dir)")
    parser.add_argument("--J", type=float, default=0.001, help="Armature inertia [kg*m^2]")
    parser.add_argument("--friction_static", type=float, default=0.05, help="Coulomb/static friction [Nm]")
    parser.add_argument("--friction_dynamic", type=float, default=0.04, help="Dynamic velocity-dependent friction [Nm]")
    parser.add_argument("--viscous", type=float, default=0.0025, help="Viscous friction [Nm/(rad/s)]")
    parser.add_argument("--vel_limit", type=float, default=8.0, help="Velocity scaling limit used in friction shaping [rad/s]")
    parser.add_argument("--eff_limit", type=float, default=0.3, help="(unused) effective command limit placeholder")
    parser.add_argument("--no-scipy-refine", dest="scipy_refine", action="store_false", help="Disable scipy refinement if scipy is available")
    parser.set_defaults(scipy_refine=True)
    parser.add_argument("--allow-negative-gains", dest="allow_negative", action="store_true", help="Allow negative Kp/Kd during refinement (default: false, enforce non-negative)")
    parser.add_argument("--ridge", type=float, default=0.0, help="Ridge regularization lambda for initial LS fit")

    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    npz_path = run_dir / "trajectory.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"trajectory.npz not found in {run_dir}")

    data = _safe_load_npz(npz_path)

    # Required keys written by collect_trajectory.py
    t = data["time_s"]
    q = data["position_rad"]
    qdot = data["velocity_rad_s"]
    torque_meas = data.get("torque_nm")

    # Position trajectory target (command) - fallback to zeros if not present
    if "position_trajectory_rad" in data:
        q_cmd = data["position_trajectory_rad"]
    else:
        logging.warning("position_trajectory_rad not found in NPZ; using measured position as command (degenerate)")
        q_cmd = q.copy()

    # If torque wasn't saved directly, try current->torque conversion
    if torque_meas is None:
        if "current_mA" in data:
            cur_mA = data["current_mA"]
            torque_meas = (cur_mA / 1000.0) * TORQUE_NM_PER_AMP
        else:
            raise KeyError("trajectory.npz has neither 'torque_nm' nor 'current_mA'")

    # compute dt and qdd (numerical derivative)
    dt = np.gradient(t)
    qdd = np.gradient(qdot, t)

    # compute commanded velocity by differentiating the commanded position trajectory
    qdot_cmd = np.gradient(q_cmd, t)

    # dynamics terms (known model parameters provided by user)
    J = float(args.J)
    viscous = float(args.viscous)
    friction_static = float(args.friction_static)
    friction_dynamic = float(args.friction_dynamic)
    vel_limit = float(args.vel_limit)

    # friction shaping: combine static coulomb + a smooth dynamic term
    # friction = Fc*sign(qdot) + Fdyn * tanh(qdot/vel_limit)
    sign_qdot = np.sign(qdot)
    smooth_dyn = np.tanh(qdot / max(1e-6, vel_limit))
    friction_term = friction_static * sign_qdot + friction_dynamic * smooth_dyn

    viscous_term = viscous * qdot
    inertial_term = J * qdd

    dynamics = inertial_term + viscous_term + friction_term

    # residual torque that should be supplied by the low-level PD controller
    residual = torque_meas - dynamics

    # form errors
    pos_err = q_cmd - q
    vel_err = qdot_cmd - qdot

    # Stack regressors and solve linear least-squares: residual = Kp*pos_err + Kd*vel_err
    A = np.vstack([pos_err, vel_err]).T

    # remove any rows with NaN or inf
    valid = np.isfinite(A).all(axis=1) & np.isfinite(residual)
    A_v = A[valid]
    r_v = residual[valid]

    if A_v.shape[0] < 10:
        raise RuntimeError("Not enough valid samples to fit gains")

    # Solve with numpy lstsq (optionally with ridge regularization)
    ridge = float(args.ridge)
    if ridge and ridge > 0.0:
        ATA = A_v.T.dot(A_v)
        ATb = A_v.T.dot(r_v)
        reg = ridge * np.eye(ATA.shape[0])
        x = np.linalg.solve(ATA + reg, ATb)
    else:
        x, *_ = np.linalg.lstsq(A_v, r_v, rcond=None)
    Kp_ls, Kd_ls = float(x[0]), float(x[1])

    Kp_refined = Kp_ls
    Kd_refined = Kd_ls

    # optionally refine with scipy least_squares to enforce positivity bounds
    if args.scipy_refine:
        try:
            from scipy.optimize import least_squares

            def res_fun(k):
                kp, kd = k
                pred = A_v.dot(np.array([kp, kd]))
                return (pred - r_v)

            # bounds: enforce non-negative by default, unless user allows negatives
            if args.allow_negative:
                lb = [-1e3, -1e3]
            else:
                lb = [0.0, 0.0]
            ub = [1e6, 1e6]
            res = least_squares(res_fun, x0=np.array([Kp_ls, Kd_ls]), bounds=(lb, ub), xtol=1e-12)
            Kp_refined, Kd_refined = float(res.x[0]), float(res.x[1])
        except Exception as e:
            logging.info("scipy refinement skipped: %s", e)

    # compute modeled torque using refined gains
    modeled_torque = dynamics + (Kp_refined * pos_err) + (Kd_refined * vel_err)

    # performance metrics
    residual_after = torque_meas - modeled_torque
    rmse = _rms(residual_after)
    ss_res = np.sum((torque_meas - modeled_torque) ** 2)
    ss_tot = np.sum((torque_meas - np.mean(torque_meas)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')

    results = {
        "J": J,
        "viscous": viscous,
        "friction_static": friction_static,
        "friction_dynamic": friction_dynamic,
        "vel_limit": vel_limit,
        "Kp_ls": Kp_ls,
        "Kd_ls": Kd_ls,
        "Kp_refined": Kp_refined,
        "Kd_refined": Kd_refined,
        "rmse_Nm": float(rmse),
        "r2": float(r2),
        "n_samples": int(A_v.shape[0]),
    }

    # save JSON summary
    summary_path = out_dir / "gains_summary.json"
    with summary_path.open("w") as f:
        json.dump(results, f, indent=2)

    # save NPZ with arrays for further analysis
    np.savez(out_dir / "gains_fit.npz",
             time_s=t,
             torque_meas=torque_meas,
             dynamics=dynamics,
             modeled_torque=modeled_torque,
             residual=residual,
             residual_after=residual_after,
             pos_err=pos_err,
             vel_err=vel_err,
             valid_mask=valid,
             )

    # plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(t, torque_meas, label="measured torque [Nm]")
    axes[0].plot(t, dynamics, label="dynamics (J*qdd + viscous + friction)")
    axes[0].plot(t, modeled_torque, label="modeled total torque (with PD)")
    axes[0].legend(fontsize="small")
    axes[0].grid(True)

    axes[1].plot(t, residual, label="residual before PD")
    axes[1].plot(t, residual_after, label="residual after PD")
    axes[1].legend(fontsize="small")
    axes[1].grid(True)

    axes[2].plot(t, pos_err, label="pos_err [rad]")
    axes[2].plot(t, vel_err, label="vel_err [rad/s]")
    axes[2].legend(fontsize="small")
    axes[2].grid(True)

    fig.suptitle(f"Estimated gains: Kp={Kp_refined:.6g} Nm/rad, Kd={Kd_refined:.6g} Nm*s/rad | RMSE={rmse:.4g} Nm | R2={r2:.4f}")
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])

    plot_path = out_dir / "gains_plot.png"
    fig.savefig(plot_path)
    plt.close(fig)

    logging.info("Saved summary JSON: %s", summary_path)
    logging.info("Saved diagnostic NPZ: %s", out_dir / "gains_fit.npz")
    logging.info("Saved gains plot: %s", plot_path)
    logging.info("Estimated Kp [Nm/rad]: %g", Kp_refined)
    logging.info("Estimated Kd [Nm*s/rad]: %g", Kd_refined)
    logging.info("RMSE [Nm]: %g", rmse)
    logging.info("R2: %g", r2)


if __name__ == "__main__":
    main()
