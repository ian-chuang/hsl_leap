import argparse
import time
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from hsl_leap.leap_hand import LeapHand, LeapHandConfig, MJ_ZERO_POSITION


def record_velocity_for_duration(hand: LeapHand, motor: str, duration: float, sample_hz: float = 100.0) -> Dict[str, List[float]]:
    dt = 1.0 / sample_hz
    end_time = time.time() + duration
    times: List[float] = []
    velocities: List[float] = []

    while time.time() < end_time:
        t = time.time()
        # read present velocity (use normalized values when available)
        # try:
        #     vel_dict = hand.bus.sync_read("Present_Velocity", normalize=True, num_retry=hand.config.read_num_retries)
        # except Exception:
        #     # fallback single read
        vel_dict = hand.bus.sync_read(
            "Present_Velocity", 
            motors=[motor],
            normalize=False, 
            num_retry=hand.config.read_num_retries
        )
        v = vel_dict.get(motor) * 0.229 * 0.10472
        # x0.229 converts to rev/min
        # x0.10472 converts to rad/s


        velocities.append(float(v) if v is not None else 0.0)
        times.append(t)

        # sleep until next sample
        sleep_time = dt - (time.time() - t)
        if sleep_time > 0:
            time.sleep(sleep_time)

    # make times relative
    t0 = times[0] if times else time.time()
    times = [tt - t0 for tt in times]
    return {"time": times, "velocity": velocities}


def main():
    parser = argparse.ArgumentParser(description="Run a single Leap Hand joint between min and max limits and record velocity.")
    parser.add_argument("--port", default="/dev/ttyDXL_leap_hand", help="Serial port for Dynamixel bus (e.g. /dev/ttyDXL_leap_hand)")
    parser.add_argument("--motor", default="if_dip", help="Motor name (e.g. if_mcp, joint_0, th_mcp)")
    parser.add_argument("--baudrate", type=int, default=4_000_000)
    parser.add_argument("--duration", type=float, default=2.0, help="Seconds to record each half-sweep")
    parser.add_argument("--cycles", type=int, default=2, help="Number of back-and-forth cycles")
    parser.add_argument("--sample_hz", type=float, default=200.0, help="Sampling frequency for velocity reading")
    parser.add_argument("--plot_out", default=None, help="Optional output filename for the plot (png). If omitted will save get_max_vel_<motor>.png")

    args = parser.parse_args()

    cfg = LeapHandConfig(port=args.port, baudrate=args.baudrate)
    hand = LeapHand(cfg)

    try:
        print("Connecting to hand...")
        hand.connect()

        # move the whole hand to zero for start
        print("Moving to zero position...")
        zero_action = MJ_ZERO_POSITION
        hand.move(zero_action, duration=3.0)

        motor_key = args.motor
        # confirm calibration exists
        if motor_key not in hand.calibration:
            raise KeyError(f"Motor '{motor_key}' not found in calibration keys: {list(hand.calibration.keys())}")

        cal = hand.calibration[motor_key]
        range_min = int(getattr(cal, "range_min"))
        range_max = int(getattr(cal, "range_max"))

        print(f"Motor {motor_key} range: min={range_min}, max={range_max}")

        all_records = []

        for cycle in range(args.cycles):
            print(f"Cycle {cycle + 1}/{args.cycles}: moving to min limit and recording...")
            # command min limit (raw encoder units) as fast as possible
            hand.bus.sync_write("Goal_Position", {motor_key: range_min}, normalize=False)
            rec_min = record_velocity_for_duration(hand, motor_key, args.duration, sample_hz=args.sample_hz)
            all_records.append((f"min_{cycle}", rec_min))

            print(f"Cycle {cycle + 1}/{args.cycles}: moving to max limit and recording...")
            hand.bus.sync_write("Goal_Position", {motor_key: range_max}, normalize=False)
            rec_max = record_velocity_for_duration(hand, motor_key, args.duration, sample_hz=args.sample_hz)
            all_records.append((f"max_{cycle}", rec_max))

        # build a single concatenated plot
        fig, ax = plt.subplots(figsize=(10, 4))

        for label, rec in all_records:
            times = np.array(rec["time"]) + (0 if label.endswith("_0") else 0)
            ax.plot(rec["time"], rec["velocity"], label=label)

        ax.set_title(f"Velocity recordings for motor {motor_key}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Velocity")
        ax.legend(loc="upper right", fontsize="small")
        ax.grid(True)

        out_file = args.plot_out or f"get_max_vel_{motor_key}.png"
        fig.tight_layout()
        fig.savefig(out_file)
        print(f"Saved plot to {out_file}")

    finally:
        try:
            print("Disconnecting...")
            hand.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()
