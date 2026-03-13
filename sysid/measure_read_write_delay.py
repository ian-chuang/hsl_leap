import argparse
import time
import logging
from typing import List, Dict

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from hsl_leap.leap_hand import LeapHand, LeapHandConfig, MJ_ZERO_POSITION

logger = logging.getLogger(__name__)


def measure_delays(hand: LeapHand, motors: List[str], trials: int, inter_delay: float):
    """Measure read and write latency for the whole hand.

    For each trial we perform one read of all joints via `get_observation()` and
    one write of all joints to zero via `send_action(MJ_ZERO_POSITION)`. The
    function returns two lists: read_times and write_times (seconds).
    """
    all_read_times: List[float] = []
    all_write_times: List[float] = []

    # Use the public hand API to read and write all joints at once.
    for t in range(trials):
        # measure read of all motors
        t0 = time.time()
        _ = hand.get_observation()
        dt_read = time.time() - t0
        all_read_times.append(dt_read)

        if inter_delay > 0:
            time.sleep(inter_delay)

        # measure write (write all zeros at once)
        t0 = time.time()
        hand.send_action(MJ_ZERO_POSITION)
        dt_write = time.time() - t0
        all_write_times.append(dt_write)

        if inter_delay > 0:
            time.sleep(inter_delay)

    return all_read_times, all_write_times


def plot_and_save(read_times: List[float], write_times: List[float], out_file: str):
    """Plot three histograms: read, write, and total (read+write).

    Saves to `out_file` (PNG).
    """
    total_times = [r + w for r, w in zip(read_times, write_times)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist(read_times, bins=50, color="C0", alpha=0.8)
    axes[0].set_title("Read latencies")
    axes[0].set_xlabel("Seconds")

    axes[1].hist(write_times, bins=50, color="C1", alpha=0.8)
    axes[1].set_title("Write latencies")
    axes[1].set_xlabel("Seconds")

    axes[2].hist(total_times, bins=50, color="C2", alpha=0.8)
    axes[2].set_title("Total latencies (read+write)")
    axes[2].set_xlabel("Seconds")

    # Reduce number of x-ticks and use scientific formatting to avoid overlapping labels
    for ax in axes:
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=6))
        fmt = mticker.ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((-4, 4))
        ax.xaxis.set_major_formatter(fmt)
        ax.ticklabel_format(axis="x", style="sci", scilimits=( -4, 4 ))

    fig.suptitle("Dynamixel read/write latency distribution")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_file)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Measure read/write delays to Leap Hand joints.")
    parser.add_argument("--port", default="/dev/ttyDXL_leap_hand", help="Serial port for Dynamixel bus")
    parser.add_argument("--baudrate", type=int, default=4_000_000)
    parser.add_argument("--trials", type=int, default=200, help="Number of measurement trials per joint")
    parser.add_argument("--delay", type=float, default=0.0, help="Inter-operation delay (s) between read/write calls")
    parser.add_argument("--plot_out", default=None, help="Output PNG file for histogram")
    parser.add_argument("--log_out", default=None, help="Optional log filename to save summary")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    cfg = LeapHandConfig(port=args.port, baudrate=args.baudrate)
    hand = LeapHand(cfg)

    try:
        logger.info("Connecting to hand...")
        hand.connect()

        logger.info("Moving hand to zero position...")
        hand.move(MJ_ZERO_POSITION, duration=1.5)

        motors = list(hand.bus.motors.keys())
        logger.info(f"Measuring {len(motors)} motors: {motors}")

        all_read_times, all_write_times = measure_delays(hand, motors, args.trials, args.delay)

        # compute and log averages
        avg_read = float(np.mean(all_read_times)) if all_read_times else 0.0
        avg_write = float(np.mean(all_write_times)) if all_write_times else 0.0

        logger.info(f"Average read time over trials: {avg_read:.6f} s")
        logger.info(f"Average write time over trials: {avg_write:.6f} s")

        out_file = args.plot_out or "measure_read_write_delay.png"
        plot_and_save(all_read_times, all_write_times, out_file)
        logger.info(f"Saved histogram to {out_file}")

        if args.log_out:
            total_times = [r + w for r, w in zip(all_read_times, all_write_times)]
            with open(args.log_out, "w") as f:
                f.write(f"avg_read,{avg_read}\n")
                f.write(f"avg_write,{avg_write}\n")
                f.write(f"avg_total,{float(np.mean(total_times)) if total_times else 0.0}\n")
                # write raw samples (one per line)
                f.write("read,write,total\n")
                for r, w in zip(all_read_times, all_write_times):
                    f.write(f"{r},{w},{r+w}\n")
            logger.info(f"Wrote summary log to {args.log_out}")

    finally:
        try:
            logger.info("Disconnecting...")
            hand.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()
