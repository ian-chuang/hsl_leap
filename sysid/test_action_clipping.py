import argparse
import time
import logging
from typing import Dict

from hsl_leap.leap_hand import LeapHand, LeapHandConfig, MJ_ZERO_POSITION


logger = logging.getLogger(__name__)


def run_clipping(
    hand: LeapHand,
    motor: str,
    delta: float,
    half_duration: float,
    cycles: int,
    rate: float,
):
    dt = 1.0 / rate
    motor_key = motor

    # prepare zero-action base
    base_action: Dict[str, float] = {f"{m}.pos": 0.0 for m in hand.bus.motors}

    for c in range(cycles):
        logger.info(f"Cycle {c+1}/{cycles}: +{delta} for {half_duration}s")
        t_end = time.time() + half_duration
        while time.time() < t_end:
            t0 = time.time()
            obs = hand.get_observation()
            cur = obs.get(f"{motor_key}.pos")
            if cur is None:
                logger.warning(f"Could not read {motor_key} position; skipping iteration")
                time.sleep(dt)
                continue

            target = cur + delta
            action = base_action.copy()
            action[f"{motor_key}.pos"] = float(target)
            hand.send_action(action)

            sleep_time = dt - (time.time() - t0)
            if sleep_time > 0:
                time.sleep(sleep_time)

        logger.info(f"Cycle {c+1}/{cycles}: -{delta} for {half_duration}s")
        t_end = time.time() + half_duration
        while time.time() < t_end:
            t0 = time.time()
            obs = hand.get_observation()
            cur = obs.get(f"{motor_key}.pos")
            if cur is None:
                logger.warning(f"Could not read {motor_key} position; skipping iteration")
                time.sleep(dt)
                continue

            target = cur - delta
            action = base_action.copy()
            action[f"{motor_key}.pos"] = float(target)
            hand.send_action(action)

            sleep_time = dt - (time.time() - t0)
            if sleep_time > 0:
                time.sleep(sleep_time)


def main():
    parser = argparse.ArgumentParser(description="Test action clipping: continuously set one joint to current+delta / current-delta")
    parser.add_argument("--port", default="/dev/ttyDXL_leap_hand")
    parser.add_argument("--motor", default="if_mcp", help="Motor name, e.g. if_mcp or joint_0")
    parser.add_argument("--delta", type=float, default="20.0", help="Delta (degrees) to add/subtract to current position")
    parser.add_argument("--half_duration", type=float, default=1.0, help="Seconds per half direction")
    parser.add_argument("--cycles", type=int, default=10, help="Number of back-and-forth cycles")
    parser.add_argument("--rate", type=float, default=20.0, help="Control loop rate (Hz)")
    parser.add_argument("--baudrate", type=int, default=4_000_000)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    cfg = LeapHandConfig(port=args.port, baudrate=args.baudrate)
    hand = LeapHand(cfg)

    try:
        logger.info("Connecting to hand...")
        hand.connect()

        logger.info("Moving to zero position...")
        hand.move(MJ_ZERO_POSITION, duration=1.0)

        if args.motor not in hand.bus.motors:
            raise KeyError(f"Motor '{args.motor}' not found. Available: {list(hand.bus.motors.keys())}")

        run_clipping(hand, args.motor, args.delta, args.half_duration, args.cycles, args.rate)

    finally:
        try:
            logger.info("Disconnecting...")
            hand.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()
