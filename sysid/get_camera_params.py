"""Dump RealSense D435 camera parameters to stdout and a log file.

Usage:
    python get_camera_params.py [--serial SERIAL]

Connects to the RealSense, prints device info, stream profiles, intrinsics,
and all supported sensor option values and ranges.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pyrealsense2 as rs

logger = logging.getLogger("get_camera_params")
logging.basicConfig(level=logging.INFO, format="%(message)s")


def option_name(opt: Any) -> str:
    try:
        return opt.name
    except Exception:
        return str(opt)


def dump_device_info(device: rs.device) -> dict:
    out = {}
    for info in [
        rs.camera_info.name,
        rs.camera_info.serial_number,
        rs.camera_info.firmware_version,
        rs.camera_info.physical_port,
        rs.camera_info.usb_type_descriptor,
        rs.camera_info.camera_locked,
    ]:
        try:
            out[str(info).split('.')[-1]] = device.get_info(info)
        except Exception:
            # some fields may not be present
            pass
    return out


def dump_stream_profiles(profile: rs.pipeline_profile) -> list:
    out = []
    try:
        for s in profile.get_streams():
            info = {}
            try:
                info["stream_name"] = s.stream_name()
            except Exception:
                pass
            try:
                info["stream_type"] = str(s.stream_type())
            except Exception:
                pass
            try:
                info["format"] = s.format()
            except Exception:
                pass
            try:
                vsp = s.as_video_stream_profile()
                info["width"] = vsp.width()
                info["height"] = vsp.height()
                info["fps"] = vsp.fps()
            except Exception:
                pass
            out.append(info)
    except Exception:
        # fallback for older API
        pass
    return out


def dump_all_supported_profiles(device: rs.device) -> dict:
    """Return a mapping of sensor name -> list of supported stream profiles.

    Each profile includes stream type, format, width, height, fps and, when
    available, intrinsics.
    """
    out: dict[str, list] = {}
    try:
        for sensor in device.sensors:
            key = None
            try:
                key = sensor.get_info(rs.camera_info.name)
            except Exception:
                key = str(sensor)
            profiles = []
            try:
                for sp in sensor.get_stream_profiles():
                    p: dict[str, Any] = {}
                    try:
                        p["stream_type"] = str(sp.stream_type())
                    except Exception:
                        pass
                    try:
                        vsp = sp.as_video_stream_profile()
                        p["width"] = vsp.width()
                        p["height"] = vsp.height()
                        p["fps"] = vsp.fps()
                        try:
                            p["format"] = vsp.format()
                        except Exception:
                            pass
                        # intrinsics
                        try:
                            intr = vsp.get_intrinsics()
                            p["intrinsics"] = {
                                "ppx": intr.ppx,
                                "ppy": intr.ppy,
                                "fx": intr.fx,
                                "fy": intr.fy,
                                "model": str(intr.model),
                                "coeffs": list(intr.coeffs),
                                "width": intr.width,
                                "height": intr.height,
                            }
                        except Exception:
                            pass
                    except Exception:
                        # non-video profile (e.g., motion) - try generic fields
                        try:
                            p["format"] = sp.format()
                        except Exception:
                            pass
                    profiles.append(p)
            except Exception:
                # sensor may not support get_stream_profiles
                profiles = []
            out[key] = profiles
    except Exception:
        pass
    return out


def dump_intrinsics(profile: rs.stream_profile) -> dict:
    try:
        vsp = profile.as_video_stream_profile()
        intr = vsp.get_intrinsics()
        return {
            "width": intr.width,
            "height": intr.height,
            "ppx": intr.ppx,
            "ppy": intr.ppy,
            "fx": intr.fx,
            "fy": intr.fy,
            "model": str(intr.model),
            "coeffs": list(intr.coeffs),
        }
    except Exception:
        return {}


def dump_sensor_options(sensor: rs.sensor) -> dict:
    out = {}
    try:
        opts = list(sensor.get_supported_options())
    except Exception:
        # older bindings may not implement get_supported_options
        opts = []
    for opt in opts:
        try:
            name = option_name(opt)
            val = sensor.get_option(opt)
            try:
                rng = sensor.get_option_range(opt)
                out[name] = {
                    "value": float(val),
                    "min": float(rng.min),
                    "max": float(rng.max),
                    "step": float(rng.step),
                    "default": float(getattr(rng, "def")),
                }
            except Exception:
                out[name] = {"value": float(val)}
        except Exception:
            # skip options we can't read
            continue
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Dump RealSense camera params")
    p.add_argument("--serial", type=str, default=None, help="device serial (optional)")
    p.add_argument("--out", type=str, default=None, help="output file path")
    args = p.parse_args()

    pipeline = rs.pipeline()
    cfg = rs.config()
    if args.serial:
        cfg.enable_device(args.serial)

    profile = None
    try:
        profile = pipeline.start(cfg)
    except Exception as e:
        logger.error("Failed to start RealSense pipeline: %s", e)
        return 2

    try:
        device = profile.get_device()
    except Exception:
        # try alternative
        try:
            device = profile.get_device()
        except Exception as e:
            logger.error("Failed to get device from profile: %s", e)
            pipeline.stop()
            return 3

    data: dict[str, Any] = {}
    data["device_info"] = dump_device_info(device)
    data["stream_profiles"] = dump_stream_profiles(profile)

    # intrinsics by stream
    intrinsics = {}
    try:
        for s in profile.get_streams():
            try:
                key = getattr(s, "stream_name", lambda: str(s))()
            except Exception:
                key = str(s)
            intrinsics[key] = dump_intrinsics(s)
    except Exception:
        pass
    data["intrinsics"] = intrinsics

    # supported profiles (all resolutions/formats/fps supported by each sensor)
    try:
        data["supported_profiles"] = dump_all_supported_profiles(device)
    except Exception:
        data["supported_profiles"] = {}

    # compact summary: per-stream unique (width,height,fps,format)
    supported_res = {}
    try:
        for sensor_name, profiles in data.get("supported_profiles", {}).items():
            uniq = []
            seen = set()
            for p in profiles:
                w = p.get("width")
                h = p.get("height")
                fps = p.get("fps")
                fmt = p.get("format")
                key = (w, h, fps, fmt)
                if key in seen:
                    continue
                seen.add(key)
                uniq.append({"width": w, "height": h, "fps": fps, "format": fmt})
            supported_res[sensor_name] = uniq
    except Exception:
        supported_res = {}
    data["supported_resolutions"] = supported_res

    # per-sensor options
    sensors = {}
    try:
        for sensor in device.sensors:
            try:
                sensors[sensor.get_info(rs.camera_info.name)] = dump_sensor_options(sensor)
            except Exception:
                try:
                    sensors[str(sensor)] = dump_sensor_options(sensor)
                except Exception:
                    continue
    except Exception:
        pass
    data["sensors"] = sensors

    # depth scale (common)
    try:
        depth_sensor = device.first_depth_sensor()
        try:
            data["depth_scale"] = depth_sensor.get_depth_scale()
        except Exception:
            pass
    except Exception:
        pass

    def _json_default(o):
        # numpy types, enums, pyrealsense enums -> primitives or strings
        try:
            import numpy as _np

            if isinstance(o, _np.generic):
                return o.item()
        except Exception:
            pass
        try:
            # some rs enums and objects stringify usefully
            return str(o)
        except Exception:
            return None

    text = json.dumps(data, indent=2, default=_json_default)
    if args.out:
        Path(args.out).write_text(text)
        logger.info("Wrote camera params to %s", args.out)
    else:
        print(text)

    pipeline.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
