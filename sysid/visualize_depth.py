"""RealSense depth + point cloud visualizer using viser.

Uses the RealSenseCamera helper for streaming and async reads.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
import pyrealsense2 as rs
import tyro

import viser

from hsl_leap.cameras.configs import ColorMode, Cv2Rotation
from hsl_leap.cameras.realsense import RealSenseCamera, RealSenseCameraConfig


@dataclass
class Args:
	serial: str = "250222071274"
	width: int | None = 424
	height: int | None = 240
	fps: int | None = 60
	max_depth_m: float = 2.5
	max_points: int = 150_000
	point_size: float = 0.002
	point_shape: Literal["square", "diamond", "circle", "rounded", "sparkle"] = "circle"
	stride: int = 1


def _get_intrinsics(camera: RealSenseCamera) -> rs.intrinsics:
	if camera.rs_profile is None:
		raise RuntimeError("Camera profile not initialized.")
	stream = camera.rs_profile.get_stream(rs.stream.depth).as_video_stream_profile()
	return stream.get_intrinsics()


def _get_depth_scale(camera: RealSenseCamera) -> float:
	if camera.rs_profile is None:
		return 0.001
	try:
		device = camera.rs_profile.get_device()
		depth_sensor = device.first_depth_sensor()
		return float(depth_sensor.get_depth_scale())
	except Exception:
		return 0.001


def _depth_to_pointcloud(
	depth_m: np.ndarray,
	intr: rs.intrinsics,
	max_depth_m: float,
	max_points: int,
	stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	depth_m = depth_m[::stride, ::stride]
	height, width = depth_m.shape
	valid = (depth_m > 0.0) & (depth_m < max_depth_m)
	if not np.any(valid):
		return (
			np.zeros((1, 3), dtype=np.float32),
			np.zeros((1,), dtype=np.int32),
			np.zeros((1,), dtype=np.int32),
		)

	ys, xs = np.nonzero(valid)
	z = depth_m[ys, xs]
	x = (xs.astype(np.float32) - intr.ppx / stride) / (intr.fx / stride) * z
	y = (ys.astype(np.float32) - intr.ppy / stride) / (intr.fy / stride) * z
	points = np.stack([x, y, z], axis=1).astype(np.float32)

	if points.shape[0] > max_points:
		indices = np.random.choice(points.shape[0], max_points, replace=False)
		points = points[indices]
		ys = ys[indices]
		xs = xs[indices]
	return points, ys, xs


def _colorize_depth(depth_m: np.ndarray, max_depth_m: float) -> np.ndarray:
	depth_norm = np.clip(depth_m / max_depth_m, 0.0, 1.0)
	depth_u8 = (depth_norm * 255.0).astype(np.uint8)
	depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_VIRIDIS)
	return cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB)


def main(args: Args) -> None:
	config = RealSenseCameraConfig(
		serial_number_or_name=args.serial,
		fps=args.fps,
		width=args.width,
		height=args.height,
		use_depth=True,
		color_mode=ColorMode.RGB,
		rotation=Cv2Rotation.NO_ROTATION,
	)
	camera = RealSenseCamera(config)
	camera.connect()

	intr = _get_intrinsics(camera)
	depth_scale = _get_depth_scale(camera)

	server = viser.ViserServer()
	server.scene.add_grid("/grid", width=2.0, height=2.0)
	point_cloud = server.scene.add_point_cloud(
		"/realsense/point_cloud",
		np.zeros((1, 3), dtype=np.float32),
		np.zeros((1, 3), dtype=np.uint8),
		point_size=args.point_size,
		point_shape=args.point_shape,
	)
	depth_image = server.gui.add_image(
		np.zeros((2, 2, 3), dtype=np.uint8),
		label="Depth (colorized)",
	)

	try:
		while True:
			camera.async_read(timeout_ms=200)
			with camera.frame_lock:
				depth = None if camera.latest_depth_frame is None else camera.latest_depth_frame.copy()
				color = None if camera.latest_color_frame is None else camera.latest_color_frame.copy()

			if depth is None:
				continue

			depth_m = depth.astype(np.float32) * depth_scale
			depth_image.image = _colorize_depth(depth_m, args.max_depth_m)

			points, ys, xs = _depth_to_pointcloud(
				depth_m,
				intr,
				args.max_depth_m,
				args.max_points,
				max(args.stride, 1),
			)

			if color is not None:
				color = color[:: max(args.stride, 1), :: max(args.stride, 1)]
				colors = color[ys, xs]
				point_cloud.colors = colors.astype(np.uint8)
			else:
				point_cloud.colors = np.tile(np.array([200, 200, 200], dtype=np.uint8), (points.shape[0], 1))

			point_cloud.points = points
			time.sleep(0.001)
	finally:
		camera.disconnect()


if __name__ == "__main__":
	main(tyro.cli(Args))
