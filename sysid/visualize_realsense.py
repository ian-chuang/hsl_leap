"""RealSense D435 point cloud visualizer using viser.

Requires: pyrealsense2, viser, numpy, tyro.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pyrealsense2 as rs
import tyro

import viser


@dataclass
class Args:
	width: int = 640
	height: int = 480
	fps: int = 30
	max_points: int = 150_000
	max_depth_m: float = 2.5
	point_size: float = 0.002
	point_shape: Literal["square", "diamond", "circle", "rounded", "sparkle"] = (
		"circle"
	)
	serial: str | None = None


def _sample_points(points: np.ndarray, colors: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
	if points.shape[0] <= max_points:
		return points, colors
	indices = np.random.choice(points.shape[0], max_points, replace=False)
	return points[indices], colors[indices]


def _points_from_frames(
	depth_frame: rs.depth_frame,
	color_frame: rs.video_frame,
	pointcloud: rs.pointcloud,
	max_depth_m: float,
) -> tuple[np.ndarray, np.ndarray]:
	pointcloud.map_to(color_frame)
	points = pointcloud.calculate(depth_frame)

	vertices = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
	texcoords = np.asanyarray(points.get_texture_coordinates()).view(np.float32).reshape(-1, 2)

	color_image = np.asanyarray(color_frame.get_data())
	height, width = color_image.shape[:2]

	u = np.clip((texcoords[:, 0] * width).astype(np.int32), 0, width - 1)
	v = np.clip((texcoords[:, 1] * height).astype(np.int32), 0, height - 1)
	colors = color_image[v, u, :]

	valid = (
		np.isfinite(vertices).all(axis=1)
		& (vertices[:, 2] > 0.0)
		& (vertices[:, 2] < max_depth_m)
	)
	return vertices[valid], colors[valid]


def main(args: Args) -> None:
	config = rs.config()
	if args.serial is not None:
		config.enable_device(args.serial)

	config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
	config.enable_stream(rs.stream.color, args.width, args.height, rs.format.rgb8, args.fps)

	pipeline = rs.pipeline()
	align = rs.align(rs.stream.color)
	pointcloud = rs.pointcloud()

	pipeline.start(config)

	server = viser.ViserServer()
	server.scene.add_grid("/grid", width=2.0, height=2.0)
	point_cloud = server.scene.add_point_cloud(
		"/realsense/point_cloud",
		np.zeros((1, 3), dtype=np.float32),
		np.zeros((1, 3), dtype=np.uint8),
		point_size=args.point_size,
		point_shape=args.point_shape,
	)

	try:
		while True:
			frames = pipeline.wait_for_frames()
			aligned = align.process(frames)
			depth_frame = aligned.get_depth_frame()
			color_frame = aligned.get_color_frame()
			if not depth_frame or not color_frame:
				continue

			points, colors = _points_from_frames(
				depth_frame,
				color_frame,
				pointcloud,
				args.max_depth_m,
			)
			points, colors = _sample_points(points, colors, args.max_points)

			point_cloud.points = points
			point_cloud.colors = colors
			time.sleep(0.001)
	finally:
		pipeline.stop()


if __name__ == "__main__":
	main(tyro.cli(Args))
