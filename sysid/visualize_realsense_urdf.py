"""RealSense D435 + URDF visualizer using viser.

Streams a colored point cloud from the RealSense and renders a URDF, with
separate pose controls for the camera frame and the robot root.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Tuple

import numpy as np
import pyrealsense2 as rs
import tyro
import yourdfpy
from scipy.spatial.transform import Rotation as R

import viser
from viser.extras import ViserUrdf

from hsl_leap.leap_hand import LeapHand, LeapHandConfig

URDF_PATH = (
    Path(__file__).parent
    / "assets"
    / "leap_hand"
    / "urdf"
    / "leap_hand_right.urdf"
)

logger = logging.getLogger(__name__)


@dataclass
class Args:
    width: int = 424
    height: int = 240
    fps: int = 60
    max_points: int = 150_000
    max_depth_m: float = 2.5
    point_size: float = 0.002
    point_shape: Literal["square", "diamond", "circle", "rounded", "sparkle"] = (
        "circle"
    )
    serial: str | None = None
    urdf_path: str = str(URDF_PATH.absolute())
    load_meshes: bool = True
    load_collision_meshes: bool = True
    use_real_robot: bool = False
    robot_port: str = "/dev/ttyDXL_leap_hand"
    use_mj_motor_config: bool = False
    real_robot_poll_hz: float = 30.0
    real_robot_torque_off: bool = True
    default_robot_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    default_robot_orientation: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    default_camera_position: Tuple[float, float, float] = (
        -0.17,
        0.023,
        0.436,
    )
    default_camera_orientation: Tuple[float, float, float] = (
        -2.62,
        0.0,
        -1.5707,
    )
    log_level: int = logging.INFO



def _sample_points(
    points: np.ndarray, colors: np.ndarray, max_points: int
) -> tuple[np.ndarray, np.ndarray]:
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
    texcoords = (
        np.asanyarray(points.get_texture_coordinates())
        .view(np.float32)
        .reshape(-1, 2)
    )

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


def _create_pose_sliders(
    server: viser.ViserServer,
    node: viser.SceneNodeHandle,
    label: str,
    initial_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    initial_rpy: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> None:
    with server.gui.add_folder(label):
        with server.gui.add_folder("Position"):
            pos_sliders = []
            for axis in ["x", "y", "z"]:
                slider = server.gui.add_slider(
                    label=axis,
                    min=-2.0,
                    max=2.0,
                    step=0.01,
                    initial_value=initial_pos[["x", "y", "z"].index(axis)],
                )
                pos_sliders.append(slider)

        with server.gui.add_folder("Orientation"):
            rpy_sliders = []
            for axis in ["roll", "pitch", "yaw"]:
                slider = server.gui.add_slider(
                    label=axis,
                    min=-np.pi,
                    max=np.pi,
                    step=0.01,
                    initial_value=initial_rpy[["roll", "pitch", "yaw"].index(axis)],
                )
                rpy_sliders.append(slider)

    def update_pose(_):
        pos = np.array([s.value for s in pos_sliders])
        rpy = np.array([s.value for s in rpy_sliders])

        node.position = pos
        quat_xyzw = R.from_euler("xyz", rpy).as_quat()
        node.wxyz = np.array(
            [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
        )

    for s in pos_sliders + rpy_sliders:
        s.on_update(update_pose)
    # set initial node pose from provided values
    update_pose(None)


def _create_robot_control_sliders(
    server: viser.ViserServer,
    viser_urdf: ViserUrdf,
) -> tuple[list[viser.GuiInputHandle[float]], list[float]]:
    slider_handles: list[viser.GuiInputHandle[float]] = []
    initial_config: list[float] = []
    for joint_name, (lower, upper) in viser_urdf.get_actuated_joint_limits().items():
        lower = lower if lower is not None else -np.pi
        upper = upper if upper is not None else np.pi
        initial_pos = 0.0 if lower < -0.1 and upper > 0.1 else (lower + upper) / 2.0
        slider = server.gui.add_slider(
            label=joint_name,
            min=lower,
            max=upper,
            step=1e-3,
            initial_value=initial_pos,
        )
        slider.on_update(
            lambda _: viser_urdf.update_cfg(
                np.array([slider.value for slider in slider_handles])
            )
        )
        slider_handles.append(slider)
        initial_config.append(initial_pos)
    return slider_handles, initial_config


def _extract_joint_positions(
    obs: dict[str, float],
    joint_names: tuple[str, ...],
    missing_warning: list[bool],
) -> np.ndarray | None:
    positions: list[float] = []
    missing: list[str] = []
    for name in joint_names:
        if name in obs:
            positions.append(obs[name])
        elif f"{name}.pos" in obs:
            positions.append(obs[f"{name}.pos"])
        elif name.endswith(".pos") and name[:-4] in obs:
            positions.append(obs[name[:-4]])
        else:
            missing.append(name)

    if missing:
        if not missing_warning[0]:
            logger.warning(
                "Missing joint(s) in real robot observation: %s",
                ", ".join(missing[:6]),
            )
            missing_warning[0] = True
        return None

    return np.deg2rad(np.array(positions, dtype=np.float32))


def _safe_disconnect(hand: LeapHand | None) -> None:
    if hand is None:
        return
    try:
        if hand.is_connected:
            hand.disconnect()
    except Exception as exc:
        logger.warning("Failed to disconnect cleanly: %s", exc)
        try:
            hand.bus.disconnect(False)
        except Exception as exc2:
            logger.warning("Forced bus disconnect failed: %s", exc2)


def main(args: Args) -> None:
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s: %(message)s")

    config = rs.config()
    if args.serial is not None:
        config.enable_device(args.serial)

    config.enable_stream(
        rs.stream.depth, args.width, args.height, rs.format.z16, args.fps
    )
    config.enable_stream(
        rs.stream.color, args.width, args.height, rs.format.rgb8, args.fps
    )

    pipeline = rs.pipeline()
    align = rs.align(rs.stream.color)
    pointcloud = rs.pointcloud()
    pipeline.start(config)

    # Log RealSense camera parameters (intrinsics + depth scale)
    try:
        profile = pipeline.get_active_profile()
        device = profile.get_device()

        # depth scale
        try:
            depth_sensor = device.first_depth_sensor()
            depth_scale = depth_sensor.get_depth_scale()
            logger.info("RealSense depth scale: %s meters", depth_scale)
        except Exception:
            logger.info("RealSense depth scale: unavailable")

        def _log_intrinsics(stream_type, prof):
            try:
                sp = prof.get_stream(stream_type).as_video_stream_profile()
                intr = sp.get_intrinsics()
                coeffs = np.array(intr.coeffs) if hasattr(intr, "coeffs") else None
                logger.info(
                    "%s intrinsics: width=%d height=%d ppx=%.2f ppy=%.2f fx=%.2f fy=%.2f model=%s coeffs=%s",
                    stream_type,
                    intr.width,
                    intr.height,
                    intr.ppx,
                    intr.ppy,
                    intr.fx,
                    intr.fy,
                    getattr(intr, "model", None),
                    coeffs,
                )
            except Exception as exc:
                logger.warning("Failed to get intrinsics for %s: %s", stream_type, exc)

        _log_intrinsics(rs.stream.depth, profile)
        _log_intrinsics(rs.stream.color, profile)
    except Exception as exc:
        logger.warning("Failed to log RealSense camera parameters: %s", exc)

    server = viser.ViserServer()

    # Make axes smaller so they are less obtrusive in the view.
    camera_root = server.scene.add_frame(
        "/camera_root", show_axes=True, axes_length=0.08, axes_radius=0.006
    )
    robot_root = server.scene.add_frame(
        "/robot_root", show_axes=True, axes_length=0.08, axes_radius=0.006
    )

    _create_pose_sliders(
        server,
        camera_root,
        "Camera Pose",
        initial_pos=tuple(args.default_camera_position),
        initial_rpy=tuple(args.default_camera_orientation),
    )
    _create_pose_sliders(
        server,
        robot_root,
        "URDF Pose",
        initial_pos=tuple(args.default_robot_position),
        initial_rpy=tuple(args.default_robot_orientation),
    )

    urdf = yourdfpy.URDF.load(args.urdf_path)
    viser_urdf = ViserUrdf(
        server,
        urdf_or_path=urdf,
        load_meshes=args.load_meshes,
        load_collision_meshes=args.load_collision_meshes,
        collision_mesh_color_override=(1.0, 0.0, 0.0, 0.5),
        root_node_name="/robot_root",
    )

    joint_names = viser_urdf.get_actuated_joint_names()
    hand: LeapHand | None = None
    use_real_robot = args.use_real_robot
    last_real_read = 0.0
    missing_warning = [False]

    with server.gui.add_folder("Joint Position Control"):
        slider_handles, initial_config = _create_robot_control_sliders(server, viser_urdf)

    with server.gui.add_folder("URDF Visibility"):
        show_meshes_cb = server.gui.add_checkbox("Show meshes", viser_urdf.show_visual)
        show_collision_meshes_cb = server.gui.add_checkbox(
            "Show collision meshes", viser_urdf.show_collision
        )

    with server.gui.add_folder("Real Robot"):
        use_real_robot_cb = server.gui.add_checkbox(
            "Use real robot", args.use_real_robot
        )

    @show_meshes_cb.on_update
    def _(_):
        viser_urdf.show_visual = show_meshes_cb.value

    @show_collision_meshes_cb.on_update
    def _(_):
        viser_urdf.show_collision = show_collision_meshes_cb.value

    show_meshes_cb.visible = args.load_meshes
    show_collision_meshes_cb.visible = args.load_collision_meshes

    viser_urdf.update_cfg(np.array(initial_config))

    point_cloud = server.scene.add_point_cloud(
        "/camera_root/point_cloud",
        np.zeros((1, 3), dtype=np.float32),
        np.zeros((1, 3), dtype=np.uint8),
        point_size=args.point_size,
        point_shape=args.point_shape,
    )

    server.scene.add_grid("/grid", width=2.0, height=2.0)

    reset_button = server.gui.add_button("Reset Joints")

    @reset_button.on_click
    def _(_):
        for s, init_q in zip(slider_handles, initial_config):
            s.value = init_q

    def _set_real_robot_enabled(enabled: bool) -> None:
        nonlocal use_real_robot, last_real_read
        use_real_robot = enabled
        if enabled:
            last_real_read = 0.0

    @use_real_robot_cb.on_update
    def _(_):
        _set_real_robot_enabled(use_real_robot_cb.value)

    try:
        hand = LeapHand(
            LeapHandConfig(
                port=args.robot_port,
                use_mj_motor_config=args.use_mj_motor_config,
            )
        )
        hand.connect()
        if args.real_robot_torque_off:
            hand.torque_off()
    except Exception as exc:
        logger.exception("Failed to connect to real robot: %s", exc)
        _safe_disconnect(hand)
        hand = None
        use_real_robot_cb.value = False
        use_real_robot = False

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

            if points.size:
                quat_wxyz = camera_root.wxyz
                quat_xyzw = np.array(
                    [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
                )
                if np.linalg.norm(quat_xyzw) < 1e-3:
                    world_rot = np.eye(3)
                else:
                    try:
                        world_rot = R.from_quat(quat_xyzw).as_matrix()
                    except Exception as exc:
                        logger.warning(
                            "Invalid camera_root quaternion %s — falling back to identity: %s",
                            quat_xyzw,
                            exc,
                        )
                        world_rot = np.eye(3)
                world_points = points @ world_rot.T + camera_root.position
                filter_table = ((world_points[:, 2] >= -0.08) & (world_points[:, 0] >= 0.0)) | \
                       ((world_points[:, 2] >= 0.01) & (world_points[:, 0] <= 0.0)) 
                filter_y = (world_points[:, 1] >= -0.3) & (world_points[:, 1] <= 0.18)
                filter_x = (world_points[:, 0] >= -0.1) & (world_points[:, 0] <= 0.3)
                keep = filter_table & filter_y & filter_x   
                points = points[keep]
                colors = colors[keep]

            point_cloud.points = points
            point_cloud.colors = colors

            if use_real_robot and hand is not None:
                now = time.time()
                if now - last_real_read >= 1.0 / max(args.real_robot_poll_hz, 1e-3):
                    last_real_read = now
                    try:
                        obs = hand.get_observation()
                        positions = _extract_joint_positions(
                            obs, joint_names, missing_warning
                        )
                        if positions is not None:
                            viser_urdf.update_cfg(positions)
                            for slider, q in zip(slider_handles, positions):
                                slider.value = float(q)
                    except Exception as exc:
                        logger.exception("Real robot read failed: %s", exc)
                        use_real_robot_cb.value = False
                        use_real_robot = False
            time.sleep(0.001)
    finally:
        _safe_disconnect(hand)
        pipeline.stop()


if __name__ == "__main__":
    main(tyro.cli(Args))
