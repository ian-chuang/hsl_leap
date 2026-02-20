"""URDF robot visualizer

Visualize robot models from URDF files with interactive joint controls.

Requires yourdfpy and robot_descriptions. Any URDF supported by yourdfpy should work.

- https://github.com/robot-descriptions/robot_descriptions.py
- https://github.com/clemense/yourdfpy

**Features:**

* :class:`viser.extras.ViserUrdf` for URDF file parsing and visualization
* Interactive joint sliders for robot articulation
* Real-time robot pose updates
* Support for local URDF files and robot_descriptions library
"""

from __future__ import annotations

import time
from typing import Literal

import numpy as np
import tyro
import yourdfpy
from scipy.spatial.transform import Rotation as R
from pathlib import Path

import viser
from viser.extras import ViserUrdf

URDF_PATH = Path(__file__).parent / "assets" / "leap_hand" / "urdf" / "leap_hand_right.urdf"

def create_robot_control_sliders(
    server: viser.ViserServer, viser_urdf: ViserUrdf
) -> tuple[list[viser.GuiInputHandle[float]], list[float]]:
    """Create slider for each joint of the robot. We also update robot model
    when slider moves."""
    slider_handles: list[viser.GuiInputHandle[float]] = []
    initial_config: list[float] = []
    for joint_name, (
        lower,
        upper,
    ) in viser_urdf.get_actuated_joint_limits().items():
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
        slider.on_update(  # When sliders move, we update the URDF configuration.
            lambda _: viser_urdf.update_cfg(
                np.array([slider.value for slider in slider_handles])
            )
        )
        slider_handles.append(slider)
        initial_config.append(initial_pos)
    return slider_handles, initial_config


def create_pose_sliders(
    server: viser.ViserServer, root_frame: viser.SceneNodeHandle
) -> None:
    """Create sliders for position and orientation of the root frame."""
    
    # Sliders for position
    with server.gui.add_folder("Root Position"):
        pos_sliders = []
        for axis in ["x", "y", "z"]:
            slider = server.gui.add_slider(
                label=axis,
                min=-2.0,
                max=2.0,
                step=0.01,
                initial_value=0.0,
            )
            pos_sliders.append(slider)

    # Sliders for orientation (Euler angles)
    with server.gui.add_folder("Root Orientation"):
        rpy_sliders = []
        for axis in ["roll", "pitch", "yaw"]:
            slider = server.gui.add_slider(
                label=axis,
                min=-np.pi,
                max=np.pi,
                step=0.01,
                initial_value=0.0,
            )
            rpy_sliders.append(slider)

    def update_pose(_):
        pos = np.array([s.value for s in pos_sliders])
        rpy = np.array([s.value for s in rpy_sliders])
        
        # Update position
        root_frame.position = pos
        
        # Update orientation (convert Euler to Quaternion wxyz)
        # viser uses wxyz
        quat_xyzw = R.from_euler("xyz", rpy).as_quat()
        # scipy returns xyzw, viser expects wxyz
        root_frame.wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        print(root_frame.wxyz)

    for s in pos_sliders + rpy_sliders:
        s.on_update(update_pose)


def main(
    urdf_path: str = str(URDF_PATH.absolute()),
    load_meshes: bool = True,
    load_collision_meshes: bool = True,
) -> None:
    # Start viser server.
    server = viser.ViserServer()

    # Create a root frame for the robot
    root_frame = server.scene.add_frame("/robot_root", show_axes=False)

    # Load URDF. it is yourdfpy.URDF
    urdf = yourdfpy.URDF.load(urdf_path)
    # This takes either a yourdfpy.URDF object or a path to a .urdf file.
    viser_urdf = ViserUrdf(
        server,
        urdf_or_path=urdf,
        load_meshes=load_meshes,
        load_collision_meshes=load_collision_meshes,
        collision_mesh_color_override=(1.0, 0.0, 0.0, 0.5),
        root_node_name="/robot_root",
    )

    # Create sliders for root pose
    create_pose_sliders(server, root_frame)

    # Create sliders in GUI that help us move the robot joints.
    with server.gui.add_folder("Joint position control"):
        (slider_handles, initial_config) = create_robot_control_sliders(
            server, viser_urdf
        )

    # Add visibility checkboxes.
    with server.gui.add_folder("Visibility"):
        show_meshes_cb = server.gui.add_checkbox(
            "Show meshes",
            viser_urdf.show_visual,
        )
        show_collision_meshes_cb = server.gui.add_checkbox(
            "Show collision meshes", viser_urdf.show_collision
        )

    @show_meshes_cb.on_update
    def _(_):
        viser_urdf.show_visual = show_meshes_cb.value

    @show_collision_meshes_cb.on_update
    def _(_):
        viser_urdf.show_collision = show_collision_meshes_cb.value

    # Hide checkboxes if meshes are not loaded.
    show_meshes_cb.visible = load_meshes
    show_collision_meshes_cb.visible = load_collision_meshes

    # Set initial robot configuration.
    viser_urdf.update_cfg(np.array(initial_config))

    # Create grid.
    trimesh_scene = viser_urdf._urdf.scene or viser_urdf._urdf.collision_scene
    server.scene.add_grid(
        "/grid",
        width=2,
        height=2,
        position=(
            0.0,
            0.0,
            # Get the minimum z value of the trimesh scene.
            trimesh_scene.bounds[0, 2] if trimesh_scene is not None else 0.0,
        ),
    )

    # Create joint reset button.
    reset_button = server.gui.add_button("Reset")

    @reset_button.on_click
    def _(_):
        for s, init_q in zip(slider_handles, initial_config):
            s.value = init_q

    # Sleep forever.
    while True:
        time.sleep(10.0)


if __name__ == "__main__":
    tyro.cli(main)