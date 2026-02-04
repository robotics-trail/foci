import time
import numpy as np

from scipy.spatial.transform import Rotation as R

import viser
from viser.extras import ViserUrdf
from pathlib import Path

from yourdfpy import URDF

from src.core.robot_loader import ManipulatorRobotURDF


class Visualizer:
    def __init__(
        self,
        robot: ManipulatorRobotURDF,
        # robot_midpoints: np.ndarray,
        robot_cov: np.ndarray,
        curve: np.ndarray,
        camera_position: np.ndarray = None,
        camera_look_at: np.ndarray = None,
    ):
        self.robot = robot
        self.n_links = self.robot.get_n_links()
        self.n_joints = self.robot.get_n_joints()

        # self.robot_midpoints = robot_midpoints  # (num_samples, num_links, 3)
        self.robot_cov = robot_cov  # (3, 3)

        self.curve = curve
        self.midpoints = self.get_midpoints(self.curve)
        self.positions = self.get_positions(self.curve)

        self.server = viser.ViserServer()

        urdf = URDF.load(self.robot.get_robot_path())
        self.viser_urdf = ViserUrdf(self.server, urdf_or_path=urdf)

        if camera_position is not None and camera_look_at is not None:
            self._setup_camera(camera_position, camera_look_at)

    def visualize_trajectory(
        self,
        dt: float = 0.1,
        loop: bool = True,
        save_recording: bool = False,
        recording_path: str = "trajectory.viser",
    ):

        if save_recording:
            self._save_trajectory_recording(recording_path, dt)

        else:
            self._visualize_trajectory_live(dt, loop)

    def visualize_goal(
        self, goal: np.ndarray, radius: float = 0.05, color: tuple = (0, 0, 255)
    ):
        self.server.scene.add_icosphere(
            name="Goal", position=goal, radius=radius, color=color
        )

    def visualize_obstacles(
        self,
        means: np.ndarray,
        covariances: np.ndarray,
        n_std: float = 2.0,
        color: tuple = (255, 100, 100),
        opacity: float = 0.6,
        name="Obstacle",
    ):
        for i, (mean, cov) in enumerate(zip(means, covariances)):
            eigvals, eigvecs = np.linalg.eigh(cov)
            radii = n_std * np.sqrt(np.abs(eigvals))

            rotation_matrix = eigvecs

            if np.linalg.det(rotation_matrix) < 0:
                rotation_matrix[:, 0] *= -1

            rotation = R.from_matrix(rotation_matrix)

            quat_xyzw = rotation.as_quat()
            quat_wxyz = np.array(
                [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
            )

            self.server.scene.add_mesh_simple(
                name=f"{name}_{i}",
                vertices=self._create_ellipsoid_mesh(radii),
                faces=self._create_ellipsoid_faces(),
                position=mean,
                wxyz=quat_wxyz,
                color=color,
                opacity=opacity,
            )

    def _create_ellipsoid_mesh(self, radii: np.ndarray, resolution: int = 20):
        u = np.linspace(0, 2 * np.pi, resolution)
        v = np.linspace(0, np.pi, resolution)

        u_grid, v_grid = np.meshgrid(u, v)

        x = radii[0] * np.cos(u_grid) * np.sin(v_grid)
        y = radii[1] * np.sin(u_grid) * np.sin(v_grid)
        z = radii[2] * np.cos(v_grid)

        vertices = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)
        return vertices

    def _create_ellipsoid_faces(self, resolution: int = 20):
        faces = []

        for i in range(resolution - 1):
            for j in range(resolution - 1):
                idx = i * resolution + j
                faces.append([idx, idx + resolution, idx + 1])
                faces.append([idx + 1, idx + resolution, idx + resolution + 1])

        return np.array(faces, dtype=np.uint32)

    def get_positions(self, curve: np.ndarray):
        num_samples, n_joints = curve.shape
        positions = []

        for i in range(num_samples):
            q = curve[i]
            sample_positions = np.array(
                self.robot.forward_kinematics(q)
            )  # (n_links + 1,)

            positions.append(sample_positions)

        return positions

    def get_midpoints(self, curve):
        num_samples = curve.shape[0]
        midpoints = np.zeros((num_samples, self.n_links, 3))

        for i in range(num_samples):
            q = curve[i]
            joint_positions = self.robot.forward_kinematics(q)  # (n_links + 1,)

            for j in range(self.n_links):
                midpoints[i, j, :] = (joint_positions[j] + joint_positions[j + 1]) / 2.0

        return midpoints

    def add_gaussians(self, means, covs, color=[0, 1, 0], opacity=1.0):
        if len(color) == 3:
            color = self.z_colormap(means)
        if type(opacity) == float:
            opacity = np.tile(opacity, (len(means), 1))

        means = np.ascontiguousarray(means)

        print(
            f"means: {means.shape}, covs: {covs.shape}, color: {color.shape}, opacity: {opacity.shape}"
        )
        self.server.add_gaussian_splats(
            "Scene Splat", means, covs, color, opacity, visible=True
        )

    def _visualize_trajectory_live(self, dt: float, loop: bool):
        self.viser_urdf.update_cfg(np.zeros(self.n_joints))

        num_samples: int = self.curve.shape[0]
        i = 0

        while True:
            q = self.curve[i]  # (n_joints, )
            self.viser_urdf.update_cfg(q)

            time.sleep(dt)

            i += 1
            if i >= num_samples:
                if loop:
                    i = 0

                else:
                    break

    def _save_trajectory_recording(self, recording_path: str, dt: float):
        serializer = self.server.get_scene_serializer()

        num_samples: int = self.curve.shape[0]
        self.viser_urdf.update_cfg(np.zeros(self.n_joints))

        for i in range(num_samples):
            q = self.curve[i]  # (n_joints, )
            self.viser_urdf.update_cfg(q)

            serializer.insert_sleep(dt)

        data = serializer.serialize()
        Path(recording_path).write_bytes(data)

    def _setup_camera(self, position: np.ndarray, look_at: np.ndarray):
        direction = look_at - position
        # distance = np.linalg.norm(direction)

        with self.server.scene.atomic():
            self.server.scene.set_up_direction("+z")
