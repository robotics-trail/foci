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
        robot_cov: np.ndarray,
        curve: np.ndarray,
    ):
        self.robot = robot
        self.n_links = self.robot.get_n_links()
        self.n_joints = self.robot.get_n_joints()

        # self.robot_midpoints = robot_midpoints  # (num_samples, num_links, 3)
        self.robot_cov = robot_cov  # (3, 3)

        self.curve = curve

        # print(self.midpoints)
        # print(self.midpoints.shape)

        self.server = viser.ViserServer()

        self.urdf = URDF.load(self.robot.get_robot_path())
        self.viser_urdf = ViserUrdf(self.server, urdf_or_path=self.urdf)

        # self._link_names = [l.name for l in self.urdf.links]
        self.chain_links = self.robot.get_links()
        self.midpoints = self.get_midpoints(self.curve)
        self.positions = self.get_positions(self.curve)

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

    def visualize_robot_gaussians(
        self,
        n_std: float = 2.0,
        color: tuple = (80, 160, 255),
        opacity: float = 0.35,
        name: str = "RobotGaussian",
    ):

        self._robot_gauss_handles = []

        for i in range(self.n_links):
            midpoint = self.midpoints[0, i, :]

            eigvals, eigvecs = np.linalg.eigh(self.robot_cov)
            radii = n_std * np.sqrt(np.abs(eigvals))

            rotation_matrix = eigvecs

            if np.linalg.det(rotation_matrix) < 0:
                rotation_matrix[:, 0] *= -1

            rotation = R.from_matrix(rotation_matrix)

            quat_xyzw = rotation.as_quat()
            quat_wxyz = np.array(
                [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
            )

            handle = self.server.scene.add_mesh_simple(
                name=f"{name}_{i}",
                vertices=self._create_ellipsoid_mesh(radii),
                faces=self._create_ellipsoid_faces(),
                position=midpoint,
                wxyz=quat_wxyz,
                color=color,
                opacity=opacity,
            )

            self._robot_gauss_handles.append(handle)

    def _update_robot_gaussians(self, i: int, n_std=2.0):
        if self._robot_gauss_handles is None or len(self._robot_gauss_handles) == 0:
            return

        for j in range(self.n_links):
            mean = self.midpoints[i, j, :]

            eigvals, eigvecs = np.linalg.eigh(self.robot_cov)
            # radii = n_std * np.sqrt(np.abs(eigvals))

            rotation_matrix = eigvecs

            if np.linalg.det(rotation_matrix) < 0:
                rotation_matrix[:, 0] *= -1

            rotation = R.from_matrix(rotation_matrix)

            quat_xyzw = rotation.as_quat()
            quat_wxyz = np.array(
                [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
            )

            h = self._robot_gauss_handles[j]
            h.position = mean
            h.wxyz = quat_wxyz

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
        midpoints = np.zeros((num_samples, self.n_links, 3), dtype=float)

        for i in range(num_samples):
            q = curve[i]

            fk_flat = (
                np.array(self.robot.forward_kinematics(q)).astype(float).reshape(-1)
            )
            joint_positions = fk_flat.reshape(self.n_links + 1, 3)  # (n_links+1,3)

            for j in range(self.n_links):
                midpoints[i, j] = 0.5 * (joint_positions[j] + joint_positions[j + 1])

        return midpoints

    def _visualize_trajectory_live(self, dt: float, loop: bool):
        self.viser_urdf.update_cfg(np.zeros(self.n_joints))

        num_samples: int = self.curve.shape[0]
        i = 0

        while True:
            q = self.curve[i]  # (n_joints, )
            self.viser_urdf.update_cfg(q)
            self._update_robot_gaussians(i)

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
            self._update_robot_gaussians(i)

            serializer.insert_sleep(dt)

        data = serializer.serialize()
        Path(recording_path).write_bytes(data)
