import time
import numpy as np

from viser import ViserServer
from viser.extras import ViserUrdf
from pathlib import Path

from yourdfpy import URDF

from typing import List, Optional

from src.core.robot_loader import ManipulatorRobotURDF
from src.visualization.utils import (
    CasadiFKMidpoints,
    SharedCovariance,
    PerLinkCovariances,
    EllipsoidFactory,
)


class Visualizer:
    def __init__(
        self,
        robot: ManipulatorRobotURDF,
        robot_cov: np.ndarray,
        curve: np.ndarray,
        ignore_link_indices: Optional[List[int]] = None,
    ):

        self.robot = robot
        self.n_links = self.robot.get_n_links()
        self.n_joints = self.robot.get_n_joints()

        ignore_link_indices = ignore_link_indices or []
        ignore_link_indices = sorted(set(ignore_link_indices))

        self.ignore_link_indices = ignore_link_indices
        self.active_link_indices = [
            i for i in range(self.n_links) if i not in ignore_link_indices
        ]
        self.n_active_links = len(self.active_link_indices)

        if robot_cov.shape == (3, 3):
            self.gaussian_model = SharedCovariance(robot_cov)
        elif robot_cov.ndim == 3 and robot_cov.shape == (self.n_links, 3, 3):
            self.gaussian_model = PerLinkCovariances(robot_cov)
        else:
            raise ValueError(
                f"robot_cov must be (3,3) or ({self.n_links},3,3), got {robot_cov.shape}"
            )

        self.kinematics = CasadiFKMidpoints(self.robot, self.n_links)

        self.curve = curve
        self.midpoints = self.kinematics.midpoints(curve)[
            :, self.active_link_indices, :
        ]

        self.server = ViserServer()

        urdf = URDF.load(self.robot.get_robot_path())
        self.viser_urdf = ViserUrdf(self.server, urdf_or_path=urdf)

        self._ellipsoid_faces = self._create_ellipsoid_faces()

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

        factory = EllipsoidFactory(n_std=float(n_std))

        for i, (mean, cov) in enumerate(zip(means, covariances)):
            radii, quat_wxyz = factory.cov_to_ellipsoid(cov)

            self.server.scene.add_mesh_simple(
                name=f"{name}_{i}",
                vertices=self._create_ellipsoid_mesh(radii),
                faces=self._ellipsoid_faces,
                position=mean,
                wxyz=quat_wxyz,
                color=color,
                opacity=opacity,
            )

    def visualize_robot_gaussians(
        self,
        name: str = "RobotGaussian",
        n_std: float = 2.0,
        color=(80, 160, 255),
        opacity: float = 0.35,
    ):
        self._robot_gauss_handles = []
        self._ellipsoid_factory = EllipsoidFactory(n_std=float(n_std))

        i0 = 0
        for local_j, link_idx in enumerate(self.active_link_indices):
            mean = self.midpoints[i0, local_j, :]
            cov = self.gaussian_model.cov(link_idx)
            radii, quat_wxyz = self._ellipsoid_factory.cov_to_ellipsoid(cov)

            h = self.server.scene.add_mesh_simple(
                name=f"{name}_{local_j}",
                vertices=self._create_ellipsoid_mesh(radii),
                faces=self._ellipsoid_faces,
                position=mean,
                wxyz=quat_wxyz,
                color=color,
                opacity=opacity,
            )
            self._robot_gauss_handles.append(h)

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

    def _update_robot_gaussians(self, i: int):
        if not self._robot_gauss_handles:
            return

        for local_j, h in enumerate(self._robot_gauss_handles):
            mean = self.midpoints[i, local_j, :]
            link_idx = self.active_link_indices[local_j]
            cov = self.gaussian_model.cov(link_idx)
            _, quat_wxyz = self._ellipsoid_factory.cov_to_ellipsoid(cov)

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


import time
import numpy as np

from typing import List, Tuple

from viser import ViserServer
from viser.extras import ViserUrdf
from pathlib import Path

from yourdfpy import URDF

from src.core.robot_loader import ManipulatorRobotURDF
from src.visualization.utils import (
    SharedCovariance,
    PerLinkCovariances,
    EllipsoidFactory,
)


class MultipleGaussiansVisualizer:
    def __init__(
        self,
        robot: ManipulatorRobotURDF,
        robot_cov: np.ndarray,  # (3,3) or (n_links,3,3)
        curve: np.ndarray,
        gaussians_per_link: List[Tuple[int, List[float]]],
        ignore_link_indices: list | None = None,
    ):
        self.robot = robot
        self.n_links = self.robot.get_n_links()
        self.n_joints = self.robot.get_n_joints()
        self.curve = curve

        # --- Ignore links ---
        ignore_link_indices = ignore_link_indices or []
        self.ignore_link_indices = sorted(set(ignore_link_indices))

        self.active_link_indices = [
            i for i in range(self.n_links) if i not in self.ignore_link_indices
        ]

        # --- Gaussian model ---
        if robot_cov.shape == (3, 3):
            self.gaussian_model = SharedCovariance(robot_cov)
        elif robot_cov.ndim == 3 and robot_cov.shape == (self.n_links, 3, 3):
            self.gaussian_model = PerLinkCovariances(robot_cov)
        else:
            raise ValueError(
                f"robot_cov must be (3,3) or ({self.n_links},3,3), got {robot_cov.shape}"
            )

        # --- Gaussian specs ---
        self.gaussians_per_link = gaussians_per_link
        self.gaussian_specs = self._build_gaussian_specs(gaussians_per_link)
        self.n_total_gaussians = len(self.gaussian_specs)

        if self.n_total_gaussians == 0:
            raise ValueError("No active gaussian specs were generated.")

        self.gaussian_points = self._compute_gaussian_points(curve)

        # --- Viewer ---
        self.server = ViserServer()
        urdf = URDF.load(self.robot.get_robot_path())
        self.viser_urdf = ViserUrdf(self.server, urdf_or_path=urdf)

        self._ellipsoid_faces = self._create_ellipsoid_faces()
        self._robot_gauss_handles = []
        self._ellipsoid_factory = None

    def _build_gaussian_specs(
        self, gaussians_per_link: List[Tuple[int, List[float]]]
    ) -> List[Tuple[int, float]]:
        specs = []

        for link_idx, t_values in gaussians_per_link:

            if link_idx not in self.active_link_indices:
                continue

            for t in t_values:
                t = float(t)
                specs.append((link_idx, t))

        return specs

    def _compute_gaussian_points(self, curve: np.ndarray) -> np.ndarray:
        num_samples = curve.shape[0]
        points = np.zeros((num_samples, self.n_total_gaussians, 3), dtype=float)

        for i in range(num_samples):
            q = curve[i]
            fk_flat = (
                np.array(self.robot.forward_kinematics(q)).astype(float).reshape(-1)
            )
            joint_positions = fk_flat.reshape(self.n_links + 1, 3)

            for g_idx, (link_idx, t) in enumerate(self.gaussian_specs):
                p0 = joint_positions[link_idx]
                p1 = joint_positions[link_idx + 1]
                points[i, g_idx, :] = (1.0 - t) * p0 + t * p1

        return points

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
        name: str = "Obstacle",
    ):
        factory = EllipsoidFactory(n_std=float(n_std))

        for i, (mean, cov) in enumerate(zip(means, covariances)):
            radii, quat_wxyz = factory.cov_to_ellipsoid(cov)

            self.server.scene.add_mesh_simple(
                name=f"{name}_{i}",
                vertices=self._create_ellipsoid_mesh(radii),
                faces=self._ellipsoid_faces,
                position=mean,
                wxyz=quat_wxyz,
                color=color,
                opacity=opacity,
            )

    def visualize_robot_gaussians(
        self,
        name: str = "RobotGaussian",
        n_std: float = 2.0,
        color=(80, 160, 255),
        opacity: float = 0.35,
    ):
        self._robot_gauss_handles = []
        self._ellipsoid_factory = EllipsoidFactory(n_std=float(n_std))

        i0 = 0
        for g_idx, (link_idx, _) in enumerate(self.gaussian_specs):
            mean = self.gaussian_points[i0, g_idx, :]
            cov = self.gaussian_model.cov(link_idx)
            radii, quat_wxyz = self._ellipsoid_factory.cov_to_ellipsoid(cov)

            h = self.server.scene.add_mesh_simple(
                name=f"{name}_{g_idx}",
                vertices=self._create_ellipsoid_mesh(radii),
                faces=self._ellipsoid_faces,
                position=mean,
                wxyz=quat_wxyz,
                color=color,
                opacity=opacity,
            )
            self._robot_gauss_handles.append(h)

    def _visualize_trajectory_live(self, dt: float, loop: bool):
        self.viser_urdf.update_cfg(np.zeros(self.n_joints))

        num_samples = self.curve.shape[0]
        i = 0

        while True:
            q = self.curve[i]
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

        num_samples = self.curve.shape[0]
        self.viser_urdf.update_cfg(np.zeros(self.n_joints))

        for i in range(num_samples):
            q = self.curve[i]
            self.viser_urdf.update_cfg(q)
            self._update_robot_gaussians(i)
            serializer.insert_sleep(dt)

        data = serializer.serialize()
        Path(recording_path).write_bytes(data)

    def _update_robot_gaussians(self, i: int):
        if not self._robot_gauss_handles or self._ellipsoid_factory is None:
            return

        for g_idx, h in enumerate(self._robot_gauss_handles):
            link_idx, _ = self.gaussian_specs[g_idx]
            mean = self.gaussian_points[i, g_idx, :]
            cov = self.gaussian_model.cov(link_idx)
            _, quat_wxyz = self._ellipsoid_factory.cov_to_ellipsoid(cov)

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
