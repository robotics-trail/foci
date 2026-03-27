"""
Visualization utilities for optimized robot trajectories and Gaussian obstacle models.

This module provides:
- a base visualizer with shared scene setup and animation logic
- a midpoint-based visualizer
- a multi-Gaussian visualizer with arbitrary sampling points along links
"""

import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from viser import ViserServer
from viser.extras import ViserUrdf
from yourdfpy import URDF

from src.core.robot_loader import ManipulatorRobotURDF
from src.visualization.utils import (
    CasadiFKGaussians,
    CasadiFKMidpoints,
    EllipsoidFactory,
    PerLinkCovariances,
    SharedCovariance,
)


class BaseVisualizer:
    """
    Base class for trajectory and Gaussian-model visualization.

    This class manages:
    - the Viser server and robot scene
    - obstacle ellipsoid rendering
    - robot Gaussian rendering
    - live animation and recording export

    Subclasses only need to implement:
    - `_compute_gaussian_points`
    - `_gaussian_link_index`
    """

    def __init__(
        self,
        robot: ManipulatorRobotURDF,
        robot_cov: np.ndarray,
        curve: np.ndarray,
        ignore_link_indices: Optional[List[int]] = None,
    ):
        """
        Parameters
        ----------
        robot : ManipulatorRobotURDF
            Robot model used for visualization.
        robot_cov : np.ndarray
            Robot covariance model. Must have shape `(3, 3)` for shared covariance
            or `(n_links, 3, 3)` for per-link covariance.
        curve : np.ndarray
            Joint trajectory of shape `(num_samples, n_joints)`.
        ignore_link_indices : list[int], optional
            Link indices excluded from the Gaussian visualization.
        """
        curve = np.asarray(curve, dtype=float)
        if curve.ndim != 2:
            raise ValueError(
                f"curve must have shape (num_samples, n_joints), got {curve.shape}"
            )

        self.robot = robot
        self.n_links = self.robot.get_n_links()
        self.n_joints = self.robot.get_n_joints()
        self.curve = curve

        ignore_link_indices = ignore_link_indices or []
        self.ignore_link_indices = sorted(set(ignore_link_indices))
        self.active_link_indices = [
            link_idx
            for link_idx in range(self.n_links)
            if link_idx not in self.ignore_link_indices
        ]
        self.n_active_links = len(self.active_link_indices)

        self.gaussian_model = self._build_gaussian_model(robot_cov)

        self.server = ViserServer()
        urdf = URDF.load(self.robot.get_robot_path())
        self.viser_urdf = ViserUrdf(self.server, urdf_or_path=urdf)

        self._ellipsoid_faces = self._create_ellipsoid_faces()
        self._robot_gauss_handles = []
        self._ellipsoid_factory = None

        self.gaussian_points = self._compute_gaussian_points(curve)
        self.n_total_gaussians = self.gaussian_points.shape[1]

    def _build_gaussian_model(self, robot_cov: np.ndarray):
        """
        Build the covariance accessor used by robot Gaussian rendering.

        Parameters
        ----------
        robot_cov : np.ndarray
            Covariance specification, either shared `(3, 3)` or per-link
            `(n_links, 3, 3)`.

        Returns
        -------
        SharedCovariance or PerLinkCovariances
            Covariance model wrapper.

        Raises
        ------
        ValueError
            If `robot_cov` has an invalid shape.
        """
        if robot_cov.shape == (3, 3):
            return SharedCovariance(robot_cov)

        if robot_cov.ndim == 3 and robot_cov.shape == (self.n_links, 3, 3):
            return PerLinkCovariances(robot_cov)

        raise ValueError(
            f"robot_cov must be (3,3) or ({self.n_links},3,3), got {robot_cov.shape}"
        )

    def _compute_gaussian_points(self, curve: np.ndarray) -> np.ndarray:
        """
        Compute Gaussian center positions for every trajectory sample.

        Parameters
        ----------
        curve : np.ndarray
            Joint trajectory of shape `(num_samples, n_joints)`.

        Returns
        -------
        np.ndarray
            Gaussian points with shape `(num_samples, n_total_gaussians, 3)`.
        """
        raise NotImplementedError

    def _gaussian_link_index(self, gaussian_idx: int) -> int:
        """
        Map one Gaussian index to its associated robot link index.

        Parameters
        ----------
        gaussian_idx : int
            Gaussian index in `[0, n_total_gaussians)`.

        Returns
        -------
        int
            Link index associated with that Gaussian.
        """
        raise NotImplementedError

    def visualize_trajectory(
        self,
        dt: float = 0.1,
        loop: bool = True,
        save_recording: bool = False,
        recording_path: str = "trajectory.viser",
    ):
        """
        Animate the robot trajectory live or save it as a Viser recording.

        Parameters
        ----------
        dt : float, default=0.1
            Time step between consecutive samples.
        loop : bool, default=True
            Whether to loop the live animation indefinitely.
        save_recording : bool, default=False
            If True, export a serialized recording instead of showing a live loop.
        recording_path : str, default="trajectory.viser"
            Output path for the saved recording.
        """
        if save_recording:
            self._save_trajectory_recording(recording_path, dt)
        else:
            self._visualize_trajectory_live(dt, loop)

    def visualize_goal(
        self,
        goal: np.ndarray,
        radius: float = 0.05,
        color: tuple = (0, 0, 255),
    ):
        """
        Add a spherical goal marker to the scene.

        Parameters
        ----------
        goal : np.ndarray
            Goal position of shape `(3,)`.
        radius : float, default=0.05
            Sphere radius.
        color : tuple, default=(0, 0, 255)
            RGB color.
        """
        self.server.scene.add_icosphere(
            name="Goal",
            position=goal,
            radius=radius,
            color=color,
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
        """
        Render Gaussian obstacles as ellipsoids.

        Parameters
        ----------
        means : np.ndarray
            Obstacle centers with shape `(n_obstacles, 3)`.
        covariances : np.ndarray
            Obstacle covariance matrices with shape `(n_obstacles, 3, 3)`.
        n_std : float, default=2.0
            Number of standard deviations used to size each ellipsoid.
        color : tuple, default=(255, 100, 100)
            RGB color.
        opacity : float, default=0.6
            Mesh opacity.
        name : str, default="Obstacle"
            Prefix used to name the rendered meshes.
        """
        factory = EllipsoidFactory(n_std=float(n_std))

        for obstacle_idx, (mean, cov) in enumerate(zip(means, covariances)):
            radii, quat_wxyz = factory.cov_to_ellipsoid(cov)

            self.server.scene.add_mesh_simple(
                name=f"{name}_{obstacle_idx}",
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
        color: tuple = (80, 160, 255),
        opacity: float = 0.35,
    ):
        """
        Render Gaussian ellipsoids attached to the robot.

        The ellipsoids are initialized at the first trajectory sample and later
        updated during animation.

        Parameters
        ----------
        name : str, default="RobotGaussian"
            Prefix used to name the rendered meshes.
        n_std : float, default=2.0
            Number of standard deviations used to size each ellipsoid.
        color : tuple, default=(80, 160, 255)
            RGB color.
        opacity : float, default=0.35
            Mesh opacity.
        """
        self._robot_gauss_handles = []
        self._ellipsoid_factory = EllipsoidFactory(n_std=float(n_std))

        first_sample_idx = 0
        for gaussian_idx in range(self.n_total_gaussians):
            mean = self.gaussian_points[first_sample_idx, gaussian_idx, :]
            link_idx = self._gaussian_link_index(gaussian_idx)

            cov = self.gaussian_model.cov(link_idx)
            radii, quat_wxyz = self._ellipsoid_factory.cov_to_ellipsoid(cov)

            handle = self.server.scene.add_mesh_simple(
                name=f"{name}_{gaussian_idx}",
                vertices=self._create_ellipsoid_mesh(radii),
                faces=self._ellipsoid_faces,
                position=mean,
                wxyz=quat_wxyz,
                color=color,
                opacity=opacity,
            )
            self._robot_gauss_handles.append(handle)

    def visualize_gaussian_splat(
        self,
        name: str,
        means: np.ndarray,
        covariances: np.ndarray,
        colors: np.ndarray,
        opacities: np.ndarray,
    ):
        self.server.add_gaussian_splats(name, means, covariances, colors, opacities)

    def _visualize_trajectory_live(self, dt: float, loop: bool):
        """
        Play the trajectory live in the Viser scene.
        """
        self.viser_urdf.update_cfg(np.zeros(self.n_joints))

        num_samples = self.curve.shape[0]
        sample_idx = 0

        while True:
            q = self.curve[sample_idx]
            self.viser_urdf.update_cfg(q)
            self._update_robot_gaussians(sample_idx)

            time.sleep(dt)

            sample_idx += 1
            if sample_idx >= num_samples:
                if loop:
                    sample_idx = 0
                else:
                    break

    def _save_trajectory_recording(self, recording_path: str, dt: float):
        """
        Save the trajectory animation to a serialized Viser recording.

        Parameters
        ----------
        recording_path : str
            Output file path.
        dt : float
            Time step between consecutive samples.
        """
        serializer = self.server.get_scene_serializer()

        num_samples = self.curve.shape[0]
        self.viser_urdf.update_cfg(np.zeros(self.n_joints))

        for sample_idx in range(num_samples):
            q = self.curve[sample_idx]
            self.viser_urdf.update_cfg(q)
            self._update_robot_gaussians(sample_idx)
            serializer.insert_sleep(dt)

        data = serializer.serialize()
        Path(recording_path).write_bytes(data)

    def _update_robot_gaussians(self, sample_idx: int):
        """
        Update robot Gaussian positions and orientations for one trajectory sample.
        """
        if not self._robot_gauss_handles or self._ellipsoid_factory is None:
            return

        for gaussian_idx, handle in enumerate(self._robot_gauss_handles):
            mean = self.gaussian_points[sample_idx, gaussian_idx, :]
            link_idx = self._gaussian_link_index(gaussian_idx)
            cov = self.gaussian_model.cov(link_idx)
            _, quat_wxyz = self._ellipsoid_factory.cov_to_ellipsoid(cov)

            handle.position = mean
            handle.wxyz = quat_wxyz

    def _create_ellipsoid_mesh(self, radii: np.ndarray, resolution: int = 20):
        """
        Create a triangulated ellipsoid surface centered at the origin.

        Parameters
        ----------
        radii : np.ndarray
            Ellipsoid radii along the principal axes, shape `(3,)`.
        resolution : int, default=20
            Angular sampling resolution.

        Returns
        -------
        np.ndarray
            Vertex array of shape `(n_vertices, 3)`.
        """
        u = np.linspace(0, 2 * np.pi, resolution)
        v = np.linspace(0, np.pi, resolution)

        u_grid, v_grid = np.meshgrid(u, v)

        x = radii[0] * np.cos(u_grid) * np.sin(v_grid)
        y = radii[1] * np.sin(u_grid) * np.sin(v_grid)
        z = radii[2] * np.cos(v_grid)

        return np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)

    def _create_ellipsoid_faces(self, resolution: int = 20):
        """
        Create triangular faces for the ellipsoid mesh grid.

        Parameters
        ----------
        resolution : int, default=20
            Angular sampling resolution used to build the mesh.

        Returns
        -------
        np.ndarray
            Face array of shape `(n_faces, 3)` with dtype `np.uint32`.
        """
        faces = []

        for i in range(resolution - 1):
            for j in range(resolution - 1):
                idx = i * resolution + j
                faces.append([idx, idx + resolution, idx + 1])
                faces.append([idx + 1, idx + resolution, idx + resolution + 1])

        return np.array(faces, dtype=np.uint32)


class Visualizer(BaseVisualizer):
    """
    Visualizer based on one Gaussian per active link midpoint.
    """

    def __init__(
        self,
        robot: ManipulatorRobotURDF,
        robot_cov: np.ndarray,
        curve: np.ndarray,
        ignore_link_indices: Optional[List[int]] = None,
    ):
        """
        Parameters
        ----------
        robot : ManipulatorRobotURDF
            Robot model.
        robot_cov : np.ndarray
            Robot covariance model.
        curve : np.ndarray
            Joint trajectory of shape `(num_samples, n_joints)`.
        ignore_link_indices : list[int], optional
            Link indices excluded from Gaussian rendering.
        """
        self.kinematics = CasadiFKMidpoints(robot, robot.get_n_links())
        super().__init__(robot, robot_cov, curve, ignore_link_indices)

    def _compute_gaussian_points(self, curve: np.ndarray) -> np.ndarray:
        """
        Compute midpoint Gaussian centers for all active links.

        Returns
        -------
        np.ndarray
            Array of shape `(num_samples, n_active_links, 3)`.
        """
        return self.kinematics.midpoints(curve)[:, self.active_link_indices, :]

    def _gaussian_link_index(self, gaussian_idx: int) -> int:
        """
        Map one midpoint Gaussian to its active link index.
        """
        return self.active_link_indices[gaussian_idx]


class MultipleGaussiansVisualizer(BaseVisualizer):
    """
    Visualizer using multiple Gaussian samples along selected links.
    """

    def __init__(
        self,
        robot: ManipulatorRobotURDF,
        robot_cov: np.ndarray,
        curve: np.ndarray,
        gaussians_per_link: List[Tuple[int, List[float]]],
        ignore_link_indices: Optional[List[int]] = None,
    ):
        """
        Parameters
        ----------
        robot : ManipulatorRobotURDF
            Robot model.
        robot_cov : np.ndarray
            Robot covariance model.
        curve : np.ndarray
            Joint trajectory of shape `(num_samples, n_joints)`.
        gaussians_per_link : list[tuple[int, list[float]]]
            Per-link Gaussian sampling specification. Each tuple contains a link
            index and a list of interpolation parameters `t in [0, 1]`.
        ignore_link_indices : list[int], optional
            Link indices excluded from Gaussian rendering.

        Raises
        ------
        ValueError
            If no active Gaussian samples remain after filtering ignored links.
        """
        self.gaussians_per_link = gaussians_per_link
        self.gaussian_specs = self._build_gaussian_specs(
            gaussians_per_link,
            robot.get_n_links(),
            ignore_link_indices,
        )

        if len(self.gaussian_specs) == 0:
            raise ValueError("No active gaussian specs were generated.")

        self.kinematics = CasadiFKGaussians(robot, robot.get_n_links())
        super().__init__(robot, robot_cov, curve, ignore_link_indices)

    @staticmethod
    def _build_gaussian_specs(
        gaussians_per_link: List[Tuple[int, List[float]]],
        n_links: int,
        ignore_link_indices: Optional[List[int]],
    ) -> List[Tuple[int, float]]:
        """
        Flatten grouped Gaussian definitions and discard ignored links.

        Parameters
        ----------
        gaussians_per_link : list[tuple[int, list[float]]]
            Per-link Gaussian sampling specification.
        n_links : int
            Total number of links in the robot chain.
        ignore_link_indices : list[int], optional
            Link indices excluded from Gaussian rendering.

        Returns
        -------
        list[tuple[int, float]]
            Flat list of `(link_idx, t)` pairs.
        """
        ignore_link_indices = sorted(set(ignore_link_indices or []))
        active_link_indices = [
            link_idx
            for link_idx in range(n_links)
            if link_idx not in ignore_link_indices
        ]

        specs = []
        for link_idx, t_values in gaussians_per_link:
            if link_idx not in active_link_indices:
                continue

            for t in t_values:
                specs.append((link_idx, float(t)))

        return specs

    def _compute_gaussian_points(self, curve: np.ndarray) -> np.ndarray:
        """
        Compute all configured Gaussian centers along links.

        Returns
        -------
        np.ndarray
            Array of shape `(num_samples, n_total_gaussians, 3)`.
        """
        return self.kinematics.gaussian_points(curve, self.gaussian_specs)

    def _gaussian_link_index(self, gaussian_idx: int) -> int:
        """
        Map one Gaussian sample to its underlying link index.
        """
        link_idx, _ = self.gaussian_specs[gaussian_idx]
        return link_idx
